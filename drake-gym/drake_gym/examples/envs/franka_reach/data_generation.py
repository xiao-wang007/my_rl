"""
Docstring for drake-gym.drake_gym.examples.envs.franka_reach.data_generation
In this module, I ran direction collocation for reaching-task to gather data for BC as well as RL

data is in this form:
[state_t, action_t, nextState_t, reward_t, done_t] for RL

[---------state_t/observation----------, -action_t-] for BC
[(q_t, v_t, ee_xyz_t, ee_rot6d_t, goal, phi_strike, time_since_contact, contact_flag), (q_t, v_t)] 
phi for timing variable, at 0.5T ee should reach the target pose at desired v_ee, at T should be 
at rest at a target pose. Use this link as a reference: https://chatgpt.com/s/t_694c776e7ab48191b655b8455fd856a0

Data generation module:
    Input: q_init, v_init, ee_pose_SE3_target, v_ee_target
Option A:
    Output: trajectory structured data, which stores the trajectories from the TO, that CAN BE TURN
            INTO THE FLAT TRANSITION DATA LATER. This is good for debugging and visualization and time
            indexed policies (MUST). 
Option B:
    Output: data as above stored in a dict() as flat transition data, with each row being a 
            transition, i.e. [state_t/observation_t, action_t, nextState/observation_t, reward_t,
            done_t] (OPTIONAL)

Then use h5py in the pipeline to store the data into hdf5 files.

# TODO:
# TO + BC + residual-RL strategy with shape/contact adaptiveness: https://chatgpt.com/s/t_694c7a0a9cc48191a0d52c810d162f7e
  TO + BC as motion prior
  RL as improvement for robustness and adaptiveness

# online system ID to bridge sim2real gap: https://chatgpt.com/s/t_694c7cd279ec819196d87624fecdeb08

# vanilla SAC is not ideal for my case, here is why: https://chatgpt.com/s/t_694c80173aa481918312d181bdbe3820
  in which it suggests that RL should output: velocity scale factor, or an impulse modulation gain, 
  or small residual near contact only; instead of full joint targets.

# Better options than vanilla SAC: 
    - TD3 + BC: works well with tiny dataset
    - SAC+BC / SACfD
    - AWAC
    - IQL

# TODO: the recommanded implementation: https://chatgpt.com/s/t_694c8270985c8191955f29b4d4da4540
    - the timing variable phi(t): https://chatgpt.com/s/t_694d3fd069308191bb35bde1810654a2
    - how to use phi(t) in RL rewards: https://chatgpt.com/s/t_694d4005c50c8191a094d748cac534e7
    - why a contact_flag in the observation is useful: https://chatgpt.com/s/t_694d48f360688191a72cf8068928541c
    - use both raw TO data and tracked sim data for collection: https://chatgpt.com/s/t_694d4b9a2b588191ab415dd62b68702d
    - only do online fine-tuning with RL after thoroughly debugged offline RL and there is a demand: 
        https://chatgpt.com/s/t_695464499a788191814c3f2f7ad63801
    - the 'exploration' in RL means differently in my case: https://chatgpt.com/s/t_695469db9c1c8191b7b314ede9f47d48

The best algorithm is the one whose inductive bias matches your problem.
"""

import numpy as np
from pydrake.all import (DiagramBuilder)
from constraints import *
from utils import MakeArmOnlyPlant

nq = 7
nv = 7
nu = 7

def build_prog(prog, N, ):

    q_arm_vars = prog.NewContinuousVariables(N, 7, "joint_pos")
    v_arm_vars = prog.NewContinuousVariables(N, 7, "joint_vel")
    u_vars = prog.NewContinuousVariables(N, 7, "tau")
    ln_vars = prog.NewContinuousVariables(1, "lambda_N")
    lt_var = prog.NewContinuousVariables(1, "lambda_t")

    v1_post_vars = prog.NewContinuousVariables(2, "obj1_v_post")
    w1_post_vars = prog.NewContinuousVariables(1, "obj1_w_post")

    ds_var = prog.NewContinuousVariables(1, "ds")
    dtheta_var = prog.NewContinuousVariables(1, "dtheta")

    h_vars = prog.NewContinuousVariables(N-1, "time step" )

    dtheta_abs_var = prog.NewContinuousVariables(1, "dtheta_abs")
    # enforce the sign of dtheta
    prog.AddLinearConstraint(dtheta_abs_var[0] >= dtheta_var[0])
    prog.AddLinearConstraint(dtheta_abs_var[0] >= -dtheta_var[0])
    prog.AddBoundingBoxConstraint(0, np.inf, dtheta_abs_var)

    # panda joint limits
    q_robot_up = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    q_robot_low = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    v_robot_up = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.61, 2.61, 2.61]) # panda hardware limits
    v_robot_low = -v_robot_up
    u_up = np.array([87, 87, 87, 87, 12, 12, 12])
    u_low = -u_up

    # dtau
    dtau_up = np.array([[dtau_scalar]]*7)
    dtau_low = -dtau_up

    # bounds for segment 1
    prog.AddBoundingBoxConstraint(np.array([q_robot_low]*(t_impact-1)), 
                                np.array([q_robot_up]*(t_impact-1)), 
                                q_arm_vars[1:t_impact, :])

    prog.AddBoundingBoxConstraint(np.array([v_robot_low]*(t_impact-1)),
                                np.array([v_robot_up]*(t_impact-1)), 
                                v_arm_vars[1:t_impact, :])

    prog.AddBoundingBoxConstraint(np.array([u_low]*(t_impact-1)), 
                                np.array([u_up]*(t_impact-1)),
                                u_vars[1:t_impact, :])

    # bounds for segment 3
    prog.AddBoundingBoxConstraint(np.array([q_robot_low]*(N-t_impact-1)), 
                                np.array([q_robot_up]*(N-t_impact-1)), 
                                q_arm_vars[t_impact+1:, :])

    prog.AddBoundingBoxConstraint(np.array([v_robot_low]*(N-t_impact-1)),
                                np.array([v_robot_up]*(N-t_impact-1)), 
                                v_arm_vars[t_impact+1:, :])

    prog.AddBoundingBoxConstraint(np.array([u_low]*(N-t_impact-1)), 
                                np.array([u_up]*(N-t_impact-1)),
                                u_vars[t_impact+1:, :])

    fn_low = 0.
    fn_up = 50.
    prog.AddBoundingBoxConstraint(fn_low,
                                fn_up,
                                ln_vars)

    ft_low = -0.5*fn_up
    # ft_low = -3.0
    # ft_low = 1.0
    ft_up = 0.5*fn_up
    # ft_up = -1.0
    # ft_up = 5.0
    prog.AddBoundingBoxConstraint(ft_low,
                                ft_up,
                                lt_var)
                                
    h_low = np.array([h_scalar_low]*(N-1))
    h_high = np.array([h_scalar_high]*(N-1))
    prog.AddBoundingBoxConstraint(h_low, h_high, h_vars)

    # ds and dtheta bounds
    prog.AddBoundingBoxConstraint(-2*3.14, 2*3.14, dtheta_var)
    # prog.AddBoundingBoxConstraint(-10., 10., w1_post_vars)

    if vhat[0] > 0:
        prog.AddBoundingBoxConstraint(0., 5., v1_post_vars[0])
    if vhat[1] > 0:
        prog.AddBoundingBoxConstraint(0., 5., v1_post_vars[1])
    if vhat[0] < 0:
        prog.AddBoundingBoxConstraint(-5., 0., v1_post_vars[0])
    if vhat[1] < 0:
        prog.AddBoundingBoxConstraint(-5., 0., v1_post_vars[1])

    # prog.AddBoundingBoxConstraint(-5., 0., v1_post_vars[0])
    # prog.AddBoundingBoxConstraint(0., 5., v1_post_vars[1])

    # arm init and end
    q0_arm = start_pose_panda
    v0_arm = np.zeros(7)
    vf_arm = np.zeros(nu)

    ## fix init
    prog.AddBoundingBoxConstraint(q0_arm, q0_arm, q_arm_vars[0, :])
    # prog.AddBoundingBoxConstraint(end_pose_panda, end_pose_panda, q_arm_vars[-1, :])
    prog.AddBoundingBoxConstraint(v0_arm, v0_arm, v_arm_vars[0, :])
    prog.AddBoundingBoxConstraint(vf_arm, vf_arm, v_arm_vars[-1, :])

    # fix u0, which is just tau_g to hold the arm up there
    plant_double.SetPositionsAndVelocities(double_context, np.concatenate((q0_arm, np.zeros(7)))) # using the arm-only plant
    u0 = plant_double.CalcGravityGeneralizedForces(double_context)
    prog.AddBoundingBoxConstraint(u0, u0, u_vars[0, :])



class DataGenerator():
    def __init__(self, T, N, q_init, ee_target_SE3, v_ee, v_init=np.zeros(nv), split_ratio=0.5):
        self.T = T
        self.N = N
        self.t_index_strike = int(split_ratio * N)  
        self.dt = T / (N - 1)
        self.q_init = q_init
        self.v_init = v_init
        self.ee_target_SE3 = ee_target_SE3
        self.v_ee = v_ee
        self.data_typeA = None
        self.data_typeB = None
        self.prog = None

        # this comes from sim or some contact detection in real robot
        self.t_index_detected_contact = None

        # build the plant here
        self.builder = DiagramBuilder()
        self.plant, self.scene_graph, self.arm = MakeArmOnlyPlant(self.builder, self.dt)
        self.plant_AD = self.plant.ToScalarType[AutoDiffXd]() # convert to AD
        self.context_AD= self.plant_AD.CreateDefaultContext()
    
    def _build_TO(self):
        # init prog and define decVars

        # decVar bounds

        # constraints

        # costs
        pass

    def _solve_TO(self):
        pass

    def _extract_solution(self):
        pass

    def _close_loop_tracking(self):
        pass
    
    def _get_contact_time_index(self) -> int:
        pass

    def _compute_phi_and_timeRemain(self, t_index: int) -> float:
        ''' Compute the phase variable phi(t) with a bound [0, 1] using smooth Gaussian kernel 
            to produce weighting function over time. '''
        # phi is indexing relative to the whole horizon
        phi = t_index / (self.N - 1)
        phi = np.clip(phi, 0.0, 1.0)

        # phi_hit is indexing relative to the contact time as per split ratio
        phi_hit = (t_index - self.t_index_strike) / (self.N - 1)

        # time since contact
        if self.t_index_detected_contact:
            time_since_contact = (t_index - self.t_index_detected_contact) * self.dt
        else:
            time_since_contact = 0.0

        return phi, phi_hit, time_since_contact

    def _build_data(self):
        pass
    
    def _save_data_hdf5(self, filename: str):
        pass

    def generate(self):
        self._build_TO()
        self._solve_TO()
        self._extract_solution()
        self._build_data()
        # self._save_data_hdf5(filename="data.h5")

if __name__ == "__main__":
    dg = DataGenerator(T=2.0, N=21,
                       q_init=np.zeros(nq),
                       ee_target_SE3=None,
                       v_ee=np.zeros(6))

