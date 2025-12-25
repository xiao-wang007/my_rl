"""
Docstring for drake-gym.drake_gym.examples.envs.franka_reach.data_generation
In this module, I ran direction collocation for reaching-task to gather data for BC as well as RL

data is in this form:
[state_t, action_t, nextState_t, reward_t, done_t] for RL

[---------state_t/observation----------, -action_t-] for BC
[(q_t, v_t, ee_xyz_t, ee_rot6d_t, goal, phi), (q_t, v_t)] # phi for timing variable, at 0.5T ee should
reach the target pose at desired v_ee, at T should be at rest at a target pose. Use this link as a 
reference: https://chatgpt.com/s/t_694c776e7ab48191b655b8455fd856a0

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

The best algorithm is the one whose inductive bias matches your problem.
"""

import numpy as np
from pydrake.all import (DiagramBuilder)
from constraints import *
from utils import MakeArmOnlyPlant


nq = 7
nv = 7
nu = 7

class DataGenerator():
    def __init__(self, T, N, q_init, ee_target_SE3, v_ee, v_init=np.zeros(nv)):
        self.T = T
        self.N = N
        self.dt = T / (N - 1)
        self.q_init = q_init
        self.v_init = v_init
        self.ee_target_SE3 = ee_target_SE3
        self.v_ee = v_ee
        self.data_typeA = None
        self.data_typeB = None
        self.prog = None

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

