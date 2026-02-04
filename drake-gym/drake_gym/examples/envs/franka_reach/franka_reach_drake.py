import os
import sys
# Add current directory (franka_reach/) for local imports (terminations, rewards, etc.)
_THIS_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_MODULE_DIR not in sys.path:
    sys.path.insert(0, _THIS_MODULE_DIR)
# franka_reach.py -> franka_reach/ -> envs/ -> examples/
_EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _EXAMPLES_DIR)
# Also add drake-gym/ for drake_gym module
_DRAKE_GYM_ROOT = os.path.dirname(os.path.dirname(_EXAMPLES_DIR))
sys.path.insert(0, _DRAKE_GYM_ROOT)

import gymnasium as gym
import matplotlib.pyplot as plt

from named_view_helpers import (
    MakeNamedViewActuation,
    MakeNamedViewPositions,
    MakeNamedViewState,
)

import numpy as np

from pydrake.all import (
    AddMultibodyPlant,
    CameraInfo,
    ClippingRange,
    ColorRenderCamera,
    ConstantVectorSource,
    DepthRange,
    DepthRenderCamera,
    DiagramBuilder,
    EventStatus,
    ExternallyAppliedSpatialForce_,
    LeafSystem,
    MakeRenderEngineVtk,
    MeshcatVisualizer,
    MultibodyPlant,
    MultibodyPlantConfig,
    Multiplexer,
    Parser,
    PassThrough,
    PublishEvent,
    RandomGenerator,
    RenderCameraCore,
    RenderEngineVtkParams,
    RgbdSensor,
    RigidTransform,
    RollPitchYaw,
    Simulator,
    SpatialForce,
    StartMeshcat,
)
from pydrake.common.cpp_param import List
from pydrake.common.value import Value
from pydrake.systems.drawing import plot_graphviz, plot_system_graphviz

from drake_gym.drake_gym import DrakeGymEnv

from terminations import *
from functools import partial

from rewards import CompositeReward, reaching_reward
from terminations import *

# Get the path to the models directory
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_THIS_DIR, "..", "..", "..", "models")

# Gym parameters.
sim_time_step = 0.01
gym_time_step = 0.1
# sim_time_step = 0.001
# gym_time_step = 0.01
controller_time_step = 0.01
gym_time_limit = 5
drake_contact_models = ['point', 'hydroelastic_with_fallback']
contact_model = drake_contact_models[0]
drake_contact_solvers = ['sap', 'tamsi']
contact_solver = drake_contact_solvers[0]


def AddAgent(plant):
    parser = Parser(plant)
    model_file = os.path.join(_MODELS_DIR, "franka_description/urdf/panda_arm.urdf")
    print(f"Loading model from: {model_file} \n")
    agent = parser.AddModels(model_file)[0]
    # Weld robot base to world (otherwise it's floating with 7 extra DOFs)
    plant.WeldFrames(plant.world_frame(), 
                     plant.GetFrameByName("panda_link0"))
    return agent

class GoalState():
    """ Shared mutable goal state gets updated on reset (in set_home()) 
        and read by RewardSystem() during sim steps.  """
    def __init__(self):
        self.goal_pos = np.array([0.5, 0.5, 0.3])
        # self.goal_quat = np.array([1., 0., 0., 0.])  # w, x, y, z
        self.goal_r1r2 = np.zeros(6)

def make_sim(generator,
             goal_state,
             meshcat=None,
             time_limit=5,
             debug=False,
             obs_noise=False,
             monitoring_camera=False,
             add_disturbances=False):

    ''' RL goal or parameters '''
    # p_ee_goal_base = [0.5, 0.0, 0.5]
    # rot_ee_goal_base = RollPitchYaw(0, np.pi/2, 0).ToQuaternion()

    builder = DiagramBuilder()

    multibody_plant_config = MultibodyPlantConfig(
        time_step=sim_time_step,
        contact_model=contact_model,
        discrete_contact_approximation=contact_solver,  # renamed in newer Drake
        )

    plant, scene_graph = AddMultibodyPlant(multibody_plant_config, builder)
    plant_compute, scene_graph_compute = AddMultibodyPlant(multibody_plant_config, builder)

    # Add assets to the plant.
    agent = AddAgent(plant)
    plant.Finalize()
    plant.set_name("plant_sim")
    scene_graph.set_name("scene_graph_sim")
    panda_model_instance_compute = AddAgent(plant_compute)
    plant_compute.Finalize()
    plant_compute.set_name("plant_compute")
    scene_graph_compute.set_name("scene_graph_compute")
    plant_compute_context = plant_compute.CreateDefaultContext()

    # Add assets to the controller plant.
    controller_plant = MultibodyPlant(time_step=controller_time_step)
    AddAgent(controller_plant)

    if meshcat:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    # Finalize the plant.
    controller_plant.Finalize()
    controller_plant.set_name("controller_plant")

    # Extract the controller plant information.
    Ns = controller_plant.num_multibody_states() # currently no underactuation
    Nv = controller_plant.num_velocities()
    Na = controller_plant.num_actuators()
    Nj = controller_plant.num_joints()
    Np = controller_plant.num_positions()

    # Make NamedViews
    StateView = MakeNamedViewState(controller_plant, "States")
    PositionView = MakeNamedViewPositions(controller_plant, "Position")
    ActuationView = MakeNamedViewActuation(controller_plant, "Actuation")

    if debug:
        print(f'\nNumber of position: {Np},',
              f'Number of velocities: {Nv},',
              f'Number of actuators: {Na},',
              f'Number of joints: {Nj},',
              f'Number of multibody states: {Ns}')
        print("State view: ", StateView(np.ones(Ns)))
        print("Position view: ", PositionView(np.ones(Np)))
        print("Actuation view: ", ActuationView(np.ones(Na)), '\n')

        # Visualize the plant.
        plt.figure()
        plot_graphviz(plant.GetTopologyGraphvizString())
        plt.plot(1)
        plt.show(block=False)

    ''' In my case, I need the velocities as the actions for impact tasks
        but the choices are 1. joint velocities or 2. ee twists (manifold)
        I think 2 is better for the task as it better describes the pose of
        ee for the task.
        But do option 1 first. Then later implement option 2. Just to 
        compare'''
    

    # joint velocities sent to the agent
    ''' ## when to use Multiplexer()
        - use Multiplexer if there is under-actuation 
        - to combine signals from different sources 
        - to control each joint independently from separate sources '''
    # actuation = builder.AddSystem(Multiplexer([1]*7)) # 7 input ports, 1 for each joint

    ''' if there is under-actuation '''
    # # Zero torque to the revolute joint --it is underactuated.
    # revolute_actuation_torque = builder.AddSystem(ConstantVectorSource([0]))
    # builder.Connect(revolute_actuation_torque.get_output_port(),
    #                 actuation.get_input_port(1))

    ''' PassThrough() for naming (e.g. multiple .sdf in the same plant) and 
        exporting the input port for RL agent.
    '''

    # jvel_actuation_ = builder.AddSystem(PassThrough(7))
    # builder.Connect(jvel_actuation_.get_output_port(),
    #                 plant.get_actuation_input_port(agent)) # agent is a model instance
    
    # # Export for RL agent
    # builder.ExportInput(jvel_actuation_.get_input_port(), "actions_jnt_vel")

    ######################### Velocity Controller
    # Converts desired joint velocities (RL action) to joint torques
    class VelocityTrackingController(LeafSystem):
        """
        PD controller that tracks desired joint velocities.
        
        Computes: τ = Kd * (v_desired - v_current) + gravity_compensation
        
        The gravity compensation term helps maintain position when v_desired = 0.
        """
        def __init__(self, plant, model_instance, Kd=None):
            LeafSystem.__init__(self)
            self.plant = plant
            self.model_instance = model_instance
            self.nv = plant.num_velocities()
            self.nq = plant.num_positions()
            self.na = plant.num_actuators()
            
            # Default gains (tuned for Panda)
            if Kd is None:
                # Higher gains for larger joints
                self.Kd = np.array([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 20.0])
            else:
                self.Kd = np.array(Kd)
            
            # Input ports
            self.DeclareVectorInputPort("desired_velocity", self.nv)
            self.DeclareVectorInputPort("state", self.nq + self.nv)
            
            # Output port for torques
            self.DeclareVectorOutputPort("torque", self.na, self.CalcTorque)
            
            # Create a context for gravity compensation calculations
            self.plant_context = plant.CreateDefaultContext()
        
        def CalcTorque(self, context, output):
            # Get desired velocity from RL action
            v_desired = self.get_input_port(0).Eval(context)
            
            # Get current state
            state = self.get_input_port(1).Eval(context)
            q_current = state[:self.nq]
            v_current = state[self.nq:]
            
            # Velocity error
            v_error = v_desired - v_current
            
            # PD control (D term only for velocity tracking)
            tau_pd = self.Kd * v_error
            
            # Gravity compensation
            self.plant.SetPositions(self.plant_context, q_current)
            tau_gravity = self.plant.CalcGravityGeneralizedForces(self.plant_context)
            
            # Total torque = PD + gravity compensation
            tau = tau_pd - tau_gravity  # negative because gravity is in the opposite direction
            
            # Clamp to effort limits
            effort_limits = np.array([87, 87, 87, 87, 12, 12, 12])
            tau = np.clip(tau, -effort_limits, effort_limits)
            
            output.SetFromVector(tau)
    
    # Create velocity controller
    velocity_controller = builder.AddSystem(
        VelocityTrackingController(plant, agent, Kd=[50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 20.0])
    )
    velocity_controller.set_name("velocity_controller")
    
    # Connect desired velocity input (from RL agent)
    desired_vel_port = builder.ExportInput(
        velocity_controller.get_input_port(0), "actions_jnt_vel"
    )
    
    # Connect current state to controller
    builder.Connect(
        plant.get_state_output_port(),
        velocity_controller.get_input_port(1)
    )
    
    # Connect controller output (torques) to plant actuation
    builder.Connect(
        velocity_controller.get_output_port(),
        plant.get_actuation_input_port(agent)
    )

    ######################### Observer
    class observation_publisher(LeafSystem):
        '''
        For observation, use r1, r2 for learning end-effector orientation. But quaternion is
        used for computing the reward in orientation since no learning there.
        
        :var might: Description
        :var narrow: Description
        :var improves: Description
        :var handle: Description
        '''

        def __init__(self, noise=False, goal_state=None):
            LeafSystem.__init__(self)
            self.Ns = plant.num_multibody_states()
            self.goal_state = goal_state  # Store reference to shared goal
            self.DeclareVectorInputPort("plant_states", self.Ns)
            self.DeclareVectorOutputPort("panda_joint_obs", self.Ns, self.CalcObs1)
            self.DeclareVectorOutputPort("ee_pose_obs", 7, self.CalcObs2)  # pos(3) + [r1, r2](6)
            self.DeclareVectorOutputPort("ee_pose_goal", 7, self.CalcObsGoal)
            self.noise = noise
            
            # Cache frame references
            self.ee_frame = plant.GetFrameByName("panda_link8")
            self.base_frame = plant.world_frame()

        def CalcObs1(self, context, output):
            plant_state = self.get_input_port(0).Eval(context)
            if self.noise:
                plant_state += np.random.uniform(low=-0.01, high=0.01, size=self.Ns)
            output.SetFromVector(plant_state)
        
        def CalcObs2(self, context, output):
            # Get plant state and set positions in plant context
            plant_state = self.get_input_port(0).Eval(context)
            q = plant_state[:plant.num_positions()]
            
            # Use the plant_compute for FK calculations
            plant_compute.SetPositions(plant_compute_context, q)
            X_WE = plant_compute.CalcRelativeTransform(
                plant_compute_context,
                plant_compute.world_frame(),
                plant_compute.GetFrameByName("panda_link8"))
            
            # Extract position and quaternion
            pos = X_WE.translation()
            rx = X_WE.rotation().matrix()[:, 0].flatten()  
            ry = X_WE.rotation().matrix()[:, 1].flatten()
            
            if self.noise:
                pos += np.random.uniform(low=-0.005, high=0.005, size=3)

                rx += np.random.uniform(low=-0.01, high=0.01, size=3)
                ry += np.random.uniform(low=-0.01, high=0.01, size=3)

                # re-normalize r1 and r2 to ensure othogonality after noise addition
                rx /= np.linalg.norm(rx)
                ry /= np.linalg.norm(ry)

                # re-orthogonalize r2 w.r.t. r1
                ry = ry -np.dot(ry, rx) * rx
                ry /= np.linalg.norm(ry)

            ee_pose = np.concatenate([pos, rx, ry])
            output.SetFromVector(ee_pose)

        # TODO: this function not needed for global policy
        def CalcObsGoal(self, context, output):
            goal = np.array(self.goal_state.goal_pos.tolist() + 
                            self.goal_state.goal_r1r2.tolist())
            output.SetFromVector(goal)
        
    obs_pub = builder.AddSystem(observation_publisher(noise=obs_noise, goal_state=goal_state))

    builder.Connect(plant.get_state_output_port(), obs_pub.get_input_port(0))
    builder.ExportOutput(obs_pub.get_output_port(0), "observations_jnt_states")
    builder.ExportOutput(obs_pub.get_output_port(1), "observations_ee_pose")
    builder.ExportOutput(obs_pub.get_output_port(2), "goal_ee_pose")

    class RewardSystem(LeafSystem):
        def __init__(self, Ns, 
                     plant_compute, plant_compute_context, composite_reward,
                     goal_state):
            LeafSystem.__init__(self)
            self.plant = plant_compute
            self.plant_context = plant_compute_context
            self.composite_reward = composite_reward
            self.goal_state = goal_state  # Shared mutable goal
            
            # Get reward component names for output ordering
            self.reward_names = [c['name'] for c in composite_reward.components]
            self.num_rewards = len(self.reward_names)

            self.DeclareVectorInputPort("state", Ns)
            self.DeclareVectorOutputPort("reward", 1, self.CalcReward)

            # Output port for individual reward components (order matches self.reward_names)
            # TODO: probably use abstract vector output for mixed data types for the breakdown
            #       such as dict with names and values: reaching: 0.8
            self.DeclareVectorOutputPort("reward_breakdown", self.num_rewards, self.CalcRewardBreakdown)
        
        def _compute_rewards(self, context):
            """Compute rewards and return (total, breakdown_dict)."""
            state = self.get_input_port(0).Eval(context)

            # set positions in plant context
            qs = state[:self.plant.num_positions()]
            self.plant.SetPositions(self.plant_context, qs)

            total, breakdown = self.composite_reward(
                state=state,
                target_pos=self.goal_state.goal_pos,
                target_r1r2=self.goal_state.goal_r1r2,
                plant=self.plant,
                plant_context=self.plant_context,
            )
            return total, breakdown

        def CalcReward(self, context, output):
            total, _ = self._compute_rewards(context)
            output[0] = total
        
        def CalcRewardBreakdown(self, context, output):
            """Output individual rewards in order of self.reward_names."""
            _, breakdown = self._compute_rewards(context)
            for i, name in enumerate(self.reward_names):
                output[i] = breakdown.get(name, 0.0) 
        
    # Create composite reward with reaching reward function
    composite_reward = CompositeReward()
    composite_reward.add_reward('reaching', reaching_reward, weight=1.0)
    
    reward_system = builder.AddSystem(RewardSystem(
        Ns=Ns,
        plant_compute=plant_compute,
        plant_compute_context=plant_compute_context,
        composite_reward=composite_reward,
        goal_state=goal_state
    ))
    builder.Connect(plant.get_state_output_port(), reward_system.get_input_port(0))
    builder.ExportOutput(reward_system.GetOutputPort("reward"), "reward")
    builder.ExportOutput(reward_system.GetOutputPort("reward_breakdown"), "reward_breakdown")
    # Reward breakdown order: print(composite_reward.components) to see names

    if monitoring_camera:
        # add an overhead camera for video logging of rollout evaluations
        scene_graph.AddRenderer(
            "renderer1", MakeRenderEngineVtk(RenderEngineVtkParams()))
        color_camera = ColorRenderCamera(
            RenderCameraCore(
            "camera1",
            CameraInfo(width=640, 
                       height=480,
                       fov_y=np.pi/4),
            ClippingRange(0.01, 10.0),
            RigidTransform()),
            False
            )
        depth_camera = DepthRenderCamera(color_camera.core(), 
                                         DepthRange(0.01, 10.0))
        parent_id = plant.GetBodyFrameIdIfExists(plant.world_body().index())
        X_PB = RigidTransform(RollPitchYaw(-np.pi/2, 0, 0),
                              np.array([0, -2.5, 0.4]))
        rgbd_camera = builder.AddSystem(RgbdSensor(parent_id=parent_id,
                                                   X_PB=X_PB,
                                                   color_camera=color_camera,
                                                   depth_camera=depth_camera))
        builder.Connect(scene_graph.get_query_output_port(),
                        rgbd_camera.query_object_input_port())
        builder.ExportOutput(rgbd_camera.color_image_output_port(), "camera1_stream") 

    ######################### Disturbance
    class DisturbanceGenerator(LeafSystem):
        def __init__(self, plant, force_mag, period):
            ''' the original example is cartpole, the disturbance force 
                is applied to the cart body along the x direction. What disturbance
                should I added for my case?? Disturbance force applied to the ee? 
                applying a random force [-force_mag, force_mag] at the CoM of the ee link every
                {period} seconds'''
            LeafSystem.__init__(self)
            forces_cls = Value[List[ExternallyAppliedSpatialForce_[float]]]

            # pull-based port (on-demand, e.g., during sim)
            self.DeclareAbstractOutputPort("spatial_forces",
                                           lambda: forces_cls(),
                                           self.CalcDisturbances)
            # push-based event (periodic)
            self.DeclarePeriodicEvent(period_sec=period,
                                      offset_sec=0.0,
                                      event=PublishEvent(
                                      callback=self._on_per_step))
            self.plant = plant
            self.ee_body = self.plant.GetBodyByName("panda_link8") # adapt to my ee link
            self.F = SpatialForce(tau=[0., 0., 0.,],
                                  f=[0., 0., 0.])
            self.force_mag = force_mag                        
        
        def CalcDisturbances(self, context, spatial_forces_vector): # 3rd arg is output
            # apply to the CoM of the ee
            force = ExternallyAppliedSpatialForce_[float]()
            force.body_index = self.ee_body.index()
            force.p_BoBq_B = self.ee_body.default_com()
            force.F_Bq_W = self.F
            spatial_forces_vector.set_value([force])
            self.F = SpatialForce(tau=[0., 0., 0.],
                                  f=[0., 0., 0.])
        
        def _on_per_step(self, context, event):
            self.F = SpatialForce(tau=[0., 0., 0.],
                                  f=[np.random.uniform(-self.force_mag, self.force_mag),   # fx
                                     np.random.uniform(-self.force_mag, self.force_mag),   # fy
                                     np.random.uniform(-self.force_mag, self.force_mag)])  # fz
            
    if add_disturbances:
        # apply a force of 1N every 1s at the CoM of the ee link
        disturbance_generator = builder.AddSystem(
            DisturbanceGenerator(plant=plant, force_mag=1.0, period=.5))
        builder.Connect(disturbance_generator.get_output_port(),
                        plant.get_applied_spatial_force_input_port()) # this is new to me!
    
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.Initialize()
            
    ######################### Episode termination
    # TODO: instantiate CompositeTermination()
    # Joint limits for Franka Panda (from URDF, in radians)
    q_max = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]  # positive limits
    q_min = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
    v_max = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.61, 2.61, 2.61]) # panda hardware limits
    # u_up = np.array([87, 87, 87, 87, 12, 12, 12])

    termination_checker = CompositeTermination()
    termination_checker.add_termination('time_limit', 
        lambda **kw: time_limit_termination(**kw, time_limit=time_limit), 
        is_success=False)
    termination_checker.add_termination('ee_position_goal_reached', 
        lambda **kw: ee_pose_goal_reached_termination(**kw, ep_threshold=0.1, eq_threshold=0.35), 
        is_success=True)
    termination_checker.add_termination('joint_limits', 
        lambda **kw: joint_limit_termination(**kw, q_min=q_min, q_max=q_max), 
        is_success=False)


    # Episode end conditions:
    def monitor(context, state_view=StateView):
        plant_context = plant.GetMyContextFromRoot(context)
        state = plant.GetOutputPort("panda_state").Eval(plant_context)
        s = state_view(state)

        # print cleanly
        t = context.get_time()
        qs = [round(getattr(s, f"panda_joint{i}_q").item(), 3) for i in range(1, 8)]
        vs = [round(getattr(s, f"panda_joint{i}_w").item(), 3) for i in range(1, 8)]

        # TODO: change to "dummy1" later after welding finger to ee_link                    
        plant_compute.SetPositions(plant_compute_context, qs)
        ee_frame = plant_compute.GetFrameByName("panda_link8")
        ee_pos = ee_frame.CalcPoseInWorld(plant_compute_context).translation()
        ee_quat = ee_frame.CalcPoseInWorld(plant_compute_context).rotation().ToQuaternion().wxyz()

        # Check all conditions
        triggered, reason, is_success = termination_checker(t=t,
                                                            qs=qs,
                                                            vs=vs,
                                                            ee_pos=ee_pos,
                                                            ee_quat=ee_quat,
                                                            target_pos=goal_state.goal_pos,
                                                            target_r1r2=goal_state.goal_r1r2
                                                            )
        
        if triggered:
            if debug:
                print(f"Episode terminated due to: {reason}")
            return EventStatus.ReachedTermination(diagram, reason)

    simulator.set_monitor(monitor)
    
    return simulator

def set_home(simulator, diagram_context, seed, goal_state):
    ''' An interface for domain and goal randomization 
        goal_state: an mutable object to store the goal position and orientation
                    it got randomized in each reset that calls set_home()
    '''
    # print(f"set_home called! New goal: {goal_state.goal_pos}")

    # Joint limits for Franka Panda (from URDF, in radians)
    q_max = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]  # positive limits
    q_min = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
    v_max = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.61, 2.61, 2.61]) # panda hardware limits
    # u_up = np.array([87, 87, 87, 87, 12, 12, 12])

    # randomize the q_init centred around some nominal home_position
    q_home = [0.9207,  0.2574, -0.9527, -2.0683,  0.2799,  2.1147, 2.]
    offset = 0.5 # rad 
    home_positions = {
        'panda_joint1' : np.random.uniform(low=q_home[0]-offset, high=q_home[0]+offset),
        'panda_joint2' : np.random.uniform(low=q_home[1]-offset, high=q_home[1]+offset),
        'panda_joint3' : np.random.uniform(low=q_home[2]-offset, high=q_home[2]+offset),
        'panda_joint4' : np.random.uniform(low=q_home[3]-offset, high=q_home[3]+offset),
        'panda_joint5' : np.random.uniform(low=q_home[4]-offset, high=q_home[4]+offset),
        'panda_joint6' : np.random.uniform(low=q_home[5]-offset, high=q_home[5]+offset),
        'panda_joint7' : np.random.uniform(low=q_home[6]-offset, high=q_home[6]+offset),
    }

    ''' No need to do this if TO always start from static. But there are benefits of doing it:
        1. If your robot might:
             -start an episode while already moving,
             - be bumped or perturbed,
             - have to reach again while not fully at rest,
             - be commanded repeatedly during real use (back-to-back tasks),
           then randomizing initial velocities helps dramatically.

        2. SAC learns better if the state distribution is diverse.
            If initial states are too narrow:
              - the robot only learns to handle a very narrow part of the state space,
              - the critic overfits,
              - the policy becomes brittle,
              - exploration collapses.
            A bit of velocity randomization improves:
              - value function generalization
              - critic stability
              - robustness to modeling errors

        3. You want the global policy to work from ANY feasible state
            If your goal-conditioned policy should handle:
              - random resets,
              - dynamic starts,
              - states out of nominal manifold,
            then randomizing q̇ is essential.

        4. You want to imitate TO trajectories AND learn robustness
             If TO always starts with q̇ = 0, but RL should be robust in deployment, your recipe is:
              - BC from TO: train on trajectories starting at rest
              - RL exploration: randomize q̇ in a safe range
              - This helps you get TO-level optimality but also RL-level robustness.
    '''
    # randomize the v_init
    home_velocities = {
        'panda_joint1' : np.random.uniform(low=-0.1*v_max[0], high=0.1*v_max[0]),
        'panda_joint2' : np.random.uniform(low=-0.1*v_max[1], high=0.1*v_max[1]),
        'panda_joint3' : np.random.uniform(low=-0.1*v_max[2], high=0.1*v_max[2]),
        'panda_joint4' : np.random.uniform(low=-0.1*v_max[3], high=0.1*v_max[3]),
        'panda_joint5' : np.random.uniform(low=-0.1*v_max[4], high=0.1*v_max[4]),
        'panda_joint6' : np.random.uniform(low=-0.1*v_max[5], high=0.1*v_max[5]),
        'panda_joint7' : np.random.uniform(low=-0.1*v_max[6], high=0.1*v_max[6]),
    }

    diagram = simulator.get_system()
    plant = diagram.GetSubsystemByName("plant_sim")
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

    # # Clip the positions (I may not need this but for keeping generality)
    # for joint_name, q in home_positions.items():
    #     joint = plant.GetJointByName(joint_name)
    #     joint.set_angle(plant_context,
    #                     np.clip(q,
    #                             joint.position_lower_limit(),
    #                             joint.position_upper_limit()))
    
    # Randomize other params:
    # e.g. friction, mass, link dimensions, gravity, etc.
    # randomize mass 
    # objects_mass_offset = {
    #     "soup_base_link": np.random.uniform(low=-0.2, high=0.2)
    #     }
    # for link_name, mass_offset in objects_mass_offset.items():
    #     body = plant.GetBodyByName(link_name)
    #     # why creating a new default context here? so mass default is always the same?
    #     mass = body.get_mass(plant.CreateDefaultContext()) 
    #     body.SetMass(plant_context, mass+mass_offset)

    ''' Randomize the target position using FK on a random config '''
    # 1. First, sample a random config for the GOAL (not the robot's starting config)
    q_goal = np.random.uniform(q_min, q_max)
    plant.SetPositions(plant_context, q_goal)
    ee_frame = plant.GetFrameByName("panda_link8")
    p_ee_goal = ee_frame.CalcPoseInWorld(plant_context).translation()

    rot_ee_goal = ee_frame.CalcPoseInWorld(plant_context).rotation().matrix()
    rx_goal = rot_ee_goal[:, 0].flatten()  
    ry_goal = rot_ee_goal[:, 1].flatten()

    goal_state.goal_pos = p_ee_goal
    goal_state.goal_r1r2 = np.concatenate([rx_goal, ry_goal])

    # 2. Now set the robot to its STARTING position (different from goal!)
    q_init = np.array([home_positions[f'panda_joint{i+1}'] for i in range(7)])
    # v_init = np.array([home_velocities[f'panda_joint{i+1}'] for i in range(7)])
    v_init = np.zeros(7) # start from static for now
    plant.SetPositions(plant_context, q_init)
    plant.SetVelocities(plant_context, v_init)

def PandaReachEnv(observations="state",
                  meshcat=None,
                  time_limit=gym_time_limit,
                  debug=False,
                  obs_noise=False,
                  monitoring_camera=False,
                  add_disturbances=False,
                  device='cpu'):
    
    # create goal state for randomized goals to be shared between RewardSystem()
    # and set_home()
    goal_state = GoalState()
    
    # make simulation
    simulator = make_sim(generator=RandomGenerator(),
                         goal_state=goal_state,
                         meshcat=meshcat,
                         time_limit=time_limit,
                         debug=debug,
                         obs_noise=obs_noise,
                         monitoring_camera=monitoring_camera,
                         add_disturbances=add_disturbances)
    
    plant_sim = simulator.get_system().GetSubsystemByName("plant_sim")

    # Define action space (always use float32 - sufficient for RL and works with MPS)
    Na = 7 # currently, only joint velocities as actions
    low_a  = plant_sim.GetVelocityLowerLimits()[:Na]  # velocity limits are typically symmetric
    high_a = plant_sim.GetVelocityUpperLimits()[:Na]
    action_space = gym.spaces.Box(
        low=np.array(low_a, dtype=np.float32),
        high=np.array(high_a, dtype=np.float32),
        dtype=np.float32
    )

                        
    # Define observation space (always use float32 - sufficient for RL and works with MPS)
    # Use -inf/inf bounds to avoid "obs not in observation space" warnings
    # The simulation can produce values slightly outside physical limits
    obs_dim = len(plant_sim.GetPositionLowerLimits()) + len(plant_sim.GetVelocityLowerLimits())
    observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(obs_dim,),
        dtype=np.float32
    )
    
    env = DrakeGymEnv(
        simulator=simulator,
        time_step=gym_time_step,
        action_space=action_space,
        observation_space=observation_space,
        reward="reward", # will change later as I progress
        action_port_id="actions_jnt_vel", # will change later as I progress
        observation_port_id="observations_jnt_states",
        set_home=partial(set_home, goal_state=goal_state),
        render_rgb_port_id="camera1_stream" if monitoring_camera else None)
    
    # Expose parameters that could be useful for learning
    env.time_step = gym_time_step
    env.sim_time_step = sim_time_step
    env.goal_state = goal_state  # Expose for visualization in test scripts

    return env

if __name__ == "__main__":
    meshcat = StartMeshcat()
    env = PandaReachEnv(meshcat=meshcat,
                        debug=True,
                        obs_noise=True,
                        monitoring_camera=True,
                        add_disturbances=True)

    input("Open Meshcat URL in browser, then press Enter...")

    obs = env.reset()
    # done = False
    terminated = False
    truncated = False

    meshcat.StartRecording()
    while not terminated and not truncated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Access reward breakdown from the diagram port
        diagram = env.simulator.get_system()
        context = env.simulator.get_context()
        
        ''' This is a way to get exported ports from the diagram '''
        reward_breakdown = diagram.GetOutputPort("reward_breakdown").Eval(context)
        # Reward names order: ['reaching'] (add more as you add rewards)
        
        print(f"obs: \n {obs}")
        print(f"reward: {reward:.3f}, breakdown: {reward_breakdown}")
        print(f"terminated: {terminated}, truncated: {truncated} \n")
        # env.render(mode='human') # don't use this during training!

    meshcat.PublishRecording()
    env.close()