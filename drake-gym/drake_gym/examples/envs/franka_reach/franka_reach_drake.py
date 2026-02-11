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
from pydrake.visualization import AddFrameTriadIllustration

from drake_gym.drake_gym import DrakeGymEnv

from functools import partial
from controller_systems import VelocityTrackingController, ActionScaler
from rewards import (CompositeReward, reaching_position, reaching_orientation, 
                     reaching_terminal, acceleration_smoothness)
from terminations import (CompositeTermination, time_limit_termination, 
                          ee_position_reached_termination, ee_orientation_reached_termination, 
                          joint_limit_termination)
from rl_systems import (ObserverSystem, DisturbanceGenerator, RewardSystem)

# Get the path to the models directory
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_THIS_DIR, "..", "..", "..", "models")

# Gym parameters.
sim_time_step = 0.01
gym_time_step = 0.1
# sim_time_step = 0.001
# gym_time_step = 0.01
controller_time_step = 0.01
gym_time_limit = 10 # in seconds
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
             add_disturbances=False,
             v_max_scale=1.0):

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
    
    # Add EE frame triad illustration (will be recorded with simulation)
    AddFrameTriadIllustration(
        scene_graph=scene_graph,
        plant=plant,
        body=plant.GetBodyByName("panda_link8"),
        length=0.1,
        radius=0.004,
        opacity=0.3,
    )
    
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

    # Action scaler: maps RL actions [-1, 1] -> physical velocities
    # v_max_scale ∈ (0, 1] caps max velocity so the agent learns slow motions first
    v_max_hardware = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.61, 2.61, 2.61])
    v_max = v_max_scale * v_max_hardware
    action_scaler = builder.AddSystem(ActionScaler(v_max))
    action_scaler.set_name("action_scaler")

    # Export normalized action input for RL agent
    builder.ExportInput(
        action_scaler.get_input_port(0), "actions_jnt_vel"
    )

    # Create velocity controller
    velocity_controller = builder.AddSystem(
        VelocityTrackingController(plant, agent, Kd=[30.0, 30.0, 30.0, 30.0, 5.0, 5.0, 5.0])
    )
    velocity_controller.set_name("velocity_controller")

    # Scaler -> Controller -> Plant
    builder.Connect(
        action_scaler.get_output_port(),
        velocity_controller.get_input_port(0)
    )
    builder.Connect(
        plant.get_state_output_port(),
        velocity_controller.get_input_port(1)
    )
    builder.Connect(
        velocity_controller.get_output_port(),
        plant.get_actuation_input_port(agent)
    )

    obs_pub = builder.AddSystem(ObserverSystem(plant_sim=plant, plant_compute=plant_compute, 
                                               plant_compute_context=plant_compute_context, 
                                               noise=obs_noise, goal_state=goal_state))

    builder.Connect(plant.get_state_output_port(), obs_pub.get_input_port(0))
    builder.ExportOutput(obs_pub.get_output_port(0), "observations_jnt_states")
    builder.ExportOutput(obs_pub.get_output_port(1), "observations_ee_pose")
    builder.ExportOutput(obs_pub.get_output_port(2), "goal_ee_pose")

    # Create composite reward with reaching reward function
    composite_reward = CompositeReward()
    composite_reward.add_reward('reaching position', 
                                partial(reaching_position, coeff=10.0), weight=1.0)
    # composite_reward.add_reward('reaching orientation', reaching_orientation, weight=1.0)
    composite_reward.add_reward('reaching terminal', 
                                partial(reaching_terminal, 
                                        epsilon_pos=0.05, epsilon_ori=10.0), # in degrees 
                                weight=1.0)
    composite_reward.add_reward('smoothness',
                                partial(acceleration_smoothness, 
                                        dt=gym_time_step, coeff=5e-3),
                                weight=1.0)
    
    reward_system = builder.AddSystem(RewardSystem(
        Ns=Ns,
        gym_time_step=gym_time_step,
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
        lambda **kw: ee_position_reached_termination(**kw, ep_threshold=0.04), 
        is_success=True)
    # termination_checker.add_termination('ee_orientation_goal_reached', 
    #     lambda **kw: ee_orientation_reached_termination(**kw, eq_threshold=0.35), 
    #     is_success=True)
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
    # q_goal = np.random.uniform(q_min, q_max)
    # rng = np.random.default_rng(seed)
    # q_goal = rng.uniform(low=q_min, high=q_max)

    # fix the goal 
    q_goal = np.array([1.1207,  0.3074, -0.9527, -2.0683,  0.2799,  2.1147, 2.])
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
                  device='cpu',
                  v_max_scale=1.0):
    
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
                         add_disturbances=add_disturbances,
                         v_max_scale=v_max_scale)
    
    plant_sim = simulator.get_system().GetSubsystemByName("plant_sim")

    # Define action space (always use float32 - sufficient for RL and works with MPS)
    Na = 7 # currently, only joint velocities as actions
    # low_a  = plant_sim.GetVelocityLowerLimits()[:Na]  # velocity limits are typically symmetric
    # high_a = plant_sim.GetVelocityUpperLimits()[:Na]

    low_a = -1.0 * np.ones(Na)
    high_a = 1.0 * np.ones(Na)
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

    obs, info = env.reset(seed=42)
    # done = False
    terminated = False
    truncated = False

    meshcat.StartRecording()
    while not terminated and not truncated:
        # sample actions
        action = env.action_space.sample()

        # # zero actions
        # action = np.zeros_like(action)

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