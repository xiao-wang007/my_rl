import gym
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
    MeshcatVisualizerCpp,
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
)
from pydrake.common.cpp_param import List
from pydrake.common.value import Value
from pydrake.systems.drawing import plot_graphviz, plot_system_graphviz

from anzu.common.cc import FindAnzuResourceOrThrow
from drake_gym.drake_gym import DrakeGymEnv

# Gym parameters.
sim_time_step = 0.01
gym_time_step = 0.05
controller_time_step = 0.01
gym_time_limit = 5
drake_contact_models = ['point', 'hydroelastic_with_fallback']
contact_model = drake_contact_models[0]
drake_contact_solvers = ['sap', 'tamsi']
contact_solver = drake_contact_solvers[0]


def AddAgent(plant):
    parser = Parser(plant)
    model_file = FindAnzuResourceOrThrow(
        "drake_gym/models/franka_description/urdf/panda_arm.urdf")
    agent = parser.AddModelFromFile(model_file)
    return agent


def make_sim(generator,
             meshcat=None,
             time_limit=5,
             debug=False,
             obs_noise=False,
             monitoring_camera=False,
             add_disturbances=False):

    ''' RL goal or parameters '''
    p_ee_goal_base = [0.5, 0.0, 0.5]
    rot_ee_goal_base = RollPitchYaw(0, np.pi/2, 0).ToQuaternion()

    builder = DiagramBuilder()

    multibody_plant_config = MultibodyPlantConfig(
        time_step=sim_time_step,
        contact_model=contact_model,
        discrete_contact_solver=contact_solver,
        )

    plant, scene_graph = AddMultibodyPlant(multibody_plant_config, builder)
    plant_compute, scene_graph_compute = AddMultibodyPlant(multibody_plant_config, builder)

    # Add assets to the plant.
    agent = AddAgent(plant)
    plant.Finalize()
    plant.set_name("plant_sim")
    panda_model_instance_compute = AddAgent(plant_compute)
    plant_compute.Finalize()
    plant_compute.set_name("plant_compute")
    plant_compute_context = plant.CreateDefaultContext()

    # Add assets to the controller plant.
    controller_plant = MultibodyPlant(time_step=controller_time_step)
    AddAgent(controller_plant)

    if meshcat:
        MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)

    # Finalize the plant.
    controller_plant.Finalize()
    controller_plant.set_name("controller_plant")

    # Extract the controller plant information.
    Ns = controller_plant.num_multibody_states()
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

    jvel_actuation_ = builder.AddSystem(PassThrough(7))
    builder.Connect(jvel_actuation_.get_output_port(),
                    plant.get_actuation_input_port(agent)) # agent is a model instance
    
    # Export for RL agent
    builder.ExportInput(jvel_actuation_.get_input_port(), "actions_jnt_vel")

    ######################### Observer
    class observation_publisher(LeafSystem):
        def __init(self, noise=False):
            LeafSystem.__init__(self)
            self.Ns = plant.num_multibody_states()
            self.DeclareVectorInputPort("plant_states", self.Ns)
            self.DeclareVectorOutputPort("observations", self.Ns, self.CalcObs)
            self.noise = noise 

        def CalcObs(self, context, output):
            plant_state = self.get_input_port(0).Eval(context)
            if self.noise:
                plant_state += np.random.uniform(low=-0.01, high=0.01, size=self.Ns)
        
            output.SetFromVector(plant_state)
        
    obs_pub = builder.AddSystem(observation_publisher(noise=obs_noise))

    builder.Connect(plant.get_state_output_port(), obs_pub.get_input_port(0))
    builder.ExportOutput(obs_pub.get_output_port(0), "observations_jnt_states")

    class RewardSystem(LeafSystem):
        def __init__(self):
            LeafSystem.__init__(self)
            self.DeclareVectorInputPort("state", Ns)
            self.DeclareVectorOutputPort("reward", 1, self.CalcReward)
            self.DeclareVectorOutputPort("safety_reward", 1, self.CalcSafetyReward)
        
        def CalcReward(self, context, output):
            reward = 1
            output[0] = reward
        
        def CalcSafetyReward(self, context, output):
            # adapt task constraints here
            pass 
        
    reward = builder.AddSystem(RewardSystem())
    builder.Connect(plant.get_state_output_port(), reward.get_input_port(0))
    builder.ExportOutput(reward.get_output_port(), "reward")

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
            self.ee_body = self.plant.GetBodyByName("ee_link") # adapt to my ee link
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
    # Episode end conditions:
    def monitor(context, state_view=StateView):
        plant_context = plant.GetMyContextFromRoot(context)
        state = plant.GetOutputPort("continuous_state").Eval(plant_context)
        s = state_view(state)

        ''' Truncation: the episode duration reaches the time limit. 
            Need this for finite horizon and stable learning '''
        if context.get_time() > time_limit: # in RL, usually this is measured in steps right?
            if debug:
                print("Episode reached time limit.")
            return EventStatus.ReachedTermination(
                diagram,
                "time limit")

        ''' Define termination by goal or safety constraints violation '''
        # for franka-reaching, defined distance-to-goal
        plant_compute.SetPositions(plant_compute_context,
                                   s.panda_joint_q)
        # TODO: change to "dummy1" later after welding finger to ee_link                    
        ee_frame = plant_compute.GetFrameByName("panda_link8")
        base_frame = plant_compute.GetFrameByName("panda_link0")
        X_base_ee = plant_compute.CalcPose(plant_compute_context, base_frame, ee_frame)
        p_ee_base = X_base_ee.translation()
        rot_ee_base = X_base_ee.rotation().ToQuaternion()

        dist_to_goal = np.linalg.norm(p_ee_base - np.array(p_ee_goal_base))
        if dist_to_goal < 0.05:  # within 5 cm of the goal
            if debug:
                print("Reached goal!")
            return EventStatus.ReachedTermination(
                   diagram,
                   "Reached goal")
        
        # quaternion distance
        def quat_error(q_d, q_c):
            """
            q = [w, x, y, z] convention
            Returns 3D orientation error vector.
            """
            # Ensure shortest path (handle double cover)
            if np.dot(q_d, q_c) < 0:
                q_c = -q_c
            
            # Error quaternion: q_e = q_d ⊗ q_c*
            w_d, x_d, y_d, z_d = q_d
            w_c, x_c, y_c, z_c = q_c
            
            # q_c conjugate (inverse for unit quaternion)
            # q_e = q_d * conj(q_c)
            w_e = w_d*w_c + x_d*x_c + y_d*y_c + z_d*z_c
            x_e = -w_d*x_c + x_d*w_c - y_d*z_c + z_d*y_c
            y_e = -w_d*y_c + x_d*z_c + y_d*w_c - z_d*x_c
            z_e = -w_d*z_c - x_d*y_c + y_d*x_c + z_d*w_c
            
            # For small errors: error ≈ 2 * [x_e, y_e, z_e]
            return 2.0 * np.array([x_e, y_e, z_e])

        ori_error = quat_error(np.array(rot_ee_goal_base),
                               rot_ee_base.wxyz())
        angle_error = np.linalg.norm(ori_error)
        if angle_error < 0.2:  # within 0.2 rads in axis-angle angle error
            if debug:
                print("Reached goal orientation!")
            return EventStatus.ReachedTermination(
                   diagram,
                   "Reached goal orientation")

        # Joint limits for Franka Panda (from URDF, in radians)
        q_max = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]  # positive limits
        q_min = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        v_max = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.61, 2.61, 2.61]) # panda hardware limits
        # u_up = np.array([87, 87, 87, 87, 12, 12, 12])
        
        # Termination: Joint position limits exceeded.
        if abs(s.panda_joint1_q) > q_max[0]:
            if debug:
                print("Joint position 1 exceeded limits.")
            return EventStatus.ReachedTermination(
                   diagram,
                   "Joint position 1 exceeded limits")
        
        if abs(s.panda_joint2_q) > q_max[1]:
            if debug:
                print("Joint position 2 exceeded limits.")
            return EventStatus.ReachedTermination(
                   diagram,
                   "Joint position 2 exceeded limits")

        if abs(s.panda_joint3_q) > q_max[2]:
            if debug:
                print("Joint position 3 exceeded limits.")
            return EventStatus.ReachedTermination(
                   diagram,
                   "Joint position 3 exceeded limits")

        if abs(s.panda_joint5_q) > q_max[4]:
            if debug:
                print("Joint position 5 exceeded limits.")
            return EventStatus.ReachedTermination(
                   diagram,
                   "Joint position 5 exceeded limits")

        if abs(s.panda_joint7_q) > q_max[6]:
            if debug:
                print("Joint position 7 exceeded limits.")
            return EventStatus.ReachedTermination(
                   diagram,
                   "Joint position 7 exceeded limits")
        
        if s.panda_joint4_q < q_min[3] or s.panda_joint4_q > q_max[3]:
            if debug:
                print("Joint position 4 exceeded limits.")
            return EventStatus.ReachedTermination(
                   diagram,
                   "Joint position 4 exceeded limits")

        if s.panda_joint6_q < q_min[5] or s.panda_joint6_q > q_max[5]:
            if debug:
                print("Joint position 6 exceeded limits.")
            return EventStatus.ReachedTermination(
                   diagram,
                   "Joint position 6 exceeded limits")
        
        # TODO: Termination: Joint velocity limits exceeded.
        # if abs(s.panda_joint1_w) > v_max[0]:
        #     if debug:
        #         print("Joint velocity 1 exceeded limits.")
        #     return EventStatus.ReachedTermination(
        #            diagram,
        #            "Joint velocity 1 exceeded limits")

        return EventStatus.Succeeded()

    simulator.set_monitor(monitor)


def set_home(simulator, diagram_context, seed):
    ''' An interface for domain randomization '''

    # Joint limits for Franka Panda (from URDF, in radians)
    q_max = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]  # positive limits
    q_min = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
    v_max = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.61, 2.61, 2.61]) # panda hardware limits
    # u_up = np.array([87, 87, 87, 87, 12, 12, 12])

    # randomize the q_init centred around some nominal home_position
    q_home = [0.9207,  0.2574, -0.9527, -2.0683,  0.2799,  2.1147, 2.]
    offset = 0.5 # rad 
    home_positions = [
        ('panda_joint1', np.random.uniform(low=q_home[0]-offset, high=q_home[0]+offset)),
        ('panda_joint2', np.random.uniform(low=q_home[1]-offset, high=q_home[1]+offset)),
        ('panda_joint3', np.random.uniform(low=q_home[2]-offset, high=q_home[2]+offset)),
        ('panda_joint4', np.random.uniform(low=q_home[3]-offset, high=q_home[3]+offset)),
        ('panda_joint5', np.random.uniform(low=q_home[4]-offset, high=q_home[4]+offset)),
        ('panda_joint6', np.random.uniform(low=q_home[5]-offset, high=q_home[5]+offset)),
        ('panda_joint7', np.random.uniform(low=q_home[6]-offset, high=q_home[6]+offset)),
    ]

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
    home_velocities = [
        ('panda_joint1', np.random.uniform(low=-0.1*v_max[0], high=0.1*v_max[0])),
        ('panda_joint2', np.random.uniform(low=-0.1*v_max[1], high=0.1*v_max[1])),
        ('panda_joint3', np.random.uniform(low=-0.1*v_max[2], high=0.1*v_max[2])),
        ('panda_joint4', np.random.uniform(low=-0.1*v_max[3], high=0.1*v_max[3])),
        ('panda_joint5', np.random.uniform(low=-0.1*v_max[4], high=0.1*v_max[4])),
        ('panda_joint6', np.random.uniform(low=-0.1*v_max[5], high=0.1*v_max[5])),
        ('panda_joint7', np.random.uniform(low=-0.1*v_max[6], high=0.1*v_max[6])),
    ]

    diagram = simulator.get_system()
    plant = diagram.GetSubsystemByName("plant_sim")
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

    # Clip the positions (I may not need this but for keeping generality)
    for pair in home_positions:
        joint = plant.GetJointByName(pair[0])
        joint.set_angle(plant_context,
                        np.clip(pair[1],
                                joint.position_lower_limit(),
                                joint.position_upper_limit()))
    
    # Randomize other params:
    # e.g. friction, mass, link dimensions, gravity, etc.
    # randomize mass 
    objects_mass_offset = [
        ("soup_base_link", np.random.uniform(low=-0.2, high=0.2))]
    for pair in objects_mass_offset:
        body = plant.GetBodyByName(pair[0])
        # why creating a new default context here? so mass default is always the same?
        mass = body.get_mass(plant.CreateDefaultContext()) 
        body.SetMass(plant_context, mass+pair[1])

def PandaReachEnv():
    pass