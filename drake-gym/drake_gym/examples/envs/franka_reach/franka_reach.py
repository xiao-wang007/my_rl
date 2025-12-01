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

    builder = DiagramBuilder()

    multibody_plant_config = MultibodyPlantConfig(
        time_step=sim_time_step,
        contact_model=contact_model,
        discrete_contact_solver=contact_solver,
        )

    plant, scene_graph = AddMultibodyPlant(multibody_plant_config, builder)

    # Add assets to the plant.
    agent = AddAgent(plant)
    plant.Finalize()
    plant.set_name("plant")

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
            
    ''' next up: # Episode end conditions, line 254 '''