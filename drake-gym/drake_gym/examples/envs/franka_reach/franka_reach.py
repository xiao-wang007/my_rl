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

    # Actions are positions sent to plant.
    actuation = builder.AddSystem(Multiplexer([1]*7)) # 7 input ports, 1 for each joint

    ''' PassThrough() for naming (e.g. multiple .sdf in the same plant) and 
        exporting the input port for RL agent.
    '''
    jvel_actuation_ = builder.AddSystem(PassThrough(7))

    ''' if there is under-actuation '''
    # # Zero torque to the revolute joint --it is underactuated.
    # revolute_actuation_torque = builder.AddSystem(ConstantVectorSource([0]))
    # builder.Connect(revolute_actuation_torque.get_output_port(),
    #                 actuation.get_input_port(1))

    builder.Connect(jvel_actuation_.get_output_port(),
                    actuation.get_input_port(0))
    builder.Connect(actuation.get_output_port(),
                    plant.get_actuation_input_port(agent)) # agent is a model instance
    
    # Export for RL agent
    builder.ExportInput(jvel_actuation_.get_input_port(), "actions")