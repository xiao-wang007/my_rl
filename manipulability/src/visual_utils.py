from pydrake.all import (
    AddMultibodyPlantSceneGraph, DiagramBuilder, DiagramBuilder_,  SceneGraph, SceneGraph_,
    FindResourceOrThrow, InverseDynamicsController, 
    MultibodyPlant, MultibodyPlant_, Parser, Simulator, InverseKinematics)

from pydrake.autodiffutils import ExtractGradient, InitializeAutoDiff
import pydrake.autodiffutils
from pydrake.symbolic import MakeVectorContinuousVariable

import pydrake
from pydrake import geometry
from pydrake.math import RigidTransform, RigidTransform_, RollPitchYaw, RollPitchYaw_, RotationMatrix, RotationMatrix_ 
from pydrake.math import inv
from pydrake.solvers import MathematicalProgram, Solve, IpoptSolver, SnoptSolver
from functools import partial
from pydrake.all import (
    JointIndex, PiecewisePolynomial, JacobianWrtVariable,
    eq, ge, le,  AutoDiffXd, SnoptSolver, IpoptSolver,  
    AddUnitQuaternionConstraintOnPlant, PositionConstraint, OrientationConstraint
    )
from pydrake.geometry import (MeshcatVisualizer, StartMeshcat, FrameId)
from pydrake.visualization import (VisualizationConfig, 
                                   ApplyVisualizationConfig, 
                                   AddDefaultVisualization, 
                                   AddFrameTriadIllustration,)
from pydrake.multibody.tree import PrismaticJoint, RevoluteJoint, BallRpyJoint

import numpy as np
from functools import partial
from collections import OrderedDict, namedtuple
from pydrake.multibody.tree import FixedOffsetFrame, SpatialInertia, UnitInertia
from pathlib import Path
from src.workspace_split import split_workspace

import sys

from src.workspace_split import *

def visualize_workspace(meshcat, z_obj=0.03):
    centre = ISO2mine_converter()
    centre_copy = centre.copy()
    centre[-1] = z_obj
    coords = split_workspace(centre=centre, z=z_obj)
    corners = np.asarray(coords["corners"], dtype=float)

    edge_x = np.linalg.norm(corners[0] - corners[1])
    edge_y = np.linalg.norm(corners[1] - corners[2])
    plane_center = centre_copy # original WS spec 
    plane_thickness = 0.4
    meshcat.SetObject(
        "/workspace/plane",
        geometry.Box(edge_x, edge_y, plane_thickness),
        geometry.Rgba(0.05, 0.55, 0.9, 0.25),
    )
    meshcat.SetTransform(
        "/workspace/plane",
        RigidTransform(RotationMatrix.Identity(), plane_center),
    )

    block_centres = coords["block_centres"]
    print("block centres: \n")
    for centre in block_centres:
        print(centre)

    centre_radius = 0.005
    for i, centre in enumerate(block_centres, start=1):
        centre_path = f"/workspace/centres/centre_{i}"
        meshcat.SetObject(
            centre_path,
            geometry.Sphere(centre_radius),
            geometry.Rgba(1.0, 0.25, 0.1, 0.85),
        )
        meshcat.SetTransform(centre_path, RigidTransform(p=centre))
    
    l1_end_pts, l2_end_pts, l3_end_pts, l4_end_pts = coords["grid_vertices"]
    edge1, edge2, edge3, edge4 = coords["edges"]

    line_starts = [l1_end_pts[0], l2_end_pts[0], l3_end_pts[0], l4_end_pts[0], 
                   edge1[0], edge2[0], edge3[0], edge4[0]]
    line_ends = [l1_end_pts[1], l2_end_pts[1], l3_end_pts[1], l4_end_pts[1], 
                 edge1[1], edge2[1], edge3[1], edge4[1]]

    meshcat.SetLineSegments(
        "/workspace/blocks/grid",
        np.asfortranarray(np.array(line_starts, dtype=float).T),
        np.asfortranarray(np.array(line_ends, dtype=float).T),
        line_width=2.0,
        rgba=geometry.Rgba(0.0, 0.0, 0.7, 0.6),
    )


def scene_visualizer(arm_pose, meshcat, h=0.001, visualize_ws=True, z_obj=0.03):
    """
    target_pose is of RigidTransform()
    """
    
    if sys.platform == "darwin":
        pandanogripperfile = "drake_models/franka_description/urdf/panda_arm.urdf"
        # tomatocanFile = "drake_models/ycb/sdf/005_tomato_soup_can.sdf"
        # canVisual    = "drake_models/ycb/sdf/005_tomato_soup_can_visual.sdf"
        # canVisual = "drake_models/objects/can_visual.sdf"
        tablefile =  "drake_models/objects/table_top.sdf"
        dummy1File = "drake_models/objects/dummy1.sdf"

    elif sys.platform == "linux":
        pandanogripperfile = "drake_models/franka_description/urdf/panda_arm.urdf"
        # tomatocanFile = "drake_models/ycb/sdf/005_tomato_soup_can.sdf"
        # canVisual    = "drake_models/ycb/sdf/005_tomato_soup_can_visual.sdf"
        tablefile =  "drake_models/objects/table_top.sdf"
        dummy1File = "drake_models/objects/dummy1.sdf"
    else:
        pandanogripperfile = "drake_models/franka_description/urdf/panda_arm.urdf"
        tablefile =  "drake_models/objects/table_top.sdf"
        dummy1File = "drake_models/objects/dummy1.sdf"

    model_root = Path(__file__).resolve().parent.parent
    pandanogripperfile = str(model_root / pandanogripperfile)
    tablefile = str(model_root / tablefile)
    dummy1File = str(model_root / dummy1File)

    X_W_base = RigidTransform(
        RollPitchYaw(np.array([0, 0, -90])*np.pi/180),
        [0, -0.2, 0.])

    X_link8_dummy1 = RigidTransform(R=RotationMatrix.Identity(), p=[0., 0., 0.15])

    X_W_table = RigidTransform(
        RotationMatrix.Identity(), 
        [0., -0.4, 0.])

    ##########################################################################################
    builder_double = DiagramBuilder()
    plant_double, sceneGraph_double = AddMultibodyPlantSceneGraph(builder_double, time_step=h)
    parser_double = Parser(plant_double)

    parser_double.AddModels(tablefile)
    panda_model_instance_double = parser_double.AddModels(pandanogripperfile)[0]
    # visual_can_model_instance_double = parser_double.AddModels(canVisual)[0]
    dummy1_model_instance_double = parser_double.AddModels(dummy1File)[0] 
    # can_model_instance_double = parser_double.AddModels(tomatocanFile)[0]

    # _ = AddPlanarPusher(plant_double, parser_double, fingerFile, "finger")

    table_top_frame_double = plant_double.GetFrameByName("table_top_center")
    # tomatocan_frame_double = plant_double.GetFrameByName("base_link_soup")
    # visual_can_frame_double = plant_double.GetFrameByName("base_link_soup_visual")
    base_frame_double = plant_double.GetFrameByName("panda_link0")
    dummy1_frame_double = plant_double.GetFrameByName("dummy1")
    link8_frame_double = plant_double.GetFrameByName("panda_link8")

    # weld static entities
    plant_double.WeldFrames(plant_double.world_frame(), table_top_frame_double, X_W_table)
    plant_double.WeldFrames(plant_double.world_frame(), base_frame_double, X_W_base) 
    plant_double.WeldFrames(link8_frame_double, dummy1_frame_double, X_link8_dummy1) # weld the contact frame in hand
    # plant_double.WeldFrames(plant_double.world_frame(), visual_can_frame_double, target_pose_SE3)
    plant_double.Finalize()
    plant_AD = plant_double.ToScalarType[AutoDiffXd]() # conv_armert to AD

    # using meshcat for vis (optional in headless/debug environments)
    if meshcat is not None:
        visualizer = MeshcatVisualizer.AddToBuilder(builder_double, sceneGraph_double, meshcat)
        visualization_config = VisualizationConfig()
        visualization_config.publish_contacts = True
        visualization_config.publish_proximity = True
        visualization_config.publish_illustration = True
        ApplyVisualizationConfig(visualization_config, builder_double, meshcat=meshcat)

    diagram_double = builder_double.Build()

    # visualizing the frames with drawn frames
    visualize_frames = True
    triad_radius = 0.05
    triad_length = 0.2
    triad_opacity = 0.1
    radius_vis = 0.003

    name_list = ["table_top_link", "dummy1", "panda_link8"] 
    # contact_name_list = ["dummy6"] 
    if visualize_frames:
        for name in name_list:
            AddFrameTriadIllustration(
                scene_graph=sceneGraph_double,
                body=plant_double.GetBodyByName(name),
                plant=plant_double,
                length=triad_length,
                radius=radius_vis,
                opacity=triad_opacity,)       

    root_context_double = diagram_double.CreateDefaultContext()
    double_context = plant_double.GetMyContextFromRoot(root_context_double)

    plant_double.SetPositions(double_context, panda_model_instance_double, arm_pose)

    # visualizing the time instance using the double type diagram
    plant_double.get_actuation_input_port().FixValue(double_context, np.zeros(7))
    diagram_double.ForcedPublish(root_context_double) # publish the corresponding diagram

    #* visualize workspace
    if visualize_ws:
        visualize_workspace(meshcat, z_obj=z_obj)

    return plant_double, double_context, sceneGraph_double, diagram_double
