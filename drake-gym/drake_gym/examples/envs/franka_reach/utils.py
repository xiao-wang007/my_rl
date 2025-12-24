from pydrake.all import (Parser, RigidTransform, RollPitchYaw,
                           RotationMatrix, AddMultibodyPlantSceneGraph)
import numpy as np                    

def AddArm(plant, files=[]):
    arm_file, finger_file = files
    parser = Parser(plant)
    arm_model_instance = parser.AddModels(arm_file)[0]
    parser.AddModels(finger_file)[0]

    X_W_base = RigidTransform(
        RollPitchYaw(np.array([0., 0., 0.])*np.pi/180), 
        [0., 0., 0.])
    arm_base_frame = plant.GetFrameByName("panda_link0")
    plant.WeldFrames(plant.world_frame(), arm_base_frame, X_W_base) 

    X_link8_dummy1 = RigidTransform(R=RotationMatrix.Identity(), p=[0., 0., 0.15])
    link8_frame = plant.GetFrameByName("panda_link8")
    finger_frame = plant.GetFrameByName("dummy1")
    plant.WeldFrames(link8_frame, finger_frame, X_link8_dummy1)

    return arm_model_instance


def MakeArmOnlyPlant(builder, h):
    """
    I should probably write a plant factory. This is currently hard-coded in AddArm()
    that the arm in sim plant and arm plant are consistent.  
    """
    arm_file    = "/Users/xiao/0_codes/ICBM_drake/drake_models/franka_description/urdf/panda_arm.urdf"
    finger_file = "/Users/xiao/0_codes/ICBM_drake/drake_models/objects/dummy1.sdf"

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=h)
    arm = AddArm(plant, [arm_file, finger_file])
    plant.Finalize()
    return plant, scene_graph, arm