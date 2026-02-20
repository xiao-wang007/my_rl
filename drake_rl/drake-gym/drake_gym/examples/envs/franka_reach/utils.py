from pydrake.all import (Parser, RigidTransform, RollPitchYaw,
                           RotationMatrix, AddMultibodyPlantSceneGraph)
import numpy as np                    
from scipy.spatial.transform import Rotation

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


#######################################################################
def r1r2_to_quaternion(r1r2):
    """
    Convert 6D rotation (r1, r2) to quaternion [w, x, y, z].
    r1r2: array of shape (6,) containing [r1_x, r1_y, r1_z, r2_x, r2_y, r2_z]
    """
    r1 = r1r2[:3]  # x-axis of rotation matrix
    r2 = r1r2[3:6]  # y-axis of rotation matrix
    
    # Ensure orthonormal (in case of numerical errors)
    r1 = r1 / np.linalg.norm(r1)
    r2 = r2 - np.dot(r2, r1) * r1  # Gram-Schmidt
    r2 = r2 / np.linalg.norm(r2)
    
    # Compute r3 = r1 × r2 (z-axis)
    r3 = np.cross(r1, r2)
    
    # Build rotation matrix (columns are r1, r2, r3)
    R = np.column_stack([r1, r2, r3])
    
    # Convert rotation matrix to quaternion [w, x, y, z]
    # Using Shepperd's method
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    quat = np.array([w, x, y, z])
    return quat / np.linalg.norm(quat)  # Normalize

#######################################################################
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

#######################################################################
def r1r2_to_quat(r1r2):
    """ r1r2 is 6d vector """
    r1 = r1r2[:3]
    r2 = r1r2[3:6]

    # Ensure orthonormal (in case of numerical errors)
    r1 = r1 / np.linalg.norm(r1)
    r2 = r2 - np.dot(r2, r1) * r1  # Gram-Schmidt
    r2 = r2 / np.linalg.norm(r2)
    r3 = np.cross(r1, r2)

    # Build rotation matrix (columns are r1, r2, r3)
    R = np.column_stack([r1, r2, r3])

    # scipy uses [x, y, z, w] order, convert to [w, x, y, z]
    quat_xyzw = Rotation.from_matrix(R).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])