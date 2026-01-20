# terminations.py
import numpy as np
from pydrake.systems.framework import EventStatus


class TerminationResult:
    """Result of a termination check."""
    def __init__(self, triggered: bool, reason: str = ""):
        self.triggered = triggered
        self.reason = reason


# Individual termination conditions (return TerminationResult)
def time_limit_termination(t, time_limit, **kwargs):
    """Truncation: episode exceeded time limit."""
    if t > time_limit:
        return TerminationResult(True, "time limit")
    return TerminationResult(False)


def ee_position_goal_reached_termination(ee_pos, target_pos, threshold=0.05, **kwargs):
    """Success: end-effector reached goal position."""
    dist = np.linalg.norm(ee_pos - target_pos)
    if dist < threshold:
        return TerminationResult(True, "position reached goal")
    return TerminationResult(False)

def ee_pose_goal_reached_termination(ee_pose, target_pose, ep_threshold=0.1, 
                                     eq_threshold=0.35, **kwargs):
    '''
    Docstring for ee_pose_goal_reached_termination
    
    :param ee_pose: 7D pose of the end-effector [x, y, z, qw, qx, qy, qz]
    :param target_pose: 7D target pose [x, y, z, qw, qx, qy, qz]
    :param ep_threshold: position error threshold
    :param eq_threshold: orientation error threshold, 0.35 ~ 20 degrees

    norm(eq) = 2 * sin(theta/2) where theta is the angle between two quaternions
    '''
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

    ep = np.linalg.norm(ee_pose[:3] - target_pose[:3])
    eq = quat_error(target_pose[3:], ee_pose[3:])

    if ep < ep_threshold and np.linalg.norm(eq) < eq_threshold:
        return TerminationResult(True, "end-effector pose reached goal")
    return TerminationResult(False)

def joint_limit_termination(qs, q_min, q_max, margin=0.01, **kwargs):
    """Safety: joint limits violated."""
    if np.any(qs < q_min + margin) or np.any(qs > q_max - margin):
        return TerminationResult(True, "joint limit violated")
    return TerminationResult(False)


def velocity_limit_termination(vs, v_max, **kwargs):
    """Safety: velocity limits exceeded."""
    if np.any(np.abs(vs) > v_max):
        return TerminationResult(True, "velocity limit exceeded")
    return TerminationResult(False)


def collision_termination(plant, plant_context, **kwargs):
    """Safety: collision detected."""
    # Your collision checking logic
    pass


# Composite termination checker
class CompositeTermination:
    """Modular termination that combines multiple conditions."""
    
    def __init__(self):
        self.conditions = []
        self.target_pos = None
    
    def add_termination(self, name: str, fn: callable, is_success: bool = False):
        """
        Add a termination condition.
        is_success: True if this is a success condition (e.g., goal reached)
                   False if this is a failure/safety condition
        """
        self.conditions.append({
            'name': name, 
            'fn': fn, 
            'is_success': is_success
        })
        return self
    
    def set_target(self, target_pos):
        self.target_pos = target_pos
    
    def __call__(self, **kwargs):
        """
        Check all termination conditions.
        Returns: (triggered: bool, reason: str, is_success: bool)
        """
        kwargs['target_pos'] = self.target_pos
        
        for cond in self.conditions:
            result = cond['fn'](**kwargs)
            if result.triggered:
                return True, result.reason, cond['is_success']
        
        return False, "", False