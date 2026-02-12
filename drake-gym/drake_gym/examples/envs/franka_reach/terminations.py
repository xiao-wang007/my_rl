# terminations.py
import numpy as np
from pydrake.systems.framework import EventStatus
from drake_gym.examples.envs.franka_reach.utils import r1r2_to_quaternion, quat_error


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

def ee_position_reached_termination(ee_pos, target_pos, ep_threshold=0.05, **kwargs):
    '''
    Docstring for ee_pose_goal_reached_termination
    
    :param ee_pose: 7D pose of the end-effector [x, y, z, qw, qx, qy, qz]
    :param target_pose: 7D target pose [x, y, z, qw, qx, qy, qz]
    :param ep_threshold: position error threshold
    :param eq_threshold: orientation error threshold, 0.35 ~ 20 degrees

    norm(eq) = 2 * sin(theta/2) where theta is the angle between two quaternions
    '''
    # Skip check if goal not set yet (happens during simulator.Initialize() before set_home())
    if target_pos is None:
        return TerminationResult(False)
    
    ep = np.linalg.norm(ee_pos - target_pos)

    if ep < ep_threshold: 
        return TerminationResult(True, "end-effector position reached goal")
    return TerminationResult(False)

def ee_orientation_reached_termination(ee_quat, target_r1r2, eq_threshold=0.35, **kwargs):
    '''
    Docstring for ee_pose_goal_reached_termination
    
    :param ee_pose: 7D pose of the end-effector [x, y, z, qw, qx, qy, qz]
    :param target_pose: 7D target pose [x, y, z, qw, qx, qy, qz]
    :param ep_threshold: position error threshold
    :param eq_threshold: orientation error threshold, 0.35 ~ 20 degrees

    norm(eq) = 2 * sin(theta/2) where theta is the angle between two quaternions
    '''
    # Skip check if goal not set yet (happens during simulator.Initialize() before set_home())
    if target_r1r2 is None:
        return TerminationResult(False)
    
    # convert target_r1r2 to quaternion
    target_quat = r1r2_to_quaternion(target_r1r2)
    eq = quat_error(target_quat, ee_quat)

    if np.linalg.norm(eq) < eq_threshold:
        return TerminationResult(True, "end-effector pose reached goal")
    return TerminationResult(False)


def joint_limit_termination(qs, q_min, q_max, margin=0.01, **kwargs):
    """Safety: joint limits violated."""
    q_min = np.array(q_min)
    q_max = np.array(q_max)
    if np.any(qs < q_min + margin) or np.any(qs > q_max - margin):
        return TerminationResult(True, "joint limit violated")
    return TerminationResult(False)


def velocity_limit_termination(vs, v_max, **kwargs):
    """Safety: velocity limits exceeded."""
    v_max = np.array(v_max)
    if np.any(np.abs(vs) > v_max):
        return TerminationResult(True, "velocity limit exceeded")
    return TerminationResult(False)


def collision_termination(plant, plant_context, **kwargs):
    """Safety: collision detected."""
    # Your collision checking logic
    pass

def consecutive_hold_at_target_termination(ee_pos, target_pos, threshold=0.05, k_steps=30, **kwargs):
    # Skip check if goal not set yet (happens during simulator.Initialize() before set_home())
    count = [0] 

    def check(ee_pos, target_pos, **kwargs):
        if target_pos is None:
            return TerminationResult(False)
        
        ep = np.linalg.norm(ee_pos - target_pos)
        if ep < threshold:
            count[0] += 1
            if count[0] >= k_steps:
                return TerminationResult(True, f"held for {k_steps} steps")
        else:
            count[0] = 0
        return TerminationResult(False)
    
    def reset():
        count[0] = 0
    
    check.reset = reset # I don't like this hacky style. TODO: be traditional!
    return check



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