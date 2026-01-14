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


def goal_reached_termination(ee_pos, target_pos, threshold=0.05, **kwargs):
    """Success: end-effector reached goal position."""
    dist = np.linalg.norm(ee_pos - target_pos)
    if dist < threshold:
        return TerminationResult(True, "reached goal")
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