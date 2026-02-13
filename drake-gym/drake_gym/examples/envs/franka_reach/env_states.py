"""
The EnvStates() class contains shared mutable states of the environment, such as 
randomized goal, termination conditions that are not 
"""

import numpy as np
from terminations import TerminationResult

########################################## Goal state
class GoalState():
    """ Shared mutable goal state gets updated on reset (in set_home()) 
        and read by RewardSystem() during sim steps.  """
    def __init__(self):
        self.goal_pos = np.array([0.5, 0.5, 0.3])
        # self.goal_quat = np.array([1., 0., 0., 0.])  # w, x, y, z
        self.goal_r1r2 = np.zeros(6)


''' 这个是邪修 '''
# def consecutive_hold_at_target_terminaton(threshold=0.05, k_steps=30, **kwargs):
#     # Skip check if goal not set yet (happens during simulator.Initialize() before set_home())
#     count = [0] 

#     def check(ee_pos, target_pos, **kwargs):
#         if target_pos is None:
#             return TerminationResult(False)
        
#         ep = np.linalg.norm(ee_pos - target_pos)
#         if ep < threshold:
#             count[0] += 1
#             if count[0] >= k_steps:
#                 return TerminationResult(True, f"held for {k_steps} steps")
#         else:
#             count[0] = 0
#         return TerminationResult(False)
    
#     def reset():
#         count[0] = 0
    
#     check.reset = reset # I don't like this hacky style. TODO: be traditional!
#     return check

########################################## ee stabilization at goal termination
class ConsecutiveHoldAtTargetTermination():
    def __init__(self, threshold=0.05, k_steps=30):
        self.threshold = threshold
        self.k_steps = k_steps
        self.count = 0
    
    def __call__(self, ee_pos, target_pos, **kwargs):
        if target_pos is None:
            return TerminationResult(False)
        
        if np.linalg.norm(ee_pos - target_pos) < self.threshold:
            self.count += 1
            if self.count >= self.k_steps:
                return TerminationResult(True, f"held for {self.k_steps} steps")
        else:
            self.count = 0
        return TerminationResult(False)
    
    def reset(self):
        self.count = 0
