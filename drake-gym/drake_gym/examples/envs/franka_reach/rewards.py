"""
Docstring for drake-gym.drake_gym.examples.envs.franka_reach.rewards

This CompositeReward class serves as an interface to combine multiple reward, which 
takes in custom reward functions for modularity and flexibility.
"""
import numpy as np

class CompositeReward():
    def __init__(self):
        self.components = []
        self.target_pose = None
    
    def add_reward(self, name: str, fn: callable, weight: float = 1.0):
        self.components.append({'name': name, 'fn': fn, 'weight': weight})
        return self # for chaining
    
    def set_target(self, target_pose):
        self.target_pose = target_pose
    
    def __call__(self, state, plant, plant_context, target_pos=None, target_r1r2=None, action=None):
        total = 0.0
        breakdown = {}

        # build kwargs with all available info
        kwargs = {
            'state': state,
            'target_pos': target_pos,
            'target_r1r2': target_r1r2,
            'plant': plant,
            'plant_context': plant_context,
            'action': action
        }

        for compo in self.components:
            # to be more general
            r = compo['fn'](**kwargs)  # each fn takes only what needed as inputs
            breakdown[compo['name']] = r 
            total += compo['weight'] * r
    
        return total, breakdown

##############################################################################
################### define the individual reward functions ###################
##############################################################################

# some rough examples:
'''
def reaching_reward(state, target, plant, plant_context, **kwargs):
    # Uses target
    ee_pos = get_ee_position(state, plant, plant_context)
    return -np.linalg.norm(ee_pos - target)

def smoothness_reward(action, **kwargs):
    # Ignores target, state, plant — only needs action
    return -np.linalg.norm(action)

def safety_reward(state, **kwargs):
    # Only needs state
    return -compute_constraint_violation(state)

'''

def reaching_reward(state, plant, plant_context, target_pos, target_r1r2, **kwargs):
    """
    Reaching reward for end-effector pose matching.
    
    Args:
        state: robot state [q(7), v(7)]
        plant: MultibodyPlant for FK computation
        plant_context: context for the plant
        target_pos: target position [x, y, z]
        target_r1r2: target orientation as [rx(3), ry(3)] where rx=x-axis, ry=y-axis
    """
    q, _ = state[:7], state[7:14]
    plant.SetPositions(plant_context, q)
    ee_frame = plant.GetFrameByName("panda_link8")
    ee_pose = ee_frame.CalcPoseInWorld(plant_context)
    p_ee_w = ee_pose.translation()
    
    r_ee_w = ee_pose.rotation().matrix()
    ee_xhat_w = r_ee_w[:, 0]
    ee_zhat_w = r_ee_w[:, 2]

    # extract the target (compute z-axis from x cross y)
    p_target_w = target_pos
    target_xhat_w = target_r1r2[:3]
    target_yhat_w = target_r1r2[3:6]
    target_zhat_w = np.cross(target_xhat_w, target_yhat_w)

    # position reward (0, 1]
    kp = 10.0
    ep = p_ee_w - p_target_w
    r_p = np.exp(-kp * np.dot(ep, ep))
    
    # oreintation error [-1, 1]
    ex = np.dot(ee_xhat_w, target_xhat_w)
    ez = np.dot(ee_zhat_w, target_zhat_w)
    r_o = 0.5 * (ex + ez)

    # terminal rewards
    epsilon_pos = 0.02  # 2 cm
    epsilon_ori = np.deg2rad(10) # 10 degrees 

    # clip for numerical stability
    e_ang_x = np.arccos(np.clip(ex, -1.0, 1.0))
    e_ang_z = np.arccos(np.clip(ez, -1.0, 1.0))

    r_terminal = 0.
    if np.linalg.norm(ep) < epsilon_pos and e_ang_x < epsilon_ori and e_ang_z < epsilon_ori:
        r_terminal = 10.0
    
    return r_p + r_o + r_terminal




