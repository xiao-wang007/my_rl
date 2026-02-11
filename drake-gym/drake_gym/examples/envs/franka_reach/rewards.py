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
    
    def __call__(self, state, plant, plant_context, target_pos=None, 
                 target_r1r2=None, action=None, v_prev=None):
        total = 0.0
        breakdown = {}

        # build kwargs with all available info
        kwargs = {
            'state': state,
            'target_pos': target_pos,
            'target_r1r2': target_r1r2,
            'plant': plant,
            'plant_context': plant_context,
            'action': action,
            'v_prev': v_prev,
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

##################### Position reaching reward
def reaching_position(state, plant, plant_context, target_pos, coeff=10.0, **kwargs):
    """
    Reaching reward for end-effector pose matching.
    
    Args:
        state: robot state [q(7), v(7)]
        plant: MultibodyPlant for FK computation
        plant_context: context for the plant
        target_pos: target position [x, y, z]
    """
    q, _ = state[:7], state[7:14]
    plant.SetPositions(plant_context, q)
    ee_frame = plant.GetFrameByName("panda_link8")
    ee_pose = ee_frame.CalcPoseInWorld(plant_context)
    p_ee_w = ee_pose.translation()


    # position reward (0, 1]
    ep = p_ee_w - target_pos
    r_p = np.exp(-coeff * np.dot(ep, ep))
    
    return r_p


##################### Orientation reaching reward
def reaching_orientation(state, plant, plant_context, target_r1r2, coeff=10., **kwargs):
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
    
    r_ee_w = ee_pose.rotation().matrix()
    ee_xhat_w = r_ee_w[:, 0]
    ee_zhat_w = r_ee_w[:, 2]

    # extract the target (compute z-axis from x cross y)
    target_xhat_w = target_r1r2[:3]
    target_yhat_w = target_r1r2[3:6]
    target_zhat_w = np.cross(target_xhat_w, target_yhat_w)
    
    # oreintation error [-1, 1]
    ex = np.dot(ee_xhat_w, target_xhat_w)
    ez = np.dot(ee_zhat_w, target_zhat_w)
    r_o = coeff * (ex + ez)
    
    return r_o 

##################### Orientation reaching reward
def reaching_terminal(state, plant, plant_context, target_pos, target_r1r2, 
                      epsilon_pos, epsilon_ori, **kwargs):
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

    ep = p_ee_w - p_target_w
    
    # oreintation error [-1, 1]
    ex = np.dot(ee_xhat_w, target_xhat_w)
    ez = np.dot(ee_zhat_w, target_zhat_w)

    # terminal rewards
    epsilon_ori_rad = np.deg2rad(epsilon_ori) 

    # clip for numerical stability
    e_ang_x = np.arccos(np.clip(ex, -1.0, 1.0))
    e_ang_z = np.arccos(np.clip(ez, -1.0, 1.0))

    r_terminal = 0.
    if np.linalg.norm(ep) < epsilon_pos and e_ang_x < epsilon_ori_rad and e_ang_z < epsilon_ori_rad:
        r_terminal = 10.0
    
    return r_terminal

##################### Penalty for acceleration (smoothness)
def acceleration_smoothness(state, v_prev, dt, coeff=5e-3, **kwargs):
    """
    Smoothness reward based on acceleration (change in velocity).
    
    Args:
        state: robot state [q(7), v(7)]
        v_prev: previous velocity (from last time step)
        dt: time step duration
    """
    v_now = state[7:14]
    a = (v_now - v_prev) / dt
    return -coeff * np.dot(a, a)
