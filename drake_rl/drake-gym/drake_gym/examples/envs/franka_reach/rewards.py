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

        # Cache MJX translated terms once per step if any mjx_* component exists.
        if any(
            callable(c.get("fn")) and getattr(c["fn"], "__name__", "").startswith("mjx_")
            for c in self.components
        ):
            kwargs["_mjx_terms"] = _mjx_reward_terms(**kwargs)

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
def reaching_position(state, plant, plant_context, target_pos, coeff=1.0, **kwargs):
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
    r_p = np.exp(-coeff * np.dot(ep, ep)) # (0, 1]
    
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
    q = state[:7]
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
    q = state[:7]
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
        r_terminal = 1.0
    
    return r_terminal

##################### Reaching position reward
def reaching_position_terminal(state, plant, plant_context, target_pos, 
                      epsilon_pos, **kwargs):
    """
    Reaching reward for end-effector pose matching.
    
    Args:
        state: robot state [q(7), v(7)]
        plant: MultibodyPlant for FK computation
        plant_context: context for the plant
        target_pos: target position [x, y, z]
    """
    q = state[:7]
    plant.SetPositions(plant_context, q)
    ee_frame = plant.GetFrameByName("panda_link8")
    ee_pose = ee_frame.CalcPoseInWorld(plant_context)
    p_ee_w = ee_pose.translation()

    # extract the target (compute z-axis from x cross y)
    p_target_w = target_pos
    ep = p_ee_w - p_target_w
    
    if np.linalg.norm(ep) < epsilon_pos:
        return 1.0
    
    return 0.0

##################### Penalty for velocity (smoothness)
def velocity_smoothness(state, v_prev, **kwargs):
    """
    Smoothness reward based on velocity change.
    
    Args:
        state: robot state [q(7), v(7)]
        v_prev: previous velocity (from last time step)
    """
    v_max = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
    max_dv_sq = np.dot(v_max, v_max)
    v_now = state[7:14]
    dv = (v_now - v_prev) 
    return -np.dot(dv, dv) / max_dv_sq # [-1, 0]

##################### Penalty for velocity (smoothness)
def hold_at_target(state, plant, plant_context, target_pos, radius, bonus=0.2, **kwargs):
    """
    Reaching reward for end-effector pose matching.
    
    Args:
        state: robot state [q(7), v(7)]
        plant: MultibodyPlant for FK computation
        plant_context: context for the plant
        target_pos: target position [x, y, z]
    """
    q = state[:7]
    plant.SetPositions(plant_context, q)
    ee_frame = plant.GetFrameByName("panda_link8")
    ee_pose = ee_frame.CalcPoseInWorld(plant_context)
    p_ee_w = ee_pose.translation()

    # position reward (0, 1]
    ep = p_ee_w - target_pos
    
    if np.linalg.norm(ep) < radius:
        ''' with bonous = 0.2, sim dt = 0.01, reward for
            1 second is 20 which is comparable to the 
            reaching reward '''
        return bonus # 
    return 0.0

##################### Penalty for velocity (smoothness)
def velocity_damping_near_target(state, plant, plant_context, target_pos, 
                                 near_radius=0.1, coeff=0.05, **kwargs): 
    q, v = state[:7], state[7:14]
    plant.SetPositions(plant_context, q)
    ee_frame = plant.GetFrameByName("panda_link8")
    ee_pose = ee_frame.CalcPoseInWorld(plant_context)
    p_ee_w = ee_pose.translation()

    # position reward (0, 1]
    ep = p_ee_w - target_pos
    
    if np.linalg.norm(ep) < near_radius:
        return -coeff * np.dot(v, v)
    return 0.0
    

########## For striking task #################################

# MJX-translated reward defaults from mjx_rl/my_builds/reward_weights_and_configs.py
MJX_R_CONFIGS = {
    "alpha1": np.float32(1.0),
    "alpha3": np.float32(0.01),
    "alpha4": np.float32(0.01),
    "eps_pos": np.float32(0.05),
    "eps_vel": np.float32(0.1),
    "eps_tilt": np.float32(0.1),
    "alpha_tilt": np.float32(1.0),
    "beta_tilt": np.float32(2.0),
    "eps_vmag": np.float32(0.05),
}

MJX_R_WEIGHTS = {
    "w_pos_mid": np.float32(1.0),
    "w_vel_mid": np.float32(1.0),
    "w_pos_final": np.float32(1.0),
    "w_vel_progress": np.float32(0.05),
    "w_action_rate": np.float32(0.001),
    "w_tilt": np.float32(1.0),
}


def _ee_pose_velocity_drake(state, plant, plant_context, ee_frame_name="panda_link8"):
    q = state[:7]
    v = state[7:14]
    plant.SetPositions(plant_context, q)
    plant.SetVelocities(plant_context, v)
    ee_frame = plant.GetFrameByName(ee_frame_name)
    world_frame = plant.world_frame()
    ee_pose = ee_frame.CalcPoseInWorld(plant_context)
    p_ee_w = ee_pose.translation()
    r_ee_w = ee_pose.rotation().matrix()

    from pydrake.all import JacobianWrtVariable

    jacp = plant.CalcJacobianTranslationalVelocity(
        plant_context,
        JacobianWrtVariable.kV,
        ee_frame,
        np.zeros(3),
        world_frame,
        world_frame,
    )
    v_ee_w = jacp @ v
    return p_ee_w, r_ee_w, v_ee_w


def _mjx_reward_terms(
    state,
    plant,
    plant_context,
    target_pos=None,
    action=None,
    v_prev=None,
    x_mid=None,
    v_mid=None,
    x_final=None,
    r_configs=None,
    r_weights=None,
):
    cfg = MJX_R_CONFIGS if r_configs is None else r_configs
    w = MJX_R_WEIGHTS if r_weights is None else r_weights
    x_mid = np.array([0.4, 0.4, 0.05]) if x_mid is None else np.asarray(x_mid)
    v_mid = np.zeros(3) if v_mid is None else np.asarray(v_mid)
    if target_pos is not None:
        x_final = np.asarray(target_pos)
    else:
        x_final = np.array([0.4, 0.4, 0.25]) if x_final is None else np.asarray(x_final)

    x_next, r_ee_w, v_next = _ee_pose_velocity_drake(
        state, plant, plant_context, ee_frame_name="panda_link8"
    )
    zhat_w = np.array([0.0, 0.0, 1.0], dtype=float)
    zhat_w_opp = np.array([0.0, 0.0, -1.0], dtype=float)

    pos_err_mid = np.linalg.norm(x_next - x_mid)
    vel_err_mid = np.linalg.norm(v_next - v_mid)
    retract_err = np.linalg.norm(x_next - x_final)

    r_pos_mid = np.exp(-cfg["alpha1"] * pos_err_mid**2)
    r_vel_mid = np.exp(-cfg["alpha3"] * vel_err_mid**2)
    r_vel_gated = r_pos_mid * r_vel_mid

    zhat_ee_w = r_ee_w[:, 2]
    v_hori = v_next - np.dot(v_next, zhat_w) * zhat_w
    v_hori_hat = v_hori / (np.linalg.norm(v_hori) + cfg["eps_tilt"])
    z_tilted_raw = zhat_w_opp + cfg["beta_tilt"] * v_hori_hat
    z_tilted = z_tilted_raw / (np.linalg.norm(z_tilted_raw) + cfg["eps_tilt"])
    align = np.dot(zhat_ee_w, z_tilted)
    r_tilt = np.exp(-cfg["alpha_tilt"] * (1.0 - align))
    r_tilt_gated = r_pos_mid * r_tilt

    v_mid_hat = v_mid / (np.linalg.norm(v_mid) + cfg["eps_vel"])
    v_next_hat = v_next / (np.linalg.norm(v_next) + cfg["eps_vel"])
    dir_speed = np.dot(v_next_hat, v_mid_hat)
    speed_gate = 1.0 if np.linalg.norm(v_mid) > cfg["eps_vmag"] else 0.0
    r_dir = r_pos_mid * w["w_vel_progress"] * dir_speed * speed_gate

    v_cmd_now = np.asarray(action[:7], dtype=float) if action is not None else np.asarray(state[7:14], dtype=float)
    v_cmd_prev = np.asarray(v_prev, dtype=float) if v_prev is not None else np.zeros(7)
    r_action_rate = -w["w_action_rate"] * np.sum((v_cmd_now - v_cmd_prev) ** 2)

    mid_achieved_now = (pos_err_mid < cfg["eps_pos"]) and (vel_err_mid < cfg["eps_vel"])
    r_retract = w["w_pos_final"] * np.exp(-cfg["alpha4"] * retract_err**2)
    r_mid_total = (
        w["w_pos_mid"] * r_pos_mid
        + w["w_vel_mid"] * r_vel_gated
        + w["w_tilt"] * r_tilt_gated
        + r_dir
    )

    return {
        "r_pos_mid": float(r_pos_mid),
        "r_vel_gated": float(r_vel_gated),
        "r_tilt_gated": float(r_tilt_gated),
        "r_dir": float(r_dir),
        "r_retract": float(r_retract),
        "r_action_rate": float(r_action_rate),
        "r_mid_total": float(r_mid_total),
        "mid_achieved_now": float(mid_achieved_now),
    }


def mjx_mid_position_reward(state, plant, plant_context, **kwargs):
    terms = kwargs.get("_mjx_terms")
    if terms is None:
        terms = _mjx_reward_terms(state, plant, plant_context, **kwargs)
    w = kwargs.get("r_weights", MJX_R_WEIGHTS)
    return float(w["w_pos_mid"] * terms["r_pos_mid"])


def mjx_mid_velocity_gated_reward(state, plant, plant_context, **kwargs):
    terms = kwargs.get("_mjx_terms")
    if terms is None:
        terms = _mjx_reward_terms(state, plant, plant_context, **kwargs)
    w = kwargs.get("r_weights", MJX_R_WEIGHTS)
    return float(w["w_vel_mid"] * terms["r_vel_gated"])


def mjx_mid_tilt_gated_reward(state, plant, plant_context, **kwargs):
    terms = kwargs.get("_mjx_terms")
    if terms is None:
        terms = _mjx_reward_terms(state, plant, plant_context, **kwargs)
    w = kwargs.get("r_weights", MJX_R_WEIGHTS)
    return float(w["w_tilt"] * terms["r_tilt_gated"])


def mjx_mid_direction_progress_reward(state, plant, plant_context, **kwargs):
    terms = kwargs.get("_mjx_terms")
    if terms is None:
        terms = _mjx_reward_terms(state, plant, plant_context, **kwargs)
    return float(terms["r_dir"])


def mjx_retract_position_reward(state, plant, plant_context, **kwargs):
    terms = kwargs.get("_mjx_terms")
    if terms is None:
        terms = _mjx_reward_terms(state, plant, plant_context, **kwargs)
    return float(terms["r_retract"])


def mjx_action_rate_penalty(state, plant, plant_context, **kwargs):
    terms = kwargs.get("_mjx_terms")
    if terms is None:
        terms = _mjx_reward_terms(state, plant, plant_context, **kwargs)
    return float(terms["r_action_rate"])


def mjx_mid_achieved_indicator(state, plant, plant_context, **kwargs):
    terms = kwargs.get("_mjx_terms")
    if terms is None:
        terms = _mjx_reward_terms(state, plant, plant_context, **kwargs)
    return float(terms["mid_achieved_now"])


def mjx_mid_total_reward(state, plant, plant_context, **kwargs):
    terms = kwargs.get("_mjx_terms")
    if terms is None:
        terms = _mjx_reward_terms(state, plant, plant_context, **kwargs)
    return float(terms["r_mid_total"])


def mjx_binary_phase_reward(state, plant, plant_context, mid_done=False, **kwargs):
    terms = kwargs.get("_mjx_terms")
    if terms is None:
        terms = _mjx_reward_terms(state, plant, plant_context, **kwargs)
    core = terms["r_retract"] if bool(mid_done) else terms["r_mid_total"]
    return float(core + terms["r_action_rate"])
