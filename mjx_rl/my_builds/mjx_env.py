"""Skeleton MJX env with a PureJaxRL-compatible interface.

This file gives you a minimal env API that can be wrapped by
`MJXGymnaxWrapper` in `purejaxrl/purejaxrl/wrappers.py`.
"""

from flax import struct
import jax
import jax.numpy as jnp
from etils import epath
import mujoco
from mujoco import mjx
from typing import Any

@struct.dataclass
class MJXState:
    """Env state returned by `reset` and consumed by `step`."""
    data: Any           #! to hold mujoco data 
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray   #! keep done as array for JAX consistency
    t: jnp.ndarray
    action_v_prev: jnp.ndarray
    x_mid: jnp.ndarray  #! end-effector target position mid episode
    v_mid: jnp.ndarray  #! end-effector target velocity mid episode
    x_final: jnp.ndarray #! end-effector target position at the end of episode
    mid_done: jnp.ndarray  #! flag indicating if mid target is achieved

    # Add task-specific physics fields here, e.g.:
    # qpos: jnp.ndarray
    # qvel: jnp.ndarray


class MyMJXEnv():
    """Minimal single-env API expected by MJXGymnaxWrapper."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        episode_length: int = 1000,
        qpos_init: jnp.ndarray = None,
    ):
        self.observation_size = observation_size
        self.action_size = action_size
        self.episode_length = episode_length
        self.init_qpos = qpos_init if qpos_init is not None else jnp.zeros((7,), dtype=jnp.float32)

        # load the franka model #TODO: currently only arm, may need add object later
        FRANKA_ROOT_PATH = epath.Path('mujoco_menagerie/franka_emika_panda')

        self.mj_model = mujoco.MjModel.from_xml_path(
            (FRANKA_ROOT_PATH / 'panda_nohand.xml').as_posix())
        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        assert self.mj_model.nu == 7, (
            f"Expected 7 actuators (arm-only model), got nu={self.mj_model.nu}. "
            "Use a no-gripper Panda XML or update controller/action dimensions."
        )

        # overwrite the dummy body "attachment" in the panda_nohand.xml
        self.ee_site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "attachment")
        self.mj_model.site_pos[self.ee_site_id] = jnp.array([0.0, 0.0, 0.15], dtype=jnp.float32)

        #TODO: turn the reward config and weights into a separate config dict and pass in
        self.r_configs = {
            "alpha1": jnp.float32(1.0),
            "alpha2": jnp.float32(0.01),
            "alpha3": jnp.float32(0.01),
            "alpha4": jnp.float32(0.01),
            "k": jnp.float32(0.1),
            "eps1": jnp.float32(0.05), # cm
            "k_pos": jnp.float32(1.0),
            "k_vel": jnp.float32(1.0),
            "eps_pos": jnp.float32(0.05), # cm
            "eps_vel": jnp.float32(0.1), # m/s
            "bonus_mid": jnp.float32(5.0)
        }
        #TODO: these are some dummies for placeholder, change accordingly
        self.r_weights = {
           "w_pos_mid": jnp.float32(1.0),
           "w_vel_mid": jnp.float32(1.0),
           "w_pos_final": jnp.float32(1.0),
           "w_vel_progress": jnp.float32(1.0),
        }

        # mujoco sim dt
        self._dt_sim = 0.004
        self.mj_model.opt.timestep = self._dt_sim

        # policy dt
        self._dt_env = 0.02 # 50Hz

        # repeat action to match env_dt to sim_dt
        self.action_repeat = int(self._dt_env / self._dt_sim)

        #! in mujoco the controllers are prescribed in the xml, of different types
        #! such as <general>, <position>, <motor>, etc. Which cannot be changed 
        #! programmatically. The <general> one with biastype="affine" is used 
        #! for better customization:
        #!     tau = g0*ctrl + b0 + b1*qpos + b2*qvel
        #! where g0 is _gainprm[:, 0], b0, b1, b2 correspond to _biasprm[:, 0:3]
        #! this is essentially making the variable ctrl symbolic.
        #! with g0 = 1, and b0, b1, b2 = 0, the control is a direct torque control
        #! if ctrl is torque.

        #* using <general> actuator with biastype="affine" for direct control of 
        #* torques, with no bias terms 
        arm = slice(0, 7)
        self.mj_model.actuator_gainprm[arm, 0] = 1.0
        self.mj_model.actuator_biasprm[arm, :] = 0.0

        # joint 1-4 
        self.mj_model.actuator_ctrlrange[:4, 0] = -81.0
        self.mj_model.actuator_ctrlrange[:4, 1] = 81.0

        # joint 5-7
        self.mj_model.actuator_ctrlrange[4:7, 0] = -12.0
        self.mj_model.actuator_ctrlrange[4:7, 1] = 12.0

        #? not sure what these are doing, ignore for now
        self.mj_model.opt.iterations = 6
        self.mj_model.opt.ls_iterations = 6

        # u of shape (nu, 2)
        u_bounds = jnp.array(self.mj_model.actuator_ctrlrange, dtype=jnp.float32)
        self.u_low = u_bounds[:, 0]
        self.u_high = u_bounds[:, 1]
        self.mjx_model = mjx.put_model(self.mj_model)

        # v bounds for joints, mujoco does not have this natively
        self.v_high = jnp.array([2.1750, 2.1750, 2.1750, 2.1750, 2.61, 2.61, 2.61], dtype=jnp.float32)
        self.v_low = -self.v_high

        # get joint limits
        self.limit_margin = 1e-3
        self.qpos_low = jnp.array(self.mj_model.jnt_range[:, 0])
        self.qpos_high = jnp.array(self.mj_model.jnt_range[:, 1])
        
        # gain bounds 
        self.kp_high = jnp.array([200.0]*7, dtype=jnp.float32)
        self.kp_low = jnp.array([20.0]*7, dtype=jnp.float32)
        self.kd_high = 0.5 * jnp.sqrt(self.kp_high)
        self.kd_low = 2.0 * jnp.sqrt(self.kp_low)


    def reset(self, key, params=None):
        """Reset one env.

        Args:
            key: JAX PRNG key for this single env.
            params: optional static env params.
        Returns:
            MJXState with fields `obs`, `reward`, `done`.
        """
        del params

        data = mjx.make_data(self.mjx_model)
        data = data.replace(
            qpos=self.init_qpos,
            qvel=jnp.zeros_like(data.qvel),
            ctrl=jnp.zeros_like(data.ctrl)
        )

        #* do the forward step 
        #* forward computes the consistent quantities for the
        #* current state, such as kinematics, contact detection/constraints, bias terms
        #* etc. at current time
        #* 
        #* step calls on contact solver (solves for constrained accelerations and related
        #* forces), then  integrate forward in time.
        data = mjx.forward(self.mjx_model, data)

        #? data.qpos and data.qvel are 1D, axis=0 is not necessary
        obs = jnp.concatenate([data.qpos, data.qvel], axis=0).astype(jnp.float32)  

        return MJXState(
            data=data,
            obs=obs,
            reward=jnp.array(0.0, dtype=jnp.float32),
            done=jnp.array(False),
            t=jnp.array(0, dtype=jnp.int32),
            action_v_prev=jnp.zeros((7,), dtype=jnp.float32),
            mid_done=jnp.array((False, False), dtype=jnp.bool_)
        )

    def step(self, state: MJXState, action: jnp.ndarray, params=None):
        """Step one env.

        Replace this placeholder with your real MJX simulation step.
        """
        del params
        
        # converting action to jax array
        action = jnp.asarray(action, dtype=jnp.float32)
        action_in_v = self.v_low + 0.5*(self.v_high - self.v_low) * (action[:7] + 1.0)
        action_in_kp = self.kp_low + 0.5*(self.kp_high - self.kp_low) * (action[7:14] + 1.0)
        action_in_kd = self.kd_low + 0.5*(self.kd_high - self.kd_low) * (action[14:21] + 1.0)

        #! I am using velocity as action for my case, convert to torque here
        action_v_prev = self.v_low + 0.5*(self.v_high - self.v_low) * (state.action_v_prev + 1.0) #* convert to physical units
        q_ref = state.obs[:7] + action_v_prev * self._dt_env
        tau = action_in_kp * (q_ref - state.obs[:7]) + action_in_kd * (action_in_v - state.obs[7:14])
        tau = jnp.clip(tau, self.u_low, self.u_high)

        #! need to repeat the action in a JAX-compatible way
        def action_repeater(_, data):
            data = data.replace(ctrl=tau)
            data = mjx.step(self.mjx_model, data)
            return data 
        data_next = jax.lax.fori_loop(0, self.action_repeat, action_repeater, state.data)
        obs_next = jnp.concatenate([data_next.qpos, data_next.qvel], axis=0).astype(jnp.float32)
        
        # increment the step count  
        t = state.t + 1

        # Placeholder dynamics/reward.
        next_state = MJXState(
            data=data_next,
            obs=obs_next,
            reward=jnp.array(0.0, dtype=jnp.float32),
            done=jnp.array(False),
            t=t,
            action_v_prev=action[:7]
        ) #! damn! I forgot here the state got reinitialized therefore the env_state from 
        #! _terminal() with rewards and done=True won't be leaking through the new episode

        # next_reward = self._total_reward(self._reward_terms(state, action, next_state))
        next_reward = self._compute_rewards_binary(state, action, next_state)
        next_state = next_state.replace(reward=next_reward.astype(jnp.float32)) 

        # compute the termination 
        done = t >= self.episode_length

        # Optional auto-reset behavior at terminal.
        def _terminal(_):
            # compute the rewards
            # TODO: need to reset using RNG, only a placeholder here 
            reset_state = self.reset(jax.random.PRNGKey(0))
            return reset_state.replace(reward=next_reward.astype(jnp.float32),
                                       done=jnp.array(True)) 

        def _non_terminal(_):
            return next_state

        return jax.lax.cond(done, _terminal, _non_terminal, operand=None)
    
    def _compute_rewards_continuous(self, state, action, next_state):
        x_mid = state.x_mid 
        v_mid = state.v_mid
        x_final = state.x_final 
        x_now = next_state.data.site_xpos[self.ee_site_id]
        v_now = next_state.data.site_xvel[self.ee_site_id]

        # error 
        e_pos_sqr_mid = jnp.sum((x_now - x_mid)**2)
        e_vel_sqr_mid = jnp.sum((v_now - v_mid)**2)
        e_pos_sqr_final = jnp.sum((x_now - x_final)**2)

        #* reward 1: end-effector position target at mid episode
        r_pos_mid = jnp.exp(-self.r_configs["alpha1"] * e_pos_sqr_mid)
        r_vel_mid = jnp.exp(-self.r_configs["alpha3"] * e_vel_sqr_mid)
        r_vel_gated = r_pos_mid * r_vel_mid

        r_mid = (self.r_weights["w_pos_mid"]*r_pos_mid 
                 + self.r_weights["w_vel_mid"]*r_vel_gated)

        #* reward 3: retract after mid target achieved
        r_retract = jnp.exp(-self.r_configs["alpha4"] * e_pos_sqr_final)
        r_retract = self.r_weights["w_pos_final"] * r_retract

        #* soft success based both position and velocity
        pos_gate = 1.0 / (1.0 + jnp.exp(-self.r_configs["k_pos"] * (self.r_configs["eps_pos"] - e_pos_sqr_mid)))
        vel_gate = 1.0 / (1.0 + jnp.exp(-self.r_configs["k_vel"] * (self.r_configs["eps_vel"] - e_vel_sqr_mid)))
        mid_soft_success = pos_gate * vel_gate

        #* with switching
        reward = (1.0 - mid_soft_success) * r_mid + mid_soft_success * r_retract
        return reward

    #TODO consider pass reward functions in? does it complies to jax convention?
    def _compute_rewards_binary(self, state, action, next_state):
        x_mid = state.x_mid
        v_mid = state.v_mid
        x_final = state.x_final

        x_now = next_state.data.site_xpos[self.ee_site_id]
        v_now = next_state.data.site_xvel[self.ee_site_id]

        pos_err_mid = jnp.linalg.norm(x_now - x_mid)
        vel_err_mid = jnp.linalg.norm(v_now - v_mid)
        retract_err = jnp.linalg.norm(x_now - x_final)

        r_pos_mid = jnp.exp(-self.r_configs["alpha1"] * pos_err_mid**2)
        r_vel_mid = jnp.exp(-self.r_configs["alpha3"] * vel_err_mid**2)

        # Velocity only matters strongly when near the mid target
        r_vel_gated = r_pos_mid * r_vel_mid

        r_mid = (
            self.r_weights["w_pos_mid"] * r_pos_mid
            + self.r_weights["w_vel_mid"] * r_vel_gated
        )
        # TODO: add velocit direction reward 
        dir_speed = jnp.dot(v_now, v_mid)
        r_mid = r_mid + self.r_weights["w_vel_progress"] * dir_speed

        # Optional: action penalty
        r_ctrl = -self.r_configs["w_ctrl"] * jnp.sum(action**2)

        mid_achieved_now = (
            (pos_err_mid < self.r_configs["eps_pos"]) &
            (vel_err_mid < self.r_configs["eps_vel"])
        )

        # Persist success once achieved
        # state.mid_done should be a bool stored in your env state
        mid_done = state.mid_done | mid_achieved_now

        r_retract = self.r_weights["w_pos_final"] * jnp.exp(
            -self.r_configs["alpha4"] * retract_err**2
        )

        # Optional success bonus at the instant the mid target is first achieved
        first_mid_hit = (~state.mid_done) & mid_achieved_now
        r_mid_bonus = jnp.where(
            first_mid_hit,
            self.r_configs["bonus_mid"],
            0.0,
        )

        # TODO: added tilt reward


        reward = jnp.where(mid_done, r_retract, r_mid)
        reward = reward + r_ctrl + r_mid_bonus

        # Return both reward and updated phase flag
        return reward, mid_done


def make_mjx_env():
    """Factory used by training config."""
    return MyMJXEnv(observation_size=14, action_size=21, episode_length=1000)


# Example wiring with purejaxrl/purejaxrl/ppo_continuous_action.py:
# env = make_mjx_env()
# config = {
#     ...,
#     "ENV_BACKEND": "mjx",
#     "MJX_ENV": env,
#     "OBSERVATION_SIZE": env.observation_size,  # optional if fields exist
#     "ACTION_SIZE": env.action_size,            # optional if fields exist
# }

