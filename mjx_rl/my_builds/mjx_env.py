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
from reward_weights_and_configs import r_configs1, r_weights1

@struct.dataclass
class MJXState:
    """Env state returned by `reset` and consumed by `step`. 
       the types should all be jnp.ndarray as the envs are parallelized
       and states are batched """
    data: Any           #! to hold mujoco data 
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray   #! keep done as array for JAX consistency
    t: jnp.ndarray
    action_v_prev: jnp.ndarray
    x_ee_mid: jnp.ndarray  #! end-effector target position mid episode
    v_ee_mid: jnp.ndarray  #! end-effector target velocity mid episode
    x_ee_final: jnp.ndarray #! end-effector target position at the end of episode
    mid_done: jnp.ndarray #! flag indicating if mid target is achieved
    rng: jnp.ndarray
    done_info: jnp.ndarray #! shape (6,) for each done condition

class MyMJXEnv():
    """Minimal single-env API expected by MJXGymnaxWrapper."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        episode_length: int = None,
        qpos_init: jnp.ndarray = None,
        r_configs: dict = None,
        r_weights: dict = None,
    ):
        self.observation_size = observation_size
        self.action_size = action_size
        self.episode_length = episode_length

        # load the franka model #TODO: currently only arm, may need add object later
        FRANKA_ROOT_PATH = epath.Path('mujoco_menagerie/franka_emika_panda') # relative path, resolves to where I call the script

        self.mj_model = mujoco.MjModel.from_xml_path(
            (FRANKA_ROOT_PATH / 'panda_nohand.xml').as_posix())
        #! Newton is faster than CG for small systems without heavy contacts.
        #! CG is only better for large contact-rich scenes.
        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        assert self.mj_model.nu == 7, (
            f"Expected 7 actuators (arm-only model), got nu={self.mj_model.nu}. "
            "Use a no-gripper Panda XML or update controller/action dimensions."
        )

        self.ee_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "ee_point")

        """ may be useful later """
        # # overwrite the dummy body "attachment" in the panda_nohand.xml
        # self.ee_site_id = mujoco.mj_name2id(self.mj_model, 
        #                                     mujoco.mjtObj.mjOBJ_SITE, 
        #                                     "attachment_site")
        # self.ee_body_id = self.mj_model.site_bodyid[self.ee_site_id]
        # self.mj_model.site_pos[self.ee_site_id] = jnp.array([0.0, 0.0, 0.15], dtype=jnp.float32)

        assert r_configs is not None, "r_configs must be provided"
        assert r_weights is not None, "r_weights must be provided"

        self._r_configs = r_configs 
        self._r_weights = r_weights 

        # mujoco sim dt
        #! 0.004 → action_repeat=5 was overkill for a free arm with no contacts.
        #! 0.01 is stable for revolute-joint arm (PD ctrl natural freq ~10 rad/s,
        #! Nyquist limit ~0.31s >> 0.01). Halves MJX compute vs 0.004.
        self._dt_sim = 0.01
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
        
        # #
        # self.q_max = jnp.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=jnp.float32) 
        # self.q_min = jnp.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=jnp.float32)
        # self.v_max = jnp.array([2.1750, 2.1750, 2.1750, 2.1750, 2.61, 2.61, 2.61], dtype=jnp.float32)
        # self.v_min = -self.v_max

        # joint 5-7
        self.mj_model.actuator_ctrlrange[4:7, 0] = -12.0
        self.mj_model.actuator_ctrlrange[4:7, 1] = 12.0

        #! Constraint solver iterations. For a free arm with no contacts,
        #! the solver barely activates. 1 iteration is enough; reduce from 6
        #! to save ~5× solver overhead per substep.
        self.mj_model.opt.iterations = 1
        self.mj_model.opt.ls_iterations = 4

        # u of shape (nu, 2)
        u_bounds = jnp.array(self.mj_model.actuator_ctrlrange, dtype=jnp.float32)
        self.u_low = u_bounds[:, 0]
        self.u_high = u_bounds[:, 1]
        self.mjx_model = mjx.put_model(self.mj_model)

        # v bounds for joints, mujoco does not have this natively
        self.v_high = jnp.array([2.1750, 2.1750, 2.1750, 2.1750, 2.61, 2.61, 2.61], dtype=jnp.float32)
        self.v_low = -self.v_high

        # get joint position limits
        #! 1e-2 was too tight — PD controller overshoots by more than 0.01 rad/s
        #! in a single env step. 0.1 gives the policy room to learn to back off.
        self.limit_margin = 0.1
        self.qpos_low = jnp.array(self.mj_model.jnt_range[:, 0])
        self.qpos_high = jnp.array(self.mj_model.jnt_range[:, 1])

        if qpos_init is not None:
            qpos_init = jnp.asarray(qpos_init, dtype=jnp.float32)
            bounded_up = qpos_init < (self.qpos_high - self.limit_margin)
            bounded_low = qpos_init > (self.qpos_low + self.limit_margin)
            assert jnp.all(bounded_up & bounded_low), "qpos_init is out of bounds"
            self.init_qpos = qpos_init
        else:
            self.init_qpos = jnp.array([0.9207,  0.2574, -0.9527, -2.0683,  0.2799,  2.1147, 2.], dtype=jnp.float32)
        
        # gain bounds 
        self.kp_base = jnp.array([100.0]*7, dtype=jnp.float32)
        self.kd_base = 2.0 * jnp.sqrt(self.kp_base)
        self.dkp_max = jnp.array([100.0]*7, dtype=jnp.float32)
        self.dkd_max = jnp.array([20.0]*7, dtype=jnp.float32)

        #! Residual gain curriculum defined on normalized train progress [0, 1].
        #! Example: 0.6 -> 1.0 means "start enabling residual gains at 60% training
        #! progress and fully enable them by the end".
        self.gain_progress_start = jnp.float32(0.6)
        self.gain_progress_end = jnp.float32(1.0)

        # some quantities for reward computation 
        self._zhat_w_opp = jnp.array([0.0, 0.0, -1.0], dtype=jnp.float32) 
        self._zhat_w = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)

    def _ramp(self, x, x0, x1):
        return jnp.clip((x - x0) / (x1 - x0), 0.0, 1.0)

    def _build_obs(self, data, x_ee_mid, v_ee_mid, x_ee_final, mid_done):
        """Build the observation vector used by both reset and step.

        Layout:
        [qpos(7), qvel(7), dx_mid(3), dv_mid(3), dx_final(3), mid_done(1)] -> (24,)
        """
        x_ee = data.xpos[self.ee_body_id]
        v_ee = data.cvel[self.ee_body_id, 3:]
        return jnp.concatenate(
            [
                data.qpos,
                data.qvel,
                x_ee_mid - x_ee,
                v_ee_mid - v_ee,
                x_ee_final - x_ee,
                mid_done.astype(jnp.float32)[None],
            ],
            axis=0,
        ).astype(jnp.float32)

    def reset(self, key, params=None):
        """Reset one env.

        Args:
            key: JAX PRNG key for this single env.
            params: optional static env params.
        Returns:
            MJXState with fields `obs`, `reward`, `done`.
        """
        del params
        #TODO: domain randomization here

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

        key, key_vel = jax.random.split(key)

        x_ee_mid = jnp.array([0.4, 0.4, 0.05], dtype=jnp.float32)
        x_ee_final = jnp.array([0.4, 0.4, 0.25], dtype=jnp.float32)
        mid_done = jnp.array(False, dtype=jnp.bool_)

        # TODO: this range should be informed by the energy equation with the target x y theta
        v_ee_mid = jax.random.uniform(key_vel, 
                                   (3,), 
                                   minval=jnp.array([-0.5, -0.5, 0.]), 
                                   maxval=jnp.array([ 0.5,  0.5, 0.])).astype(jnp.float32)
        obs = self._build_obs(data, x_ee_mid, v_ee_mid, x_ee_final, mid_done)

        return MJXState(
            data=data,
            obs=obs,
            reward=jnp.array(0.0, dtype=jnp.float32),
            done=jnp.array(False),
            t=jnp.array(0, dtype=jnp.int32),
            action_v_prev=jnp.zeros((7,), dtype=jnp.float32),
            mid_done=mid_done,
            x_ee_final=x_ee_final,
            x_ee_mid=x_ee_mid,
            v_ee_mid=v_ee_mid,
            rng=key,
            done_info=jnp.zeros((6,), dtype=jnp.float32)
        )

    def step(self, state: MJXState, action: jnp.ndarray, params=None):
        """Step one env.

        Replace this placeholder with your real MJX simulation step.
        """

        train_progress = (
            jnp.asarray(params.get("train_progress", 0.0), dtype=jnp.float32)
            if params is not None
            else jnp.array(0.0, dtype=jnp.float32)
        )
        gain_progress_start = (
            jnp.asarray(
                params.get("gain_schedule_split", self.gain_progress_start),
                dtype=jnp.float32,
            )
            if params is not None
            else self.gain_progress_start
        )
        gain_progress_end = (
            jnp.asarray(
                params.get("gain_schedule_end", self.gain_progress_end),
                dtype=jnp.float32,
            )
            if params is not None
            else self.gain_progress_end
        )
        rng, rng_reset = jax.random.split(state.rng)

        action = jnp.asarray(action, dtype=jnp.float32)
        action_v = action[:7]
        action_dkp = action[7:14]
        action_dkd = action[14:21]
        
        # converting action to jax array
        #! Cap commanded velocity to 70% of joint limits.
        #! The policy outputs [-1, 1] which maps to full v_low..v_high.
        #! At extremes, the PD controller generates huge torques that overshoot
        #! velocity limits within one env step, causing instant termination.
        v_cmd_scale = 0.7
        v_low_cmd = v_cmd_scale * self.v_low
        v_high_cmd = v_cmd_scale * self.v_high
        action_in_v = v_low_cmd + 0.5*(v_high_cmd - v_low_cmd) * (action_v[:7] + 1.0)

        # residual gain with scheduling and clipping
        schedule = self._ramp(
            train_progress, gain_progress_start, gain_progress_end
        )
        kp = self.kp_base + schedule * (self.dkp_max * action_dkp)
        kd = self.kd_base + schedule * (self.dkd_max * action_dkd)

        kp = jnp.clip(kp, self.kp_base, self.kp_base + self.dkp_max)
        kd = jnp.clip(kd, self.kd_base, self.kd_base + self.dkd_max)

        #! I am using velocity as action for my case, convert to torque here
        action_v_prev = v_low_cmd + 0.5*(v_high_cmd - v_low_cmd) * (state.action_v_prev + 1.0) #* convert to physical units
        q_ref = state.obs[:7] + action_v_prev * self._dt_env #TODO: try using action_v than action_v_prev 
        tau = kp * (q_ref - state.obs[:7]) + kd * (action_in_v - state.obs[7:14]) + state.data.qfrc_bias
        tau = jnp.clip(tau, self.u_low, self.u_high)

        #! need to repeat the action in a JAX-compatible way
        def action_repeater(_, data):
            data = data.replace(ctrl=tau)
            data = mjx.step(self.mjx_model, data)
            return data 
        
        #TODO: need to randomize self.action_repeat for Sim-To-Real transfer
        data_next = jax.lax.fori_loop(0, self.action_repeat, action_repeater, state.data)

        #TODO: expanding gradually for curriculum learning.
        #* currently fixing the observation for x_mid, and x_final
        obs_next = self._build_obs(
            data_next,
            state.x_ee_mid,
            state.v_ee_mid,
            state.x_ee_final,
            state.mid_done,
        )
        
        # increment the step count  
        t = state.t + 1

        # Placeholder dynamics/reward.
        next_state = MJXState(
            data=data_next,
            obs=obs_next,
            reward=jnp.array(0.0, dtype=jnp.float32),
            done=jnp.array(False),
            t=t,
            action_v_prev=action[:7],
            x_ee_mid=state.x_ee_mid,
            v_ee_mid=state.v_ee_mid,
            x_ee_final=state.x_ee_final,
            mid_done=state.mid_done,
            rng=rng,
            done_info=jnp.zeros((6,), dtype=jnp.float32)
        ) #! damn! I forgot here the state got reinitialized therefore the env_state from 
        #! _terminal() with rewards and done=True won't be leaking through the new episode

        #? actions are velocities, not gains, should I pass tau for action penalty?
        next_reward, mid_done = self._compute_rewards_binary(state, action[:7], next_state, train_progress)
        next_state = next_state.replace(reward=next_reward.astype(jnp.float32),
                                        mid_done=mid_done) 

        # compute the termination 
        done_time = t >= self.episode_length
        done_noninfinite =  ~jnp.all(jnp.isfinite(next_state.obs))  # Terminate if any obs is non-finite
        done_qpos_low = jnp.any(next_state.obs[:7] < self.qpos_low + self.limit_margin) 
        done_qpos_high = jnp.any(next_state.obs[:7] > self.qpos_high - self.limit_margin)
        done_qvel_low = jnp.any(next_state.obs[7:14] < self.v_low + self.limit_margin)
        done_qvel_high = jnp.any(next_state.obs[7:14] > self.v_high - self.limit_margin)

        done = done_time | done_noninfinite | done_qpos_low | done_qpos_high | done_qvel_low | done_qvel_high
        next_state = next_state.replace(done=done)

        done_info = jnp.array([done_time, 
                               done_noninfinite, 
                               done_qpos_low, 
                               done_qpos_high, 
                               done_qvel_low, 
                               done_qvel_high], dtype=jnp.float32)
        next_state = next_state.replace(done_info=done_info)

        # Optional auto-reset behavior at terminal.
        def _terminal(_):
            # compute the rewards
            reset_state = self.reset(rng_reset)
            return reset_state.replace(reward=next_reward.astype(jnp.float32),
                                       done=jnp.array(True),
                                       done_info=done_info) 

        def _non_terminal(_):
            return next_state

        return jax.lax.cond(done, _terminal, _non_terminal, operand=None)
    
    def _compute_rewards_continuous(self, state, action, next_state):
        """ Smooth blending of two phases """
        pass

    def _compute_rewards_binary(self, state, action, next_state, train_progress=None):
        """ should always use next_state for reward """
        if train_progress is None:
            train_progress = jnp.array(1.0, dtype=jnp.float32)

        # Curriculum: anneal mid-target thresholds from easy → tight over training.
        # At progress=0: use eps_*_start (easy). At progress=1: use eps_*_end (tight).
        eps_pos = (
            self._r_configs["eps_pos_start"]
            + train_progress * (self._r_configs["eps_pos_end"] - self._r_configs["eps_pos_start"])
        )
        eps_vel = (
            self._r_configs["eps_vel_start"]
            + train_progress * (self._r_configs["eps_vel_end"] - self._r_configs["eps_vel_start"])
        )

        x_ee_mid = next_state.x_ee_mid
        v_ee_mid = next_state.v_ee_mid
        x_ee_final = next_state.x_ee_final

        x_next = next_state.data.xpos[self.ee_body_id]

        #! no need to compute explicitly, as mjx.step already gives it
        # # v_now (linear) of the end-effector using jacobian
        # jacp, _ = mjx.jac(self.mjx_model, next_state.data, 
        #                   next_state.data.site_xpos[self.ee_site_id],
        #                   self.ee_body_id)
        # # mjx.jac returns jacp with shape (nv, 3), so use qvel @ jacp.
        # v_next = next_state.data.qvel @ jacp
        v_next = next_state.data.cvel[self.ee_body_id, 3:] 

        pos_err_mid = jnp.linalg.norm(x_next - x_ee_mid)
        vel_err_mid = jnp.linalg.norm(v_next - v_ee_mid)
        retract_err = jnp.linalg.norm(x_next - x_ee_final)

        r_pos_mid = jnp.exp(-self._r_configs["alpha1"] * pos_err_mid**2)
        r_vel_mid = jnp.exp(-self._r_configs["alpha3"] * vel_err_mid**2)

        # Velocity only matters strongly when near the mid target
        r_vel_gated = r_pos_mid * r_vel_mid
            
        # tilting reward at mid episode
        R_w_ee = next_state.data.xmat[self.ee_body_id].reshape(3, 3)
        zhat_ee_w = R_w_ee[:, 2] 

        # v_ee_horizontal
        v_hori = v_next - jnp.dot(v_next, self._zhat_w) * self._zhat_w
        v_hori_norm = jnp.linalg.norm(v_hori)

        v_hori_hat = v_hori / (v_hori_norm + self._r_configs["eps_tilt"])
        z_tilted_raw = self._zhat_w_opp + self._r_configs["beta_tilt"] * v_hori_hat
        z_tilted = z_tilted_raw / (jnp.linalg.norm(z_tilted_raw) + self._r_configs["eps_tilt"]) 

        # compute the alignment
        align = jnp.dot(zhat_ee_w, z_tilted)

        # exp reward and gated 
        r_tilt = jnp.exp(-self._r_configs["alpha_tilt"] * (1.0 - align))
        r_tilt_gated = r_pos_mid * r_tilt

        r_mid = (
            self._r_weights["w_pos_mid"] * r_pos_mid
            + self._r_weights["w_vel_mid"] * r_vel_gated
            + self._r_weights["w_tilt"] * r_tilt_gated
        )

        # velocity direction at mid episode (use directional v to bound it)
        v_ee_mid_hat = v_ee_mid / (jnp.linalg.norm(v_ee_mid) + self._r_configs["eps_vel"])
        v_next_hat = v_next / (jnp.linalg.norm(v_next) + self._r_configs["eps_vel"])
        dir_speed = jnp.dot(v_next_hat, v_ee_mid_hat)

        # gating on speed and direction
        v_ee_mid_norm = jnp.linalg.norm(v_ee_mid)
        speed_gate = jnp.where(v_ee_mid_norm > self._r_configs["eps_vmag"], 1.0, 0.0)
        r_dir = r_pos_mid * self._r_weights["w_vel_progress"] * dir_speed * speed_gate 
        r_mid = r_mid + r_dir

        # Smoothness penalty on commanded joint velocity change (physical units).
        v_cmd_now = self.v_low + 0.5 * (self.v_high - self.v_low) * (action + 1.0)
        v_cmd_prev = self.v_low + 0.5 * (self.v_high - self.v_low) * (state.action_v_prev + 1.0)
        r_action_rate = -self._r_weights["w_action_rate"] * jnp.sum((v_cmd_now - v_cmd_prev) ** 2)

        # Penalty for approaching joint velocity limits.
        # Measures how close each joint is to its velocity bound (0 = center, 1 = at limit).
        qvel = next_state.obs[7:14]
        vel_ratio = jnp.abs(qvel) / self.v_high  # 0..1, 1 = at limit
        # Quadratic penalty that activates above 80% of limit
        vel_excess = jnp.clip(vel_ratio - 0.8, 0.0, None)
        r_vel_limit = -self._r_weights["w_vel_limit"] * jnp.sum(vel_excess ** 2)

        mid_achieved_now = (
            (pos_err_mid < eps_pos) &
            (vel_err_mid < eps_vel)
        )

        # Persist success once achieved
        mid_done = state.mid_done | mid_achieved_now

        r_retract = self._r_weights["w_pos_final"] * jnp.exp(
            -self._r_configs["alpha4"] * retract_err**2
        )

        # Soft blend: as EE approaches mid target, gradually mix in retract reward.
        # This avoids the hard binary switch that creates a reward cliff.
        # blend_factor goes 0→1 as pos_err_mid shrinks (uses same kernel as r_pos_mid).
        blend_factor = r_pos_mid  # already = exp(-alpha1 * pos_err_mid²)
        reward = (1.0 - blend_factor) * r_mid + blend_factor * (r_mid + r_retract)
        reward = reward + r_action_rate + r_vel_limit 

        # Return both reward and updated phase flag
        return reward, mid_done


def make_mjx_env():
    """Factory used by training config."""
    return MyMJXEnv(observation_size=24, 
                    action_size=21, 
                    episode_length=200, 
                    r_configs=r_configs1, 
                    r_weights=r_weights1)


# Example wiring with purejaxrl/purejaxrl/ppo_continuous_action.py:
# env = make_mjx_env()
# config = {
#     ...,
#     "ENV_BACKEND": "mjx",
#     "MJX_ENV": env,
#     "OBSERVATION_SIZE": env.observation_size,  # optional if fields exist
#     "ACTION_SIZE": env.action_size,            # optional if fields exist
# }
