"""Skeleton MJX env with a PureJaxRL-compatible interface.

This file gives you a minimal env API that can be wrapped by
`MJXGymnaxWrapper` in `purejaxrl/purejaxrl/wrappers.py`.
"""

from flax import struct
import jax
import jax.numpy as jnp


@struct.dataclass
class MJXState:
    """Env state returned by `reset` and consumed by `step`."""

    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    t: jnp.ndarray
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
    ):
        self.observation_size = observation_size
        self.action_size = action_size
        self.episode_length = episode_length

        #TODO: these are some dummies for placeholder, change accordingly
        self.reward_weights = {
           "forward": 1.0,
           "ctrl_cost": 0.01,
        }

    def reset(self, key, params=None):
        """Reset one env.

        Args:
            key: JAX PRNG key for this single env.
            params: optional static env params.
        Returns:
            MJXState with fields `obs`, `reward`, `done`.
        """
        del params
        obs = jnp.zeros((self.observation_size,), dtype=jnp.float32)
        return MJXState(
            obs=obs,
            reward=jnp.array(0.0, dtype=jnp.float32),
            done=jnp.array(False),
            t=jnp.array(0, dtype=jnp.int32),
        )

    def step(self, state: MJXState, action: jnp.ndarray, params=None):
        """Step one env.

        Replace this placeholder with your real MJX simulation step.
        """
        del params
        action = jnp.asarray(action, dtype=jnp.float32)
        t = state.t + 1

        # Placeholder dynamics/reward.
        next_state = MJXState(
            obs=jnp.zeros((self.observation_size,), dtype=jnp.float32),
            reward=jnp.array(0.0, dtype=jnp.float32),
            done=jnp.array(False),
            t=t,
        )
        next_obs = next_state.obs
        reward = self._total_reward(self._reward_terms(state, action, next_state))
        done = t >= self.episode_length

        # Optional auto-reset behavior at terminal.
        def _terminal(_):
            # TODO: need to reset using RNG, only a placeholder here 
            return self.reset(jax.random.PRNGKey(0))

        def _non_terminal(_):
            return MJXState(
                obs=next_obs,
                reward=reward.astype(jnp.float32),
                done=done,
                t=t,
            )

        return jax.lax.cond(done, _terminal, _non_terminal, operand=None)
    
    def _reward_terms(self, state, action, next_state, **kwargs):
        del kwargs
        forward = (next_state.obs[0] - state.obs[0]).astype(jnp.float32)
        ctrl_cost = jnp.sum(action * action, dtype=jnp.float32)
        return {
            "forward": forward, 
            "ctrl_cost": ctrl_cost
        }
    
    def _total_reward(self, reward_terms):
        return sum(self.reward_weights[key] * reward_terms[key] 
                   for key in self.reward_weights)


def make_mjx_env():
    """Factory used by training config."""
    return MyMJXEnv(observation_size=17, action_size=6, episode_length=1000)


# Example wiring with purejaxrl/purejaxrl/ppo_continuous_action.py:
# env = make_mjx_env()
# config = {
#     ...,
#     "ENV_BACKEND": "mjx",
#     "MJX_ENV": env,
#     "OBSERVATION_SIZE": env.observation_size,  # optional if fields exist
#     "ACTION_SIZE": env.action_size,            # optional if fields exist
# }
