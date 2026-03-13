# python - <<'PY'
import sys, jax, jax.numpy as jnp
sys.path += ["purejaxrl/purejaxrl", "mjx_rl/my_builds"]

from mjx_env import make_mjx_env
from ppo_continuous_action import make_train

env = make_mjx_env()
s = env.reset(jax.random.PRNGKey(0))
a = jnp.zeros((env.action_size,), dtype=jnp.float32)
_ = env.step(s, a, {"train_step": jnp.array(0, dtype=jnp.int32)})
print("env smoke: OK")

config = {
    "LR": 3e-4,
    "NUM_ENVS": 8,
    "NUM_STEPS": 8,
    "TOTAL_TIMESTEPS": 8 * 8 * 2,
    "UPDATE_EPOCHS": 1,
    "NUM_MINIBATCHES": 2,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.0,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh",
    "ENV_BACKEND": "mjx",
    "MJX_ENV": env,
    "OBSERVATION_SIZE": env.observation_size,
    "ACTION_SIZE": env.action_size,
    "ANNEAL_LR": False,
    "NORMALIZE_ENV": False,
    "DEBUG": False,
}
train_fn = jax.jit(make_train(config))
out = train_fn(jax.random.PRNGKey(0))
jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, out)
print("ppo smoke: OK")
# PY
