# python - <<'PY'
import sys
import time

import jax
import jax.numpy as jnp
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
    "NUM_ENVS": 2048,
    "NUM_STEPS": 200,
    "TOTAL_TIMESTEPS": 200 * 2048 * 1,
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

compile_key = jax.random.PRNGKey(0)
t0 = time.perf_counter()
compiled_train_fn = train_fn.lower(compile_key).compile()
compile_sec = time.perf_counter() - t0
print(f"compile sec: {compile_sec:.2f}")

run_key = jax.random.PRNGKey(1)
t1 = time.perf_counter()
out = compiled_train_fn(run_key)
jax.tree_util.tree_map(
    lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, out
)
execute_sec = time.perf_counter() - t1
print(f"execute sec: {execute_sec:.2f}")

print("ppo smoke: OK")
# PY
