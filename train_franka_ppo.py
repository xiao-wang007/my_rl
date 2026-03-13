import os
import sys
import time

os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/jax_compilation_cache")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
os.makedirs(os.environ["JAX_COMPILATION_CACHE_DIR"], exist_ok=True)

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

#! per training data collected: NUM_ENVS * NUM_STEPS, e.g. 256 * 200 = 51200
#! minibatch size: NUM_ENVS * NUM_STEPS // NUM_MINIBATCHES, e.g. 51200 // 32 = 1600
#! per training updates (gradient steps): NUM_MINIBATCHES * UPDATE_EPOCHS
#! total training loops: TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS)

TOTAL_TIMESTEPS = 200 * 128 * 100 # = 2,560,000

config = {
    "LR": 3e-4,
    "NUM_ENVS": 128,
    "NUM_STEPS": 200,
    "TOTAL_TIMESTEPS": TOTAL_TIMESTEPS,
    "UPDATE_EPOCHS": 1,
    "NUM_MINIBATCHES": 32,
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
    "DEBUG": True,
    "COLLECT_METRICS": False,
    "WANDB_LOG": True,
    "WANDB_PROJECT": "my_rl",
    "WANDB_RUN_NAME": "train_franka_ppo",
    # Lower unroll speeds up compile time (often at some runtime cost).
    "GAE_SCAN_UNROLL": 1,
}

if config.get("WANDB_LOG", False):
    import wandb

    wandb_config = {}
    for key, value in config.items():
        if key == "MJX_ENV":
            continue
        if isinstance(value, (str, int, float, bool)):
            wandb_config[key] = value

    wandb_init_kwargs = {
        "project": config.get("WANDB_PROJECT", "my_rl"),
        "config": wandb_config,
    }
    run_name = config.get("WANDB_RUN_NAME")
    if run_name:
        wandb_init_kwargs["name"] = run_name
    wandb.init(**wandb_init_kwargs)

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

if config.get("WANDB_LOG", False):
    import wandb

    wandb.finish()
