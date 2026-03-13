import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/jax_compilation_cache")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
os.makedirs(os.environ["JAX_COMPILATION_CACHE_DIR"], exist_ok=True)

import jax
import jax.numpy as jnp
from flax import serialization

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

TOTAL_TIMESTEPS = 200 * 128 * 100  # = 2,560,000
CHECKPOINT_CHUNK_TIMESTEPS = 200 * 128 * 10
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_FILE = CHECKPOINT_DIR / "train_franka_ppo.msgpack"
CHECKPOINT_META_FILE = CHECKPOINT_DIR / "train_franka_ppo.meta.json"

config_base = {
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


def _load_done_env_steps() -> int:
    if not CHECKPOINT_META_FILE.exists():
        return 0
    with CHECKPOINT_META_FILE.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return int(meta.get("env_transition_step", 0))


def _save_checkpoint(out, target_total_timesteps: int) -> int:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    runner_state = out["runner_state"]
    train_state = runner_state[0]
    global_train_step = runner_state[4]
    payload = {
        "train_state": train_state,
        "global_train_step": global_train_step,
    }
    CHECKPOINT_FILE.write_bytes(serialization.to_bytes(payload))

    global_train_step_int = int(jax.device_get(global_train_step))
    env_transition_step = global_train_step_int * config_base["NUM_ENVS"]
    meta = {
        "global_train_step": global_train_step_int,
        "env_transition_step": env_transition_step,
        "target_total_timesteps": int(target_total_timesteps),
        "saved_at_unix": time.time(),
    }
    with CHECKPOINT_META_FILE.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return env_transition_step


def _run_chunk(chunk_timesteps: int, chunk_idx: int, resume: bool):
    chunk_config = dict(config_base)
    chunk_config["TOTAL_TIMESTEPS"] = int(chunk_timesteps)
    if resume and CHECKPOINT_FILE.exists():
        chunk_config["RESUME_CHECKPOINT_PATH"] = str(CHECKPOINT_FILE)

    train_fn = jax.jit(make_train(chunk_config))

    compile_key = jax.random.PRNGKey(1000 + chunk_idx)
    t0 = time.perf_counter()
    compiled_train_fn = train_fn.lower(compile_key).compile()
    compile_sec = time.perf_counter() - t0
    print(f"\nchunk {chunk_idx} compile sec: {compile_sec:.2f}")

    run_key = jax.random.PRNGKey(2000 + chunk_idx)
    t1 = time.perf_counter()
    out = compiled_train_fn(run_key)
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, out
    )
    execute_sec = time.perf_counter() - t1
    print(f"\nchunk {chunk_idx} execute sec: {execute_sec:.2f}")
    return out


if config_base.get("WANDB_LOG", False):
    import wandb

    wandb_config = {}
    for key, value in config_base.items():
        if key == "MJX_ENV":
            continue
        if isinstance(value, (str, int, float, bool)):
            wandb_config[key] = value
    wandb_config["CHECKPOINT_CHUNK_TIMESTEPS"] = CHECKPOINT_CHUNK_TIMESTEPS

    wandb_init_kwargs = {
        "project": config_base.get("WANDB_PROJECT", "my_rl"),
        "config": wandb_config,
        "resume": "allow",
    }
    run_name = config_base.get("WANDB_RUN_NAME")
    if run_name:
        wandb_init_kwargs["name"] = run_name
    wandb.init(**wandb_init_kwargs)

done_env_steps = _load_done_env_steps()
if done_env_steps >= TOTAL_TIMESTEPS:
    print(f"training already complete at {done_env_steps}/{TOTAL_TIMESTEPS} timesteps")
else:
    chunk_idx = 0
    while done_env_steps < TOTAL_TIMESTEPS:
        remaining = TOTAL_TIMESTEPS - done_env_steps
        chunk_timesteps = min(CHECKPOINT_CHUNK_TIMESTEPS, remaining)
        out = _run_chunk(
            chunk_timesteps=chunk_timesteps,
            chunk_idx=chunk_idx,
            resume=CHECKPOINT_FILE.exists(),
        )
        done_env_steps = _save_checkpoint(out, TOTAL_TIMESTEPS)
        print(
            f"checkpoint saved: {done_env_steps}/{TOTAL_TIMESTEPS} env transitions "
            f"-> {CHECKPOINT_FILE}"
        )
        chunk_idx += 1

print("ppo training complete")

if config_base.get("WANDB_LOG", False):
    import wandb

    wandb.finish()
