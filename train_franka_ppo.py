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
_ = env.step(
    s,
    a,
    {
        "train_step": jnp.array(0, dtype=jnp.int32),
        "train_progress": jnp.array(0.0, dtype=jnp.float32),
    },
)
print("env smoke: OK")

#! per training data collected: NUM_ENVS * NUM_STEPS, e.g. 256 * 200 = 51200
#! minibatch size: NUM_ENVS * NUM_STEPS // NUM_MINIBATCHES, e.g. 51200 // 32 = 1600
#! per training updates (gradient steps): NUM_MINIBATCHES * UPDATE_EPOCHS
#! total training loops: TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS)

# TOTAL_TIMESTEPS = 20 * 256 * 500 # = 2_560_000, i.e. ~2.56M env transitions
TOTAL_TIMESTEPS = 20 * 400 * 500 # = 4_000_000, i.e. ~4M env transitions

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_FILE = CHECKPOINT_DIR / "train_franka_ppo.msgpack"
CHECKPOINT_META_FILE = CHECKPOINT_DIR / "train_franka_ppo.meta.json"

config_base = {
    "LR": 3e-4,
    "NUM_ENVS": 400,           # 1080Ti has 11GB; 2048 OOMs
    "NUM_STEPS": 20,
    "TOTAL_TIMESTEPS": TOTAL_TIMESTEPS,
    
    #* Keep a stable total-timestep target for progress scheduling across resumes.
    "TOTAL_TIMESTEPS_TARGET": TOTAL_TIMESTEPS,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 8,     # minibatch = 256*64/8 = 2048
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
    
    #* Host debug callback every N PPO updates (1 = every update).
    #* Higher = fewer GPU→host sync stalls. 50 is a good balance.
    "DEBUG_PRINT_INTERVAL_UPDATES": 50,
    "COLLECT_METRICS": False,
    "WANDB_LOG": True,
    
    #* Host wandb callback every N PPO updates (1 = every update).
    #* Each callback is a jax.debug.callback → GPU stall + HTTP request.
    "WANDB_LOG_INTERVAL_UPDATES": 50,
    "WANDB_PROJECT": "my_rl",
    "WANDB_RUN_NAME": "franka_ppo_v1",
    
    #* Lower unroll speeds up compile time (often at some runtime cost).
    "GAE_SCAN_UNROLL": 8,
    
    #* Periodic checkpoint every N PPO updates (inside the scan, no recompile).
    #* Each checkpoint is a full GPU→host→disk round trip. Keep infrequent.
    #* 0 = disabled.
    "CHECKPOINT_INTERVAL_UPDATES": 500,
    
    #* Print approximate per-update split: compute vs host callbacks.
    "PROFILE_CALLBACK_OVERHEAD": True,

    #* PD gain as actions, scheduling
    "GAIN_SCHEDULE_SPLIT": 0.5,
    "GAIN_SCHEDULE_END": 0.8,
}

#! With NUM_STEPS = 20, set GAE_SCAN_UNROLL to 4.
#! 20 / 4 = 5 rolled iterations — small enough that loop overhead is negligible
#! Unroll 4 keeps the HLO IR compact (only 4 copies of the GAE body), so compile time stays low
#! Unroll 1 (your current setting) is overly conservative and adds minor loop overhead per step
#! Unroll 16 or 20 (full unroll) would work at this horizon but bloats compile IR for no meaningful runtime gain


def _load_done_env_steps() -> int:
    if not CHECKPOINT_META_FILE.exists():
        return 0
    with CHECKPOINT_META_FILE.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return int(meta.get("env_transition_step", 0))


def _save_checkpoint_from_out(out, target_total_timesteps: int) -> int:
    """Save checkpoint from the final training output dict."""
    runner_state = out["runner_state"]
    return _save_checkpoint_core(
        runner_state[0], runner_state[4], target_total_timesteps
    )


def _new_perf_stats() -> dict:
    return {
        "wandb_calls": 0,
        "wandb_host_sec": 0.0,
        "debug_calls": 0,
        "debug_host_sec": 0.0,
        "checkpoint_calls": 0,
        "checkpoint_host_sec": 0.0,
    }


def _print_perf_breakdown(execute_sec: float, num_updates: int, perf_stats: dict | None):
    if perf_stats is None or num_updates <= 0:
        return

    wandb_sec = float(perf_stats.get("wandb_host_sec", 0.0))
    debug_sec = float(perf_stats.get("debug_host_sec", 0.0))
    ckpt_sec = float(perf_stats.get("checkpoint_host_sec", 0.0))
    callback_sec = wandb_sec + debug_sec + ckpt_sec
    approx_compute_sec = max(execute_sec - callback_sec, 0.0)

    per_update_total_ms = 1000.0 * execute_sec / num_updates
    per_update_callback_ms = 1000.0 * callback_sec / num_updates
    per_update_compute_ms = 1000.0 * approx_compute_sec / num_updates
    callback_pct = 100.0 * callback_sec / execute_sec if execute_sec > 0 else 0.0

    print("\ncallback timing (approx):")
    print(
        f"  per-update total:    {per_update_total_ms:.3f} ms "
        f"({num_updates} updates)"
    )
    print(f"  per-update compute:  {per_update_compute_ms:.3f} ms")
    print(
        f"  per-update callbacks:{per_update_callback_ms:.3f} ms "
        f"({callback_pct:.1f}% of execute)"
    )
    print(
        "  callback totals:"
        f" wandb={wandb_sec:.3f}s ({int(perf_stats.get('wandb_calls', 0))} calls),"
        f" debug={debug_sec:.3f}s ({int(perf_stats.get('debug_calls', 0))} calls),"
        f" checkpoint={ckpt_sec:.3f}s ({int(perf_stats.get('checkpoint_calls', 0))} calls)"
    )


def _save_checkpoint_core(
    train_state,
    global_train_step,
    target_total_timesteps,
    perf_stats: dict | None = None,
    include_in_profile: bool = False,
) -> int:
    """Shared logic for saving a checkpoint."""
    t0 = time.perf_counter()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
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
    print(f"  periodic checkpoint: {env_transition_step}/{int(target_total_timesteps)} env transitions")
    if include_in_profile and perf_stats is not None:
        perf_stats["checkpoint_calls"] = int(perf_stats.get("checkpoint_calls", 0)) + 1
        perf_stats["checkpoint_host_sec"] = float(
            perf_stats.get("checkpoint_host_sec", 0.0)
        ) + (time.perf_counter() - t0)
    return env_transition_step


def _make_checkpoint_callback(target_total_timesteps: int, perf_stats: dict | None = None):
    """Return a callback for jax.debug.callback inside the training scan."""
    def _cb(train_state, global_train_step):
        _save_checkpoint_core(
            train_state,
            global_train_step,
            target_total_timesteps,
            perf_stats=perf_stats,
            include_in_profile=True,
        )
    return _cb


def _run_once(remaining_timesteps: int, resume: bool):
    run_config = dict(config_base)
    run_config["TOTAL_TIMESTEPS"] = int(remaining_timesteps)
    if resume and CHECKPOINT_FILE.exists():
        run_config["RESUME_CHECKPOINT_PATH"] = str(CHECKPOINT_FILE)
    num_updates = (
        run_config["TOTAL_TIMESTEPS"]
        // run_config["NUM_STEPS"]
        // run_config["NUM_ENVS"]
    )

    perf_stats = None
    if run_config.get("PROFILE_CALLBACK_OVERHEAD", False):
        perf_stats = _new_perf_stats()
        run_config["PROFILE_PERF"] = True
        run_config["PERF_STATS"] = perf_stats
    else:
        run_config["PROFILE_PERF"] = False

    # Inject checkpoint callback (called from jax.debug.callback inside scan).
    if run_config.get("CHECKPOINT_INTERVAL_UPDATES", 0) > 0:
        run_config["CHECKPOINT_FN"] = _make_checkpoint_callback(
            TOTAL_TIMESTEPS, perf_stats=perf_stats
        )

    train_fn = jax.jit(make_train(run_config))

    compile_key = jax.random.PRNGKey(1000)
    t0 = time.perf_counter()
    compiled_train_fn = train_fn.lower(compile_key).compile()
    compile_sec = time.perf_counter() - t0
    print(f"\ncompile sec: {compile_sec:.2f}")

    run_key = jax.random.PRNGKey(2000)
    t1 = time.perf_counter()
    out = compiled_train_fn(run_key)
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, out
    )
    execute_sec = time.perf_counter() - t1
    print(f"\nexecute sec: {execute_sec:.2f}")
    _print_perf_breakdown(execute_sec, num_updates, perf_stats)
    return out


if config_base.get("WANDB_LOG", False):
    import wandb

    wandb_config = {}
    for key, value in config_base.items():
        if key == "MJX_ENV":
            continue
        if isinstance(value, (str, int, float, bool)):
            wandb_config[key] = value

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
    remaining = TOTAL_TIMESTEPS - done_env_steps
    out = _run_once(
        remaining_timesteps=remaining,
        resume=CHECKPOINT_FILE.exists(),
    )
    done_env_steps = _save_checkpoint_from_out(out, TOTAL_TIMESTEPS)
    print(
        f"checkpoint saved: {done_env_steps}/{TOTAL_TIMESTEPS} env transitions "
        f"-> {CHECKPOINT_FILE}"
    )

print("ppo training complete")

if config_base.get("WANDB_LOG", False):
    import wandb

    wandb.finish()
