"""
Train a policy for manipuolation.gym.envs.box_flipup

Example usage:

python solutions/notebooks/rl/train_box_flipup.py --checkpoint_freq 100000 --wandb
"""

#TODO: this is adopted from Tedrake's code, adapt to mine later

import argparse
import os
import sys
from pathlib import Path

import gymnasium as gym
import torch
import wandb

# `multiprocessing` also provides this method, but empirically `psutil`'s
# version seems more reliable.
from psutil import cpu_count
from pydrake.all import StartMeshcat
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EveryNTimesteps,
    ProgressBarCallback,
)
from stable_baselines3.common.env_checker import check_env

''' Noticing this is for vectorizing envs to be run on all CPU cores '''
from stable_baselines3.common.env_util import make_vec_env 

from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

# register the env to gym
gym.envs.register(
    id="panda-reach-v0",
    entry_point=("envs.franka_reach.franka_reach_drake:PandaReachEnv"), # need to modify this later to point to my env
)

class OffsetCheckpointCallback(BaseCallback):
    """
    Saves checkpoints with a global step count that includes an offset, so that
    resumed training from, e.g., 3,000,000 steps will save checkpoints named
    with accumulated steps (e.g., 4,000,000 after 1,000,000 more steps).

    This callback is intended to be wrapped by EveryNTimesteps for frequency control.
    """

    def __init__(
        self,
        save_path: Path,
        name_prefix: str,
        expected_resume_steps: int | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.name_prefix = name_prefix
        self.expected_resume_steps = expected_resume_steps

    def _on_step(self) -> bool:
        # Determine effective offset only at save-time to avoid relying on construction order.
        loaded_steps = int(getattr(self.model, "num_timesteps", 0))
        offset = 0
        if (
            self.expected_resume_steps is not None
            and loaded_steps < self.expected_resume_steps
        ):
            offset = int(self.expected_resume_steps)
        total_steps = offset + loaded_steps
        ckpt_path = self.save_path / f"{self.name_prefix}_{total_steps}_steps.zip"
        if self.verbose > 0:
            print(f"Saving checkpoint to {ckpt_path}")
        self.model.save(str(ckpt_path))
        return True

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--train_single_env", action="store_true")
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--checkpoint_freq", type=int, default=100_000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--resume_steps",
        type=int,
        default=None,
        help="If set, resume from checkpoint at this timestep (e.g., 3000000).",
    )
    parser.add_argument(
        "--log_path",
        help="path to the logs directory.",
        default="logs/panda_reach_ppo",
    )
    parser.add_argument(
        "--custom_net",
        action="store_true",
        help="Use custom network architecture [256, 256] with ReLU instead of default [64, 64] with Tanh.",
    )
    parser.add_argument(
        "--net_arch",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Hidden layer sizes for custom network (e.g., --net_arch 256 256 128).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device to use: 'auto' (detect), 'cpu', 'mps' (Apple Silicon), or 'cuda'.",
    )
    args = parser.parse_args()

    # Set up policy kwargs for custom network
    if args.custom_net:
        policy_kwargs = dict(
            net_arch=args.net_arch,
            activation_fn=torch.nn.ReLU,
        )
    else:
        policy_kwargs = None  # use SB3 defaults: [64, 64] with Tanh

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 5e5 if not args.test else 5,
        "env_name": "panda-reach-v0",
        "env_time_limit": 10 if not args.test else 0.5,
        "local_log_dir": args.log_path,
        "observations": "state",
        "net_arch": args.net_arch if args.custom_net else [64, 64],
        "custom_net": args.custom_net,
    }

    if args.wandb:
        run = wandb.init(
            project="panda-reach-v0",
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload videos
            save_code=True,
        )
    else:
        run = wandb.init(mode="disabled")

    # Where to put checkpoints
    ckpt_dir = Path(args.log_path).parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save a checkpoint when this callback is called.
    # We'll call it via EveryNTimesteps so save_freq can be 1.
    if True:
        checkpoint_cb = OffsetCheckpointCallback(
            save_path=ckpt_dir,
            name_prefix="ppo_panda_reach",
            expected_resume_steps=args.resume_steps,
        )

    # Trigger the checkpoint exactly every 50,000 timesteps (robust to n_envs)
    every_n_timesteps = EveryNTimesteps(
        n_steps=args.checkpoint_freq, callback=checkpoint_cb
    )

    # Combine with your existing Wandb callback
    callbacks = CallbackList(
        [WandbCallback(), every_n_timesteps, ProgressBarCallback()]
    )

    zip = f"data/panda_reach_ppo_{config['observations']}.zip"
    Path("data").mkdir(parents=True, exist_ok=True)

    num_cpu = int(cpu_count() / 2) if not args.test else 2
    if args.train_single_env:
        meshcat = StartMeshcat()
        env = gym.make(
            "panda-reach-v0",
            meshcat=meshcat,
            observations=config["observations"],
            time_limit=config["env_time_limit"],
            device=args.device,
        )
        check_env(env)
        input("Open meshcat (optional). Press Enter to continue...")
    else:
        # Use a callback so that the forked process imports and registers the environment.
        # Capture device in closure
        device = args.device
        
        def make_pandareach():
            import sys
            import os
            # Add examples dir to path so imports work in subprocess
            examples_dir = os.path.dirname(os.path.abspath(__file__))
            if examples_dir not in sys.path:
                sys.path.insert(0, examples_dir)
            # Add drake-gym root for drake_gym module
            drake_gym_root = os.path.dirname(os.path.dirname(examples_dir))
            if drake_gym_root not in sys.path:
                sys.path.insert(0, drake_gym_root)
            
            import gymnasium as gym
            # Register env in subprocess
            if "panda-reach-v0" not in gym.envs.registry:
                gym.envs.register(
                    id="panda-reach-v0",
                    entry_point="envs.franka_reach.franka_reach_drake:PandaReachEnv",
                )
            return gym.make(
                "panda-reach-v0",
                observations=config["observations"],
                time_limit=config["env_time_limit"],
                device=device,
            )

        env = make_vec_env(
            make_pandareach,
            n_envs=num_cpu,
            seed=0,
            vec_env_cls=SubprocVecEnv,
        )

    if args.test:
        model = PPO("MlpPolicy", env, n_steps=4, n_epochs=2, batch_size=8, policy_kwargs=policy_kwargs)
    elif (
        args.resume_steps is not None
        and (ckpt_dir / f"ppo_panda_reach_{args.resume_steps}_steps.zip").exists()
    ):
        print(f"Loading checkpoint at {args.resume_steps} steps")
        model = PPO.load(
            str(ckpt_dir / f"ppo_panda_reach_{args.resume_steps}_steps.zip"),
            env,
            verbose=1,
            tensorboard_log=f"runs/{run.id}",
            device=args.device,
        )
    elif os.path.exists(zip):
        model = PPO.load(
            zip, env, verbose=1, tensorboard_log=f"runs/{run.id}", device=args.device
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"runs/{run.id}",
            device=args.device,
            policy_kwargs=policy_kwargs,
        )

    model.learn(
        total_timesteps=3e6 if not args.test else 4,
        callback=callbacks,
    )
    model.save(zip)


if __name__ == "__main__":
    sys.exit(main())