"""
Test a trained PPO policy on the Panda reach environment.

Usage:
    python test_policy.py
    python test_policy.py --model_path path/to/model.zip
    python test_policy.py --episodes 5 --render
"""

import argparse
import sys
import os

# Add paths for imports
_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
_DRAKE_GYM_ROOT = os.path.dirname(os.path.dirname(_EXAMPLES_DIR))
sys.path.insert(0, _DRAKE_GYM_ROOT)

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from pydrake.all import StartMeshcat, Rgba, RigidTransform
from pydrake.geometry import Sphere

# Register the environment
gym.envs.register(
    id="panda-reach-v0",
    entry_point="envs.franka_reach.franka_reach_drake:PandaReachEnv",
)


def visualize_target(meshcat, goal_pos, radius=0.05):
    """Visualize target position as a red sphere in Meshcat."""
    meshcat.SetObject(
        "target",
        Sphere(radius),
        Rgba(1.0, 0.0, 0.0, 0.7)  # Red, semi-transparent
    )
    meshcat.SetTransform(
        "target",
        RigidTransform(goal_pos)
    )


def main():
    parser = argparse.ArgumentParser(description="Test trained PPO policy")
    parser.add_argument(
        "--model_path",
        type=str,
        default="data/panda_reach_ppo_state.zip",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=True,
        help="Render with Meshcat visualization",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        default=False,
        help="Use stochastic actions (with exploration noise). Default is deterministic.",
    )
    args = parser.parse_args()
    
    # Deterministic is the opposite of stochastic
    deterministic = not args.stochastic

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}")
        print("Available models in data/:")
        if os.path.exists("data"):
            for f in os.listdir("data"):
                if f.endswith(".zip"):
                    print(f"  - data/{f}")
        return 1

    # Create environment
    if args.render:
        meshcat = StartMeshcat()
        env = gym.make(
            "panda-reach-v0",
            meshcat=meshcat,
            time_limit=5,
            debug=True,
        )
        input("Open Meshcat URL in browser, then press Enter to start...")
    else:
        env = gym.make("panda-reach-v0", time_limit=5)

    # Load the trained model
    print(f"Loading model from {args.model_path}")
    model = PPO.load(args.model_path)

    # Run episodes
    for episode in range(args.episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0

        # Visualize target position after reset (goal gets randomized)
        if args.render and hasattr(env.unwrapped, 'goal_state'):
            goal_pos = env.unwrapped.goal_state.goal_pos
            visualize_target(meshcat, goal_pos)
            print(f"Target position: {goal_pos}")

        print(f"\n=== Episode {episode + 1}/{args.episodes} ===")

        if args.render:
            meshcat.StartRecording()

        action_traj = []
        while not terminated and not truncated:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=deterministic)
            action_traj.append(action)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            if step_count % 10 == 0:
                print(f"  Step {step_count}: reward = {reward:.3f}")

        if args.render:
            meshcat.PublishRecording()

        print(f"Episode {episode + 1} finished:")
        print(f"  Total reward: {total_reward:.3f}")
        print(f"  Steps: {step_count}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        print(f"Saving action trajectory run {episode + 1}!")
        np.save(f"action_trajectory_{episode + 1}.npy", np.array(action_traj))

        if args.render and episode < args.episodes - 1:
            input("Press Enter for next episode...")

    env.close()
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
