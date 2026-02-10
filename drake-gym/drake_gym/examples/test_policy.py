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
from pydrake.all import StartMeshcat, Rgba, RigidTransform, RotationMatrix
from pydrake.geometry import Sphere
from pydrake.common.eigen_geometry import Quaternion

# Register the environment
gym.envs.register(
    id="panda-reach-v0",
    entry_point="envs.franka_reach.franka_reach_drake:PandaReachEnv",
)


def draw_triad(meshcat, name, pose, length=0.1, radius=0.005):
    """Draw a coordinate frame triad (RGB = XYZ axes) at the given pose."""
    from pydrake.geometry import Cylinder
    from pydrake.math import RollPitchYaw
    
    # X-axis (red) - cylinder along X, rotated from Z
    meshcat.SetObject(f"{name}/x_axis", Cylinder(radius, length), Rgba(1, 0, 0, 1))
    X_x = RigidTransform(
        RotationMatrix(RollPitchYaw(0, np.pi/2, 0)),  # Rotate Z to X
        np.array([length/2, 0, 0])  # Offset to center
    )
    meshcat.SetTransform(f"{name}/x_axis", pose @ X_x)
    
    # Y-axis (green) - cylinder along Y
    meshcat.SetObject(f"{name}/y_axis", Cylinder(radius, length), Rgba(0, 1, 0, 1))
    X_y = RigidTransform(
        RotationMatrix(RollPitchYaw(-np.pi/2, 0, 0)),  # Rotate Z to Y
        np.array([0, length/2, 0])
    )
    meshcat.SetTransform(f"{name}/y_axis", pose @ X_y)
    
    # Z-axis (blue) - cylinder along Z (default)
    meshcat.SetObject(f"{name}/z_axis", Cylinder(radius, length), Rgba(0, 0, 1, 1))
    X_z = RigidTransform(np.array([0, 0, length/2]))
    meshcat.SetTransform(f"{name}/z_axis", pose @ X_z)


def visualize_target(meshcat, goal_pos, goal_r1r2=None, radius=0.02):
    """Visualize target position as a red sphere and orientation as a triad."""
    meshcat.SetObject(
        "target",
        Sphere(radius),
        Rgba(0.0, 1.0, 0.0, 0.5)  # green, semi-transparent
    )
    meshcat.SetTransform(
        "target",
        RigidTransform(goal_pos)
    )

    # Add goal frame triad
    if goal_r1r2 is not None:
        rx = goal_r1r2[:3]
        ry = goal_r1r2[3:6]
        rz = np.cross(rx, ry)
        R = RotationMatrix(np.column_stack([rx, ry, rz]))
        goal_pose = RigidTransform(R, goal_pos)
        draw_triad(meshcat, "goal_frame", goal_pose, length=0.15, radius=0.005)


def update_ee_frame(meshcat, env):
    """Update end-effector frame triad visualization.
    #NOTE: this is for live viewing of meshcat visualization. Would not
    # work in the playback with meshcat.StartRecording() """
    diagram = env.unwrapped.simulator.get_system()
    context = env.unwrapped.simulator.get_context()
    
    # Get EE pose from the exported port (pos=3 + r1=3 + r2=3)
    ee_pose_vec = diagram.GetOutputPort("observations_ee_pose").Eval(context)
    ee_pos = ee_pose_vec[:3]
    rx = ee_pose_vec[3:6]
    ry = ee_pose_vec[6:9]
    rz = np.cross(rx, ry)
    
    # Build rotation matrix from r1, r2, r3
    R = RotationMatrix(np.column_stack([rx, ry, rz]))
    ee_pose = RigidTransform(R, ee_pos)
    
    draw_triad(meshcat, "ee_frame", ee_pose, length=0.1, radius=0.004)


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
    parser.add_argument(
        "--output_dir",
        type=str,
        default="envs/franka_reach/diagnostic",
        help="where to save the action traj from the test policy",
    )
    parser.add_argument(
        "--time_limit",
        type=float,
        default=5.0,
        help="Time limit for each episode",
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
            time_limit=args.time_limit,
            debug=True,
        )
        input("Open Meshcat URL in browser, then press Enter to start...")
    else:
        env = gym.make("panda-reach-v0", time_limit=args.time_limit)

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
            goal_r1r2 = env.unwrapped.goal_state.goal_r1r2
            visualize_target(meshcat, goal_pos, goal_r1r2)
            print(f"Target position: {goal_pos}")

        print(f"\n=== Episode {episode + 1}/{args.episodes} ===")

        if args.render:
            meshcat.StartRecording()

        action_traj = []
        while not terminated and not truncated:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=deterministic)
            # action = 0.1 * action  # scale down actions for testing
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

        diagram = env.unwrapped.simulator.get_system()
        context = env.unwrapped.simulator.get_context()
        reward_breakdown = diagram.GetOutputPort("reward_breakdown").Eval(context)
        print(f"  Reward breakdown: {reward_breakdown}")

        print(f"  Steps: {step_count}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        print(f"Saving action trajectory run {episode + 1}!")
        np.save(f"{args.output_dir}/action_trajectory_{episode + 1}.npy", np.array(action_traj))

        if args.render and episode < args.episodes - 1:
            input("Press Enter for next episode...")

    env.close()
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
