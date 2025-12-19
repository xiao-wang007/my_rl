import gymnasium as gym
from tianshou.data import Batch, ReplayBuffer
import numpy as np
from typing import Tuple, Any, Dict, List
from franka_reach_drake import PandaReachEnv

from tianshou.env import *

# obs = env.reset()
# done = False
# while not done:
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     print(f"obs: {obs}, reward: {reward}, done: {done}")
# env.close()

observation = Dict[str, np.ndarray]
reward = float
terminated = bool
truncated = bool
info = Dict[str, Any]

''' Wrap drake gym inside tianshou interface '''
class FrankaReach_ts(gym.Env):
    def __init__(self):
            super().__init__()
            self.env = PandaReachEnv(debug=True,
                                     obs_noise=True,
                                     monitoring_camera=True,
                                     add_disturbances=True)
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space

    def reset(self, seed=None, options=None) -> Tuple[observation, info]:
        """Reset environment to initial state."""
        return self.env.reset(seed=seed, return_info=True, options=options)
        
    
    def step(self, action) -> Tuple[observation, reward, terminated, truncated, info]:
        """Execute one step in the environment."""
        return self.env.step(action)
    
    def seed(self, seed: int) -> List[int]:
        """Set random seed."""
        np.random.seed(seed)
        return self.env.seed(seed) 
    
    def render(self, mode='human') -> Any:
        """Render the environment."""
        return self.env.render(mode=mode)
    
    def close(self) -> None:
        """Clean up resources."""
        return self.env.close()
    
    # # Required spaces
    # observation_space: gym.Space
    # action_space: gym.Space

if __name__ == "__main__":
    env = FrankaReach_ts()
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"obs: {obs}, reward: {reward}, done: {done}")
    env.close()