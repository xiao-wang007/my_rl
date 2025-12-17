import gymnasium as gym
from tianshou.data import Batch, ReplayBuffer
import numpy as np
from typing import Tuple, Any, Dict, List
from franka_reach_drake import PandaReachEnv

from tianshou.env import *

env = PandaReachEnv(debug=True,
                    obs_noise=True,
                    monitoring_camera=True,
                    add_disturbances=True)
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
class MyEnv(gym.Env):
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
        pass
    
    def step(self, action) -> Tuple[observation, reward, terminated, truncated, info]:
        """Execute one step in the environment."""
        pass
    
    def seed(self, seed: int) -> List[int]:
        """Set random seed."""
        pass
    
    def render(self, mode='human') -> Any:
        """Render the environment."""
        pass
    
    def close(self) -> None:
        """Clean up resources."""
        pass
    
    # Required spaces
    observation_space: gym.Space
    action_space: gym.Space