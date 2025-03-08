import gym
import numpy as np

class PartialObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, mask):
        super().__init__(env)
        self.mask = mask  # A binary mask with the same shape as the observation

    def observation(self, obs):
        return obs * self.mask

