import gym
import numpy as np


class SingleAgentWrapper(gym.Env):
    def __init__(self, env):
        super().__init__()
        self.env = env

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata

        try:
            self.info = self.env.info
        except AttributeError:
            pass

    def reset(self):
        obs = self.env.reset()
        return np.concatenate(obs)

    def step(self, actions):
        actions = np.split(actions, 2)
        obs, rew, done, info = self.env.step(actions)
        return np.concatenate(obs), np.sum(rew), np.all(done), np.concatenate(info)
