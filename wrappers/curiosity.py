import gym
import numpy as np
from threading import get_ident
import os

class CuriosityWrapper(gym.Env):
    def __init__(self, env, counter=None):
        super().__init__()
        self.env = env

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata
        if counter is None:
            self.counter = np.zeros([30, 30, 4, 30, 30, 4], dtype=np.int32)
        else:
            self.counter = counter
        try:
            self.info = self.env.info
        except AttributeError:
            pass

    def _get_index(self, obs):
        s = [o for o in obs]
        s[0] *= self.metadata['width']
        s[1] *= self.metadata['height']
        s[3] *= self.metadata['width']
        s[4] *= self.metadata['height']
        s = [int(s) for s in s]
        return tuple(s)

    def get_curiosity(self, obs):
        return 1. / np.sqrt(self.counter[self._get_index(obs)] + 1)

    def reset(self):
        obs = self.env.reset()
        self.counter[self._get_index(obs)] += 1
        return obs

    def step(self, actions):
        obs, rew, done, info = self.env.step(actions)
        self.counter[self._get_index(obs)] += 1
        print(os.getpid(), np.sum(self.counter))
        c_rew = self.get_curiosity(obs)
        info['cnt'] = self.counter
        return obs, rew + c_rew, done, info

    def render(self, mode='human'):
        self.env.render(mode)
