import gym
import numpy as np


class MinimalWrapper(gym.Env):
    def __init__(self, env):
        super().__init__()
        self.env = env

        self.observation_space = gym.spaces.Box(np.array([0, 0, 0, 0, 0, 0]), np.array([1, 1, 4, 1, 1, 4]), shape=(6,),
                                                dtype=np.float32)
        # gym.spaces.Tuple([gym.spaces.Discrete(act.n) for act in self.env.action_space])
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata

        try:
            self.info = self.env.info
        except AttributeError:
            pass

    def reset(self):
        obs = self.env.reset()
        new_obs = []
        for ob in obs:
            new_obs.append(np.concatenate([ob['position'], [ob['orientation']]]))
        return np.concatenate(new_obs)

    def step(self, actions):
        obs, rew, done, info = self.env.step(actions)
        new_obs = []
        for ob in obs:
            new_obs.append(np.concatenate([ob['position'], [ob['orientation']]]))
        return np.concatenate(new_obs), np.sum(rew), done, info

    def render(self, mode='human'):
        self.env.render(mode)
