import gym
import numpy as np


class MinimalWrapper(gym.Env):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.n_agent = env.metadata['n_agent']
        if self.n_agent == 1:
            self.observation_space = gym.spaces.Box(np.array([0, 0, 0]), np.array([1, 1, 4]),
                                                    shape=(3,),
                                                    dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Box(np.array([0, 0, 0, 0, 0, 0]), np.array([1, 1, 4, 1, 1, 4]),
                                                    shape=(6,),
                                                    dtype=np.float32)
        # gym.spaces.Tuple([gym.spaces.Discrete(act.n) for act in self.env.action_space])
        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(3) for _ in range(self.n_agent)])
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

    # def reward(self, obs):
    #     if obs[0] > 0.4 and obs[1] > 0.9:
    #         return 1000
    #     return obs[0] + obs[1] - 2

    def render(self, mode='human'):
        self.env.render(mode)
