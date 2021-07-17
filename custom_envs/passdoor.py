import gym
import numpy as np
from marlgrid.base import MultiGridEnv, MultiGrid
from marlgrid.objects import *
from marlgrid.envs import register_marl_env


class PassMultiGrid(MultiGridEnv):
    mission = "get to the other side"
    metadata = {}

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = MultiGrid((width, height))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        # self.put_obj(Goal(color="green", reward=1), width - 2, height - 2)

        # Create a vertical splitting wall
        # splitIdx = self._rand_int(2, width - 2)
        splitIdx = width // 2
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        # self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        # doorIdx = self._rand_int(1, width - 2)
        doorIdx = width // 2 - 3
        d1 = Door(color="yellow", state=Door.states.locked)
        self.put_obj(d1, splitIdx, doorIdx)

        doorIdx = width // 2 + 3
        d2 = Door(color="red", state=Door.states.locked)
        self.put_obj(d2, splitIdx, doorIdx)

        # Place a yellow key on the left side
        # self.place_obj(obj=Key("yellow"), top=(0, 0), size=(splitIdx, height))

        b1 = Button("yellow")
        b2 = Button("red")
        self.place_obj(obj=b1, top=(width // 4, height - 5), size=(3, 3))
        self.place_obj(obj=b2, top=(3 * width // 4 - 3, 5), size=(3, 3))

        self.agent_spawn_kwargs = {'top': (0, 0), 'size': (width // 2, height // 2)}

        b1.signal.connect(d2.switch)
        b2.signal.connect(d1.switch)

        s1 = Switch("yellow")
        self.place_obj(obj=s1, top=(width // 4, height - 5), size=(3, 3))
        s2 = Switch("red")
        self.place_obj(obj=s2, top=(3 * width // 4 - 3, 5), size=(3, 3))
        s1.signal.connect(d2.switch)
        s2.signal.connect(d1.switch)

    def reward(self):
        count = 0
        for i, agent in enumerate(self.agents):
            if agent.pos[0] > self.width // 2:
                count += 1

        return np.array([(count >= 2) * 1000, (count >= 2) * 1000])

    def done(self):
        count = 0
        for i, agent in enumerate(self.agents):
            if agent.pos[0] > self.width // 2:
                count += 1

        if count >= 2 or (self.step_count >= self.max_steps):
            self.reset()
            return 1
        return 0


class MinimalWrapper(gym.Env):
    def __init__(self, env):
        super().__init__()
        self.env = env

        self.observation_space = gym.spaces.Box(np.array([0, 0, 0, 0, 0, 0]), np.array([1, 1, 4, 1, 1, 4]), shape=(6,), dtype=np.float32)
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


class PassMultiGridWrapper(PassMultiGrid):
    def reset(self, **kwargs):
        obs = super(PassMultiGridWrapper, self).reset(**kwargs)
        return np.concatenate(obs)

    def step(self, actions):
        actions = np.split(actions, 2)
        obs, rew, done, info = super(PassMultiGridWrapper, self).step(actions)
        return np.concatenate(obs), np.sum(rew), np.all(done), np.concatenate(info)


n_agent = 2
register_marl_env(
    "MarlGrid-PassDoor30x30-v1",
    PassMultiGrid,
    n_agents=n_agent,
    grid_size=30,
    view_size=20,
    env_kwargs={},
    observe_position=True,
    observe_orientation=True,
    observation_style='rich'
)
