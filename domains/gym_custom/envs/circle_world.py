import gym
from gym import spaces
import numpy as np


class CircleWorld(gym.Env):
  # uncomment below line if you need to render the environment
  # metadata = {'render.modes': ['console']}

  def __init__(self):
    super().__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Box(low=-1,
                                   high=1,
                                   shape=(2, ),
                                   dtype=np.float32)

    self.half_sz = 5
    self.observation_space = spaces.Box(low=np.array([-self.half_sz, 0]),
                                        high=np.array(
                                            [self.half_sz, 2 * self.half_sz]),
                                        shape=(2, ),
                                        dtype=np.float32)

    self.reset()

  def step(self, action):
    info = {}

    scaler = 0.5

    center = np.array([0, self.half_sz])
    dir = self.cur_obstate - center
    len_dir = np.linalg.norm(dir)
    if len_dir != 0:
      dir /= len_dir

    ortho = np.array([-dir[1], dir[0]])

    inner = ortho.dot(action)
    reward = scaler * inner

    self.cur_obstate += scaler * action
    self.cur_obstate[0] = min(self.half_sz,
                              max(-self.half_sz, self.cur_obstate[0]))
    self.cur_obstate[1] = min(2 * self.half_sz, max(0, self.cur_obstate[1]))

    return self.cur_obstate, reward, False, info

  def reset(self):
    self.cur_obstate = np.array([0.0, 0.0])

    return self.cur_obstate
