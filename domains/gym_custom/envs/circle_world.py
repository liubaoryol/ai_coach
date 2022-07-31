import gym
from gym import spaces
import numpy as np


class CircleWorld(gym.Env):
  # uncomment below line if you need to render the environment
  # metadata = {'render.modes': ['console']}

  def __init__(self, ccw=True):
    super().__init__()
    self.ccw = ccw

    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Box(low=-1,
                                   high=1,
                                   shape=(2, ),
                                   dtype=np.float32)

    self.half_sz = 5
    self.observation_space = spaces.Box(low=[-self.half_sz, 0],
                                        high=[self.half_sz, 2 * self.half_sz],
                                        shape=(2, ),
                                        dtype=np.float32)

    self.reset()

  def step(self, action):
    info = {}

    scaler = 0.5

    center = np.array([0, self.half_sz])
    dir = self.cur_obstate - center
    ortho = np.array([-dir[1], dir[0]])
    len_ortho = np.linalg.norm(ortho)
    if len_ortho != 0:
      ortho /= len_ortho

    inner = ortho.dot(action)
    reward = scaler * inner
    if not self.ccw:
      reward = -reward

    self.cur_obstate += scaler * action
    self.cur_obstate[0] = min(self.half_sz,
                              max(-self.half_sz, self.cur_obstate[0]))
    self.cur_obstate[1] = min(2 * self.half_sz, max(0, self.cur_obstate[1]))

    return self.cur_obstate, reward, False, info

  def reset(self):
    self.cur_obstate = np.array([0.0, 0.0])

    return self.cur_obstate
