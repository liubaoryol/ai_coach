import gym
from gym import spaces
import cv2
import numpy as np


class MultiGoals2D(gym.Env):
  # uncomment below line if you need to render the environment
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super().__init__()

    self.action_space = spaces.Box(low=-1,
                                   high=1,
                                   shape=(2, ),
                                   dtype=np.float32)

    self.half_sz = 5

    self.observation_space = spaces.Box(
        low=np.array([-self.half_sz, -self.half_sz]),
        high=np.array([self.half_sz, self.half_sz]),
        shape=(2, ),
        dtype=np.float32)

    self.goals = np.array([(-4, 4), (4, 4), (-4, -4), (4, -4)])
    self.tolerance = 0.3

    self.reset()

  def reset(self):
    self.cur_obstate = self.observation_space.sample()
    self.cur_goal_idx = np.random.choice(len(self.goals))

    return self.cur_obstate

  def step(self, action):
    info = {}

    next_obstate = self.cur_obstate + action

    next_obstate[0] = min(self.observation_space.high[0],
                          max(self.observation_space.low[0], next_obstate[0]))
    next_obstate[1] = min(self.observation_space.high[1],
                          max(self.observation_space.low[1], next_obstate[1]))
    self.cur_obstate = next_obstate

    reward = float(
        np.linalg.norm(self.goals[self.cur_goal_idx] -
                       self.cur_obstate) < self.tolerance)

    return self.cur_obstate, reward, False, info

  def get_canvas(self):
    canvas_sz = 300

    canvas = np.ones((canvas_sz, canvas_sz, 3), dtype=np.uint8) * 255

    def env_pt_2_scr_pt(env_pt):
      pt = env_pt - self.observation_space.low
      pt = canvas_sz * pt / (self.observation_space.high -
                             self.observation_space.low)
      return pt.astype(np.int64)

    cur_pt = env_pt_2_scr_pt(self.cur_obstate)

    for idx, goal in enumerate(self.goals):
      if idx == self.cur_goal_idx:
        color = (0, 255, 0)
      else:
        color = (0, 0, 255)
      goal_pt = env_pt_2_scr_pt(goal)
      radius = int(
          canvas_sz * self.tolerance /
          (self.observation_space.high - self.observation_space.low)[0])
      canvas = cv2.circle(canvas, goal_pt, radius, color, thickness=-1)

    color = (255, 0, 0)
    canvas = cv2.circle(canvas, cur_pt, 5, color, thickness=-1)

    return canvas

  def render(self, mode='human'):
    if mode == 'human':
      cv2.imshow("MultiGoals on Plane", self.get_canvas())
      cv2.waitKey(10)

  def close(self):
    cv2.destroyAllWindows()


if __name__ == "__main__":
  env = MultiGoals2D()

  for _ in range(10):
    state = env.reset()
    vec_dir = env.goals[env.cur_goal_idx] - state
    len_vec = np.linalg.norm(vec_dir)
    if len_vec != 0:
      vec_dir /= len_vec
    for _ in range(50):
      env.step(0.1 * vec_dir)
      env.render()
