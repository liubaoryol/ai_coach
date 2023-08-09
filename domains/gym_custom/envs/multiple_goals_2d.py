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

    self.goals = np.array([(-4, 4), (4, 4), (0, -4)])
    self.visited = np.zeros(len(self.goals))
    self.tolerance = 0.3

    self.reset()

  def reset(self):
    self.cur_obstate = self.observation_space.sample()
    self.visited = np.zeros(len(self.goals))
    # self.cur_goal_idx = np.random.choice(len(self.goals))

    return self.cur_obstate

  def step(self, action):
    info = {}

    next_obstate = self.cur_obstate + action

    next_obstate[0] = min(self.observation_space.high[0],
                          max(self.observation_space.low[0], next_obstate[0]))
    next_obstate[1] = min(self.observation_space.high[1],
                          max(self.observation_space.low[1], next_obstate[1]))
    self.cur_obstate = next_obstate

    PANELTY = -0.1
    GOAL_POINT = 10

    reward = PANELTY
    for idx, goal in enumerate(self.goals):
      if self.visited[idx] != 0:
        continue

      if np.linalg.norm(goal - self.cur_obstate) < self.tolerance:
        reward += GOAL_POINT
        self.visited[idx] = 1

    done = np.sum(self.visited) == len(self.visited)

    return self.cur_obstate, reward, done, info

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
      color = (0, 0, 255)
      # if idx == self.cur_goal_idx:
      #   color = (0, 255, 0)
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

  def get_action(goal, state):
    RANDOM = False
    if RANDOM:
      vx = np.random.rand() * 2 - 1
      vy = np.random.rand() * 2 - 1
      return np.array([vx, vy])
    else:
      vec_dir = goal - state
      len_vec = np.linalg.norm(vec_dir)
      if len_vec != 0:
        vec_dir /= len_vec
      return 0.3 * vec_dir

  def get_new_goal_idx(visited, goals, cur_goal_idx, state, tolerance):
    if np.sum(visited) == len(visited):
      return None

    if cur_goal_idx is not None:
      if np.linalg.norm(goals[cur_goal_idx] - state) > tolerance:
        return cur_goal_idx

    return np.random.choice(np.where(env.visited == 0)[0])

  for _ in range(10):
    state = env.reset()
    goal_idx = None
    episode_reward = 0
    for cnt in range(200):
      goal_idx = get_new_goal_idx(env.visited, env.goals, goal_idx, state,
                                  env.tolerance)
      action = get_action(env.goals[goal_idx], state)
      next_state, reward, done, info = env.step(action)
      episode_reward += reward
      env.render()
      if done:
        break
      state = next_state
    print(episode_reward, cnt)
