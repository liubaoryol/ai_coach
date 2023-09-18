import os
import pickle
from collections import defaultdict
import gym
from gym import spaces
import cv2
import numpy as np
from PIL import Image


def read_transparent_png(filename):
  # ref: https://stackoverflow.com/a/41896175
  image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
  alpha_channel = image_4channel[:, :, 3]
  rgb_channels = image_4channel[:, :, :3]

  # White Background Image
  white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

  # Alpha factor
  alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
  alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor),
                                axis=2)

  # Transparent Image Rendered on White Background
  base = rgb_channels.astype(np.float32) * alpha_factor
  white = white_background_image.astype(np.float32) * (1 - alpha_factor)
  final_image = base + white
  return final_image.astype(np.uint8)


class MultiGoals2D(gym.Env):
  # uncomment below line if you need to render the environment
  metadata = {'render.modes': ['human']}

  def __init__(self, possible_goals):
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

    curdir = os.path.dirname(__file__)
    self.goals = possible_goals
    self.visited = np.zeros(len(self.goals))
    self.tolerance = 0.3
    img_pirate = read_transparent_png(os.path.join(curdir, 'images/pirate.png'))
    img_island = read_transparent_png(os.path.join(curdir, 'images/island.png'))

    self.img_pirate = cv2.resize(img_pirate, (30, 30))
    self.img_island = cv2.resize(img_island, (30, 30))
    self.delay = 10

    self.reset()

  def reset(self):
    self.cur_obstate = self.observation_space.sample()
    self.visited = np.zeros(len(self.goals))

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

    bg_unit = canvas_sz // 6
    for idx in range(6):
      if idx % 2 == 0:
        canvas[bg_unit * idx:bg_unit * (idx + 1), :, 0] = 255
        canvas[bg_unit * idx:bg_unit * (idx + 1), :, 1] = 197
        canvas[bg_unit * idx:bg_unit * (idx + 1), :, 2] = 120
      else:
        canvas[bg_unit * idx:bg_unit * (idx + 1), :, 0] = 255
        canvas[bg_unit * idx:bg_unit * (idx + 1), :, 1] = 150
        canvas[bg_unit * idx:bg_unit * (idx + 1), :, 2] = 14

    def env_pt_2_scr_pt(env_pt):
      pt = env_pt - self.observation_space.low
      pt = canvas_sz * pt / (self.observation_space.high -
                             self.observation_space.low)
      return pt.astype(np.int64)

    cur_pt = env_pt_2_scr_pt(self.cur_obstate)

    for idx, goal in enumerate(self.goals):
      color = (0, 0, 255)

      goal_pt = env_pt_2_scr_pt(goal)
      # radius = int(
      #     canvas_sz * self.tolerance /
      #     (self.observation_space.high - self.observation_space.low)[0])
      # canvas = cv2.circle(canvas, goal_pt, radius, color, thickness=-1)
      x_p = int(goal_pt[0] - self.img_island.shape[0] / 2)
      y_p = int(goal_pt[1] - self.img_island.shape[1] / 2)
      canvas[y_p:y_p + self.img_island.shape[1],
             x_p:x_p + self.img_island.shape[0]] = self.img_island

    # color = (255, 0, 0)
    # canvas = cv2.circle(canvas, cur_pt, 5, color, thickness=-1)
    x_p = int(cur_pt[0] - self.img_pirate.shape[0] / 2)
    y_p = int(cur_pt[1] - self.img_pirate.shape[1] / 2)
    canvas[y_p:y_p + self.img_pirate.shape[1],
           x_p:x_p + self.img_pirate.shape[0]] = self.img_pirate

    return canvas

  def render(self, mode='human'):
    if mode == 'human':
      cv2.imshow("MultiGoals on Plane", self.get_canvas())
      cv2.waitKey(self.delay)

  def set_render_delay(self, delay):
    self.delay = delay

  def close(self):
    cv2.destroyAllWindows()


class MultiGoals2D_1(MultiGoals2D):

  def __init__(self):
    super().__init__([(-4, 4)])


class MultiGoals2D_2(MultiGoals2D):

  def __init__(self):
    super().__init__([(-4, 4), (4, 4)])


class MultiGoals2D_3(MultiGoals2D):

  def __init__(self):
    super().__init__([(-4, 4), (4, 4), (0, -4)])


class MultiGoals2D_4(MultiGoals2D):

  def __init__(self):
    super().__init__([(-4, 4), (4, 4), (4, -4), (-4, -4)])


class MultiGoals2D_5(MultiGoals2D):

  def __init__(self):
    super().__init__([(-2.5, 4), (2.5, 4), (0, -4), (-4, -0.5), (4, -0.5)])


# synthetic agent
def get_action(goal, state):
  nx = 0.1 * (np.random.rand() * 2 - 1)
  ny = 0.1 * (np.random.rand() * 2 - 1)
  noise = np.array([nx, ny])
  RANDOM = False
  if RANDOM:
    vx = np.random.rand() * 2 - 1
    vy = np.random.rand() * 2 - 1
    return np.array([vx, vy]) + noise
  else:
    vec_dir = goal - state
    len_vec = np.linalg.norm(vec_dir)
    if len_vec != 0:
      vec_dir /= len_vec
    return 0.3 * vec_dir + noise


def get_new_goal_idx(visited, goals, cur_goal_idx, state, tolerance):
  if np.sum(visited) == len(visited):
    return None

  if cur_goal_idx is not None:
    if np.linalg.norm(goals[cur_goal_idx] - state) > tolerance:
      return cur_goal_idx

  return np.random.choice(np.where(visited == 0)[0])


def generate_data(save_dir, env_name, n_traj, render=False, render_delay=10):
  expert_trajs = defaultdict(list)
  if env_name == "MultiGoals2D_1-v0":
    env = MultiGoals2D_1()
  elif env_name == "MultiGoals2D_2-v0":
    env = MultiGoals2D_2()
  elif env_name == "MultiGoals2D_3-v0":
    env = MultiGoals2D_3()
  elif env_name == "MultiGoals2D_4-v0":
    env = MultiGoals2D_4()
  elif env_name == "MultiGoals2D_5-v0":
    env = MultiGoals2D_5()
  else:
    raise NotImplementedError

  env.set_render_delay(render_delay)

  for _ in range(n_traj):
    state = env.reset()
    goal_idx = None
    episode_reward = 0

    samples = []
    for cnt in range(200):
      goal_idx = get_new_goal_idx(env.visited, env.goals, goal_idx, state,
                                  env.tolerance)
      action = get_action(env.goals[goal_idx], state)
      next_state, reward, done, info = env.step(action)

      samples.append((state, action, next_state, goal_idx, reward, done))

      episode_reward += reward
      if render:
        env.render()
      if done:
        break
      state = next_state

    if render:
      print(episode_reward, cnt)
    states, actions, next_states, latents, rewards, dones = list(zip(*samples))

    expert_trajs["states"].append(states)
    expert_trajs["next_states"].append(next_states)
    expert_trajs["actions"].append(actions)
    expert_trajs["latents"].append(latents)
    expert_trajs["rewards"].append(rewards)
    expert_trajs["dones"].append(dones)
    expert_trajs["lengths"].append(len(states))

  if save_dir is not None:
    file_path = os.path.join(save_dir, f"{env_name}_{n_traj}.pkl")
    with open(file_path, 'wb') as f:
      pickle.dump(expert_trajs, f)

  return expert_trajs


if __name__ == "__main__":
  cur_dir = os.path.dirname(__file__)

  # for idx in range(1, 6):
  # traj = generate_data(cur_dir, f"MultiGoals2D_{idx}-v0", 500, False)
  traj = generate_data(None, "MultiGoals2D_3-v0", 10, True, 50)
