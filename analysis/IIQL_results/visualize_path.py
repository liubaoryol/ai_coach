import os
import numpy as np
from gym_custom.envs.multiple_goals_2d import MultiGoals2D_2
from omegaconf import OmegaConf
from aic_ml.MentalIQL.agent.make_agent import make_miql_agent
import cv2
from itertools import count


def draw_triangle(canvas, pt, dir, color):
  ortho = np.array([dir[1], -dir[0]])
  ortho = ortho / np.linalg.norm(ortho)
  len_dir = np.linalg.norm(dir)
  pt1 = pt + 0.5 * dir
  pt2 = pt + len_dir * 0.3 * ortho
  pt3 = pt - len_dir * 0.3 * ortho
  pts = np.array([pt1, pt2, pt3])
  pts = np.int32(pts)
  canvas = cv2.fillPoly(canvas, [pts], color)
  return canvas


if __name__ == "__main__":
  env = MultiGoals2D_2()

  # load agent
  log_path = ("/home/sangwon/Projects/ai_coach/train_dnn/result/" +
              "MultiGoals2D_2-v0/miql/Ttx001Tpi001valSv0/2023-09-15_16-46-25/")
  model_path = log_path + "model/iq_MultiGoals2D_2-v0_n50_l0_best"
  config_path = log_path + "log/config.yaml"
  config = OmegaConf.load(config_path)
  agent = make_miql_agent(config, env)
  agent.load(model_path)

  # draw on canvas
  canvas_sz = 300
  canvas = np.ones((canvas_sz, canvas_sz, 3), dtype=np.uint8) * 255

  def env_pt_2_cnv_pt(env_pt):
    pt = env_pt - env.observation_space.low
    pt = canvas_sz * pt / (env.observation_space.high -
                           env.observation_space.low)
    return pt.astype(np.int64)

  for idx, goal in enumerate(env.goals):
    goal_pt = env_pt_2_cnv_pt(goal)
    x_p = int(goal_pt[0] - env.img_island.shape[0] / 2)
    y_p = int(goal_pt[1] - env.img_island.shape[1] / 2)
    canvas[y_p:y_p + env.img_island.shape[1],
           x_p:x_p + env.img_island.shape[0]] = env.img_island

  state = env.reset()

  prev_lat, prev_act = agent.PREV_LATENT, agent.PREV_ACTION
  latent = agent.choose_mental_state(state, prev_lat, sample=False)

  for episode_step in count():
    cur_pt = env_pt_2_cnv_pt(state)
    color = (255, 0, 0) if latent == 0 else (0, 255, 0)
    # canvas = cv2.circle(canvas, cur_pt, 3, color, thickness=-1)

    action = agent.choose_policy_action(state, latent, sample=False)

    next_state, reward, done, info = env.step(action)
    # canvas = cv2.arrowedLine(canvas, cur_pt, env_pt_2_cnv_pt(next_state), color,
    #                          1)
    canvas = draw_triangle(canvas, cur_pt,
                           env_pt_2_cnv_pt(next_state) - cur_pt, color)
    next_latent = agent.choose_mental_state(next_state, latent, sample=False)

    if done:
      break
    state = next_state
    prev_lat = latent
    prev_act = action
    latent = next_latent

  cur_dir = os.path.dirname(__file__)
  output_path = os.path.join(cur_dir, "path3.png")
  cv2.imwrite(output_path, canvas)
