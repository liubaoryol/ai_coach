import os
import numpy as np
from omegaconf import OmegaConf
import cv2
from itertools import count
import gym_custom
from aic_ml.baselines.IQLearn.utils.utils import make_env
from infer_latent import load_model
import torch


def draw_triangle(canvas, pt, dir, color):
  if np.linalg.norm(dir) < 0.01:
    return canvas

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


def save_path(env_name,
              alg,
              modelpath,
              logroot,
              output_path,
              fixed_latent=None,
              n_epi=1):
  resdir = f"/home/sangwon/Projects/ai_coach/train_dnn/{logroot}/{env_name}/{alg}/"
  modelpath = resdir + modelpath

  logdir = os.path.dirname(os.path.dirname(modelpath))

  config_path = os.path.join(logdir, "log/config.yaml")
  config = OmegaConf.load(config_path)

  # add updated keys
  if alg == "miql":
    if 'miql_tx_method_div' not in config.keys():
      config['miql_tx_method_div'] = ""
      print("Missing key - miql_tx_method_div is added as \"\".")
    if 'miql_pi_method_div' not in config.keys():
      config['miql_pi_method_div'] = ""
      print("Missing key - miql_pi_method_div is added as \"\".")
  elif alg == "oiql":
    if 'method_div' not in config.keys():
      config['method_div'] = ""
      print("Missing key - method_div is added as \"\".")

  env = make_env(env_name, env_make_kwargs={})

  # config['device'] = 'cpu'
  # load model
  agent = load_model(config, env, modelpath)

  # draw on canvas
  canvas_sz = 300
  canvas = np.ones((canvas_sz, canvas_sz, 3), dtype=np.uint8) * 255
  canvas = env.draw_background(canvas)

  for i_e in range(n_epi):
    state = env.reset()

    prev_lat, prev_act = agent.PREV_LATENT, agent.PREV_ACTION
    if fixed_latent is None:
      latent = agent.choose_mental_state(state, prev_lat, sample=False)
    else:
      latent = fixed_latent

    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]

    for episode_step in count():
      cur_pt = env.env_pt_2_scr_pt(state)
      color = colors[latent]

      action = agent.choose_policy_action(state, latent, sample=False)

      state, reward, done, info = env.step(action)

      canvas = draw_triangle(canvas, cur_pt,
                             env.env_pt_2_scr_pt(state) - cur_pt, color)
      if fixed_latent is None:
        latent = agent.choose_mental_state(state, latent, sample=False)

      if done:
        break
      prev_lat = latent
      prev_act = action

  cv2.imwrite(output_path, canvas)


if __name__ == "__main__":

  ntnt = 2
  if False:
    cur_dir = os.path.dirname(__file__)
    model_path = ("Ttx001Tpi001tol5Sv2/2023-09-20_16-43-42/" +
                  "model/iq_MultiGoals2D_3-v0_n50_l10_best")
    output_path = os.path.join(cur_dir, f"iiql_path{ntnt+1}.png")
    save_path("MultiGoals2D_3-v0", "miql", model_path, "result", output_path,
              ntnt, 10)

  if False:
    cur_dir = os.path.dirname(__file__)
    model_path = ("T001tol5Sv2/2023-09-21_15-45-55/" +
                  "model/iq_MultiGoals2D_3-v0_n50_l10_best")
    output_path = os.path.join(cur_dir, f"oiql_path{ntnt+1}.png")
    save_path("MultiGoals2D_3-v0", "oiql", model_path, "result_lambda",
              output_path, ntnt, 10)

  if True:
    cur_dir = os.path.dirname(__file__)
    model_path = ("tol5Sv2/2023-09-21_00-20-45/" +
                  "model/MultiGoals2D_3-v0_n50_l10_best.torch")
    output_path = os.path.join(cur_dir, f"ogail_path{ntnt+1}.png")
    save_path("MultiGoals2D_3-v0", "ogail", model_path, "result", output_path,
              ntnt, 10)
