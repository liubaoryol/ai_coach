from typing import Sequence
import os
import glob
import click
from tqdm import tqdm
import random
from ai_coach_core.latent_inference.decoding import smooth_inference_sa

# from aicoach_baselines.sb3_algorithms import behavior_cloning_sb3
from aicoach_baselines.ikostrikov_gail import bc_dnn
import numpy as np
from datetime import datetime
# rl algorithm


def conv_2_joint_index(tup_num_latent, tup_latent_idx):
  return np.ravel_multi_index(tup_latent_idx, tup_num_latent)


def conv_2_indv_index(tup_num_latent, idx):
  return np.unravel_index(idx, tup_num_latent)


@click.command()
@click.option("--domain", type=str, default="movers", help="")
def main(domain):

  # define the domain where trajectories were generated
  ##################################################
  if domain == "movers":
    from ai_coach_domain.box_push.utils import BoxPushTrajectories
    from ai_coach_domain.box_push_v2.agent import BoxPushAIAgent_Team
    from ai_coach_domain.box_push_v2.maps import MAP_MOVERS
    from ai_coach_domain.box_push_v3.simulator import BoxPushSimulatorV3
    from ai_coach_domain.box_push_v3.policy import Policy_MoversV3
    from ai_coach_domain.box_push_v3.mdp import (MDP_MoversV3_Agent,
                                                 MDP_MoversV3_Task)
    sim = BoxPushSimulatorV3(False)
    TEMPERATURE = 0.3
    GAME_MAP = MAP_MOVERS
    SAVE_PREFIX = GAME_MAP["name"]
    MDP_TASK = MDP_MoversV3_Task(**GAME_MAP)
    MDP_AGENT = MDP_MoversV3_Agent(**GAME_MAP)
    POLICY_1 = Policy_MoversV3(MDP_TASK, MDP_AGENT, TEMPERATURE, 0)
    POLICY_2 = Policy_MoversV3(MDP_TASK, MDP_AGENT, TEMPERATURE, 1)
    # init_states = ([0] * len(GAME_MAP["boxes"]), GAME_MAP["a1_init"],
    #                GAME_MAP["a2_init"])
    AGENT_1 = BoxPushAIAgent_Team(POLICY_1, agent_idx=sim.AGENT1)
    AGENT_2 = BoxPushAIAgent_Team(POLICY_2, agent_idx=sim.AGENT2)
    AGENTS = [AGENT_1, AGENT_2]
    train_data = BoxPushTrajectories(MDP_TASK, MDP_AGENT)

  # load files
  ##################################################
  DATA_DIR = os.path.join(os.path.dirname(__file__), "data/")
  TRAIN_DIR = os.path.join(DATA_DIR, SAVE_PREFIX + '_train_opt')

  file_names = glob.glob(os.path.join(TRAIN_DIR, '*.txt'))
  random.shuffle(file_names)

  train_data.load_from_files(file_names)
  # traj_labeled_ver = train_data.get_as_row_lists(no_latent_label=False,
  #                                                include_terminal=True)
  # traj_unlabel_ver = train_data.get_as_row_lists(no_latent_label=True,
  #                                                include_terminal=True)
  list_trajs = train_data.get_as_column_lists(include_terminal=True)
  num_traj = len(list_trajs)

  # load models
  ##################################################
  list_np_pi = []
  list_np_tx = []
  list_np_bx = []

  if domain == "rescue_3":
    tx_dependency = "FTTTT"
  else:
    tx_dependency = "FTTT"
  model_dir = os.path.join(DATA_DIR, "learned_models/")

  num_train = 500
  num_x = 4

  num_agents = len(AGENTS)
  model_dir = os.path.join(DATA_DIR, "learned_models/")
  filename_pre = (f"{domain}_btil_hdp_{tx_dependency}_{num_train}_{num_x}")

  for idx_a in range(num_agents):
    file_name = filename_pre + f"_pi_a{idx_a+1}.npy"
    list_np_pi.append(np.load(model_dir + file_name))

    file_name = filename_pre + f"_tx_a{idx_a+1}.npy"
    list_np_tx.append(np.load(model_dir + file_name))

    file_name = filename_pre + f"_bx_a{idx_a+1}.npy"
    list_np_bx.append(np.load(model_dir + file_name))

  # prediction
  ##################################################
  tup_num_latent = (num_x, ) * num_agents

  use_confidence = True
  list_traj_sx = []
  if not use_confidence:
    for i, traj in tqdm(enumerate(list_trajs)):
      list_states, list_actions, list_latents = traj
      list_np_px = smooth_inference_sa(list_states, list_actions, num_agents,
                                       tup_num_latent, list_np_pi, list_np_tx,
                                       list_np_bx)
      list_sx = []
      for t in range(len(list_actions)):
        tmp_x = []
        for np_px in list_np_px:
          list_same_idx = np.argwhere(np_px[t] == np.max(np_px[t]))
          xhat = random.choice(list_same_idx)[0]
          tmp_x.append(xhat)

        xidx = conv_2_joint_index(tup_num_latent, tuple(tmp_x))
        list_sx.append((list_states[t], xidx))

      list_traj_sx.append(list_sx)
  else:
    for i, traj in tqdm(enumerate(list_trajs)):
      list_states, list_actions, list_latents = traj
      list_np_px = smooth_inference_sa(list_states, list_actions, num_agents,
                                       tup_num_latent, list_np_pi, list_np_tx,
                                       list_np_bx)
      list_sx = []
      for t in range(len(list_actions)):
        tmp_x = []
        np_px_all = np.ones(tup_num_latent)
        for idx_a in range(num_agents):
          index = [1] * num_agents
          index[idx_a] = -1
          index = tuple(index)
          np_px_all = np_px_all * list_np_px[idx_a][t].reshape(index)
        np_px_all_flat = np_px_all.reshape(-1)
        ind = np.argpartition(np_px_all_flat, -3)[-3:]
        np_conf = np_px_all_flat[ind]
        for idx, xidx in enumerate(ind):
          list_sx.append((list_states[t], xidx, np_conf[idx]))

      list_traj_sx.append(list_sx)

  LOG_DIR = os.path.join(os.path.dirname(__file__), "logs/")
  if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
  logpath = LOG_DIR + str(datetime.today())

  BC = True
  num_iterations = 1000
  save_prefix = SAVE_PREFIX
  policy = None
  if BC:
    save_prefix += "_bc_hdp_"
    policy = bc_dnn(MDP_TASK.num_states, np.prod(tup_num_latent), list_traj_sx,
                    logpath + "/bc_x", 64, num_iterations, use_confidence)
    policy = policy.reshape(MDP_TASK.num_states, *tup_num_latent)

  temp_dir = DATA_DIR + "learned_models/"
  if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
  save_prefix += ("%d" % (num_traj, ))

  # save models
  save_prefix = os.path.join(temp_dir, save_prefix)
  if policy is not None:
    np.save(save_prefix + "_pi_x", policy)


if __name__ == "__main__":
  main()
