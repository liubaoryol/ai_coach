from typing import Sequence
import os
import glob
import click
from tqdm import tqdm
import random

from aic_ml.baselines.ikostrikov_gail import bc_dnn
import numpy as np
from datetime import datetime


@click.command()
@click.option("--domain", type=str, default="movers", help="")
@click.option("--num-abstates", type=int, default=30, help="")
def main(domain, num_abstates):

  # define the domain where trajectories were generated
  ##################################################
  if domain == "movers":
    from aic_domain.box_push.utils import BoxPushTrajectories
    from aic_domain.box_push_v2.agent import BoxPushAIAgent_Team
    from aic_domain.box_push_v2.maps import MAP_MOVERS
    from aic_domain.box_push_v3.simulator import BoxPushSimulatorV3
    from aic_domain.box_push_v3.policy import Policy_MoversV3
    from aic_domain.box_push_v3.mdp import (MDP_MoversV3_Agent,
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
  traj_labeled_ver = train_data.get_as_row_lists(no_latent_label=False,
                                                 include_terminal=True)
  num_traj = len(traj_labeled_ver)

  # load models
  ##################################################
  list_np_pi = []

  model_dir = os.path.join(DATA_DIR, "learned_models/")

  num_train = 500
  num_agents = len(AGENTS)

  filename_pre = (f"{domain}_bayes_abs_{num_train}_{num_abstates}")
  np_abs = np.load(model_dir + filename_pre + "_abs.npy")

  for idx_a in range(num_agents):
    file_name = filename_pre + f"_pi_a{idx_a+1}.npy"
    list_np_pi.append(np.load(model_dir + file_name))

  # convert traj
  ##################################################
  num_agents = len(AGENTS)
  list_traj_za_each = [[] for _ in range(num_agents)]
  for traj in traj_labeled_ver:
    list_za = [[] for _ in range(num_agents)]
    for s, a, _ in traj:
      p = np_abs[s]

      if a is not None:
        for idx_a in range(num_agents):
          p = p * list_np_pi[idx_a][:, a[idx_a]]

      idx_z = np.argmax(p)
      for idx_a in range(num_agents):
        a_each = a[idx_a] if a is not None else None
        list_za[idx_a].append((idx_z, a_each))

    for idx_a in range(num_agents):
      list_traj_za_each[idx_a].append(list_za[idx_a])

  # prediction
  ##################################################

  LOG_DIR = os.path.join(os.path.dirname(__file__), "logs/")
  if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
  logpath = LOG_DIR + str(datetime.today())

  temp_dir = DATA_DIR + "learned_models/"
  if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
  save_prefix += ("%d_%d" % (num_traj, num_abstates))

  BC = True
  num_iterations = 300
  save_prefix = SAVE_PREFIX
  policy = None
  if BC:
    list_policy = []
    save_prefix += "_bc_bayes_abs_"
    for idx_a in range(num_agents):
      policy = bc_dnn(num_abstates, MDP_AGENT.num_actions,
                      list_traj_za_each[idx_a], logpath + "/bc_bayes_abs", 64,
                      num_iterations)
      list_policy.append(policy)

    # save models
    save_prefix = os.path.join(temp_dir, save_prefix)
    for idx_a, policy in enumerate(list_policy):
      if policy is not None:
        np.save(save_prefix + f"_pi_a{idx_a + 1}", policy)


if __name__ == "__main__":
  main()
