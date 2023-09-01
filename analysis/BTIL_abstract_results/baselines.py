import os
import glob
import logging
import random
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aic_ml.baselines.sb3_algorithms import gail_w_ppo
from aic_ml.baselines.ikostrikov_gail import bc_dnn
from aic_ml.IQLearn.iql import run_iql
from datetime import datetime


@click.command()
@click.option("--domain", type=str, default="movers", help="")
@click.option("--opt", type=bool, default=True, help="")
def main(domain, opt):

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
  elif domain == "cleanup_v3":
    from aic_domain.box_push.utils import BoxPushTrajectories
    from aic_domain.box_push_v2.agent import BoxPushAIAgent_Indv
    from aic_domain.box_push_v2.simulator import BoxPushSimulatorV2
    from aic_domain.box_push_v2.maps import MAP_CLEANUP_V3
    from aic_domain.box_push_v2.policy import Policy_Cleanup
    from aic_domain.box_push_v2.mdp import (MDP_Cleanup_Agent, MDP_Cleanup_Task)
    sim = BoxPushSimulatorV2(0)
    TEMPERATURE = 0.3
    GAME_MAP = MAP_CLEANUP_V3
    SAVE_PREFIX = GAME_MAP["name"]
    MDP_TASK = MDP_Cleanup_Task(**GAME_MAP)
    MDP_AGENT = MDP_Cleanup_Agent(**GAME_MAP)
    POLICY_1 = Policy_Cleanup(MDP_TASK, MDP_AGENT, TEMPERATURE, 0)
    POLICY_2 = Policy_Cleanup(MDP_TASK, MDP_AGENT, TEMPERATURE, 1)
    AGENT_1 = BoxPushAIAgent_Indv(POLICY_1, agent_idx=sim.AGENT1)
    AGENT_2 = BoxPushAIAgent_Indv(POLICY_2, agent_idx=sim.AGENT2)
    AGENTS = [AGENT_1, AGENT_2]
    train_data = BoxPushTrajectories(MDP_TASK, MDP_AGENT)

  # load files
  ##################################################
  DATA_DIR = os.path.join(os.path.dirname(__file__), "data/")
  dir_postifx = "_train"
  if opt:
    dir_postifx += "_opt"

  TRAIN_DIR = os.path.join(DATA_DIR, SAVE_PREFIX + dir_postifx)

  file_names = glob.glob(os.path.join(TRAIN_DIR, '*.txt'))
  random.shuffle(file_names)

  train_data.load_from_files(file_names)
  traj_labeled_ver = train_data.get_as_row_lists(no_latent_label=False,
                                                 include_terminal=True)
  traj_unlabel_ver = train_data.get_as_row_lists(no_latent_label=True,
                                                 include_terminal=True)
  num_traj = len(traj_labeled_ver)

  # baselines
  ##################################################
  # all data
  traj_trainset = traj_labeled_ver
  num_agents = len(AGENTS)
  traj_sa_each_agent = [[] for _ in range(num_agents)]
  for idx_a in range(num_agents):
    for traj in traj_trainset:
      traj_sa = []
      for s, a, x in traj:
        act = None if a is None else a[idx_a]

        traj_sa.append((s, act))

      traj_sa_each_agent[idx_a].append(traj_sa)

  traj_sa_joint = []
  for traj in traj_trainset:
    traj_sa = []
    for s, a, x in traj:
      traj_sa.append((s, a))

    traj_sa_joint.append(traj_sa)

  LOG_DIR = os.path.join(os.path.dirname(__file__), "logs/")
  if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
  logpath = LOG_DIR + str(datetime.today())

  BC = True
  num_iterations = 500
  list_policy = []
  save_prefix = SAVE_PREFIX
  save_prefix += "_opt" if opt else ""
  if BC:
    save_prefix += "_bc_"
    for idx_a in range(num_agents):
      policy = bc_dnn(MDP_TASK.num_states, MDP_AGENT.num_actions,
                      traj_sa_each_agent[idx_a], logpath + f"/bc_a{idx_a}", 64,
                      num_iterations)
      list_policy.append(policy)
  else:
    save_prefix += "_gail_"
    init_states = ([0] * len(GAME_MAP["boxes"]), GAME_MAP["a1_init"],
                   GAME_MAP["a2_init"])
    init_sidx = MDP_TASK.conv_sim_states_to_mdp_sidx(init_states)
    list_policy = gail_w_ppo(MDP_TASK, [init_sidx],
                             traj_sa_joint,
                             n_envs=1,
                             logpath=logpath + "/gail",
                             n_steps=128,
                             total_timesteps=num_iterations)

  temp_dir = DATA_DIR + "learned_models/"
  if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
  save_prefix += ("%d" % (len(traj_trainset), ))

  # save models
  save_prefix = os.path.join(temp_dir, save_prefix)

  for idx, policy in enumerate(list_policy):
    np.save(save_prefix + "_pi" + f"_a{idx + 1}", policy)


if __name__ == "__main__":
  main()
