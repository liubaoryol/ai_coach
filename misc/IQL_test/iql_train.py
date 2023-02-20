import os
import glob
import logging
import random
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ai_coach_core.model_learning.IQLearn.iql import run_iql
from ai_coach_core.model_learning.IQLearn.utils.utils import (
    conv_trajectories_2_iql_format)
from datetime import datetime
import ai_coach_core.gym


@click.command()
@click.option("--domain", type=str, default="movers", help="")
@click.option("--opt", type=bool, default=True, help="")
def main(domain, opt):

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

  # gym env
  ##################################################
  sim = BoxPushSimulatorV3(False)
  sim.init_game(**GAME_MAP)
  possible_init_states = []
  init_bstate = [0] * len(GAME_MAP["boxes"])
  for pos1 in sim.possible_positions:
    for pos2 in sim.possible_positions:
      init_states = (init_bstate, pos1, pos2)
      init_sidx = MDP_TASK.conv_sim_states_to_mdp_sidx(init_states)
      possible_init_states.append(init_sidx)

  env_kwargs = {
      'mdp': MDP_TASK,
      'possible_init_states': possible_init_states,
      'use_central_action': True
  }

  # load files
  ##################################################
  DATA_DIR = os.path.join(os.path.dirname(__file__), "data/")
  dir_suffix = "_train"
  if opt:
    dir_suffix += "_opt"

  TRAIN_DIR = os.path.join(DATA_DIR, SAVE_PREFIX + dir_suffix)

  file_names = glob.glob(os.path.join(TRAIN_DIR, '*.txt'))
  random.shuffle(file_names)

  train_data.load_from_files(file_names)
  traj_labeled_ver = train_data.get_as_row_lists(no_latent_label=False,
                                                 include_terminal=True)
  num_traj = len(traj_labeled_ver)

  # convert trajectories
  dir_iq_data = os.path.join(DATA_DIR, "iq_data", f"{domain}_{num_traj}")
  if not os.path.exists(dir_iq_data):
    os.makedirs(dir_iq_data)
  path_iq_data = os.path.join(dir_iq_data, f"{domain}_{num_traj}.pkl")

  conv_trajectories_2_iql_format(traj_labeled_ver, MDP_TASK.conv_action_to_idx,
                                 lambda s, a: -1, path_iq_data)

  ##################################################
  LOG_DIR = os.path.join(os.path.dirname(__file__), "logs/")
  if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
  output_dir = os.path.join(os.path.dirname(__file__), "output/")
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  num_iterations = 20000
  save_prefix = SAVE_PREFIX
  save_prefix += "_opt" if opt else ""

  IQL = True
  if IQL:
    run_iql('envfrommdp-v0',
            env_kwargs,
            0,
            128,
            path_iq_data,
            num_traj,
            LOG_DIR,
            output_dir,
            output_suffix="_opt100",
            replay_mem=50000,
            initial_mem=1000,
            eps_steps=200,
            eps_window=10,
            num_learn_steps=num_iterations,
            agent_name='sac',
            log_interval=100,
            eval_interval=1000,
            hidden_dim=128,
            hidden_depth=2,
            gumbel_temperature=1.0)


if __name__ == "__main__":
  main()
