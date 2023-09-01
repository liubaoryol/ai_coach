import os
import glob
import logging
import random
from tqdm import tqdm
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aic_ml.baselines.sb3_algorithms import behavior_cloning_sb3, gail_w_ppo
from datetime import datetime


def is_compatible(domain, state, tup_latents):
  if domain == "movers":
    return tup_latents[0] == tup_latents[1]


@click.command()
@click.option("--domain", type=str, default="movers", help="")
@click.option("--gen-data", type=bool, default=True, help="")
@click.option("--num-data", type=int, default=100, help="")
def main(domain, gen_data, num_data):

  # define the domain where trajectories were generated
  ##################################################
  if domain == "movers":
    from aic_domain.box_push.utils import BoxPushTrajectories
    from aic_domain.box_push_v2.agent import BoxPushAIAgent_Team
    from aic_domain.box_push_v3.simulator import BoxPushSimulatorV3
    from aic_domain.box_push_v2.maps import MAP_MOVERS
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

  # generate data
  ############################################################################
  sim.init_game(**GAME_MAP)
  sim.set_autonomous_agent(*AGENTS)

  DATA_DIR = os.path.join(os.path.dirname(__file__), "data/")
  TRAIN_DIR = os.path.join(DATA_DIR, SAVE_PREFIX + '_train_opt')

  train_prefix = "opt_train_"
  if gen_data:
    count = 0
    for _ in tqdm(range(num_data * 100)):
      completed = True
      while not sim.is_finished():
        map_agent_2_action = sim.get_joint_action()
        sim.take_a_step(map_agent_2_action)
        if not is_compatible(domain, sim.get_state_for_each_agent(0),
                             (sim.agent_1.get_current_latent(),
                              sim.agent_2.get_current_latent())):
          completed = False
          break

      if completed:
        file_name = os.path.join(TRAIN_DIR, train_prefix) + "%d.txt" % (count, )
        sim.save_history(file_name, "header")
        count += 1
        if count >= num_data:
          break

      sim.reset_game()


if __name__ == "__main__":
  main()
