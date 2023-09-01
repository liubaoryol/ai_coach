import os
import glob
import click
import logging
import random
import numpy as np
from aic_ml.BTIL.btil_abstraction import BTIL_Abstraction


# yapf: disable
@click.command()
@click.option("--domain", type=str, default="movers", help="movers / cleanup_v3 / rescue_2 /rescue_3")  # noqa: E501
@click.option("--num-data", type=int, default=1000, help="")
@click.option("--po", type=bool, default=False, help="")
# yapf: enable
def main(domain, num_data, po):
  logging.info("domain: %s" % (domain, ))
  logging.info("num training data: %s" % (num_data, ))

  # define the domain where trajectories were generated
  ##################################################
  if domain == "movers":
    from aic_domain.box_push.utils import BoxPushTrajectories
    from aic_domain.box_push_v2.agent import BoxPushAIAgent_PO_Team
    from aic_domain.box_push_v2.agent import BoxPushAIAgent_Team
    from aic_domain.box_push_v2.maps import MAP_MOVERS
    from aic_domain.box_push_v3.simulator import BoxPushSimulatorV3
    from aic_domain.box_push_v3.policy import Policy_MoversV3
    from aic_domain.box_push_v3.mdp import (MDP_MoversV3_Agent,
                                            MDP_MoversV3_Task)
    sim = BoxPushSimulatorV3(0)
    TEMPERATURE = 0.3
    GAME_MAP = MAP_MOVERS
    SAVE_PREFIX = GAME_MAP["name"]
    MDP_TASK = MDP_MoversV3_Task(**GAME_MAP)
    MDP_AGENT = MDP_MoversV3_Agent(**GAME_MAP)
    POLICY_1 = Policy_MoversV3(MDP_TASK, MDP_AGENT, TEMPERATURE, 0)
    POLICY_2 = Policy_MoversV3(MDP_TASK, MDP_AGENT, TEMPERATURE, 1)
    if po:
      init_states = ([0] * len(GAME_MAP["boxes"]), GAME_MAP["a1_init"],
                     GAME_MAP["a2_init"])
      AGENT_1 = BoxPushAIAgent_PO_Team(init_states,
                                       POLICY_1,
                                       agent_idx=sim.AGENT1)
      AGENT_2 = BoxPushAIAgent_PO_Team(init_states,
                                       POLICY_2,
                                       agent_idx=sim.AGENT2)
    else:
      AGENT_1 = BoxPushAIAgent_Team(POLICY_1, agent_idx=sim.AGENT1)
      AGENT_2 = BoxPushAIAgent_Team(POLICY_2, agent_idx=sim.AGENT2)
    AGENTS = [AGENT_1, AGENT_2]
    train_data = BoxPushTrajectories(MDP_TASK, MDP_AGENT)
  elif domain == "cleanup_v3":
    from aic_domain.box_push.utils import BoxPushTrajectories
    from aic_domain.box_push_v2.agent import BoxPushAIAgent_PO_Indv
    from aic_domain.box_push_v2.agent import BoxPushAIAgent_Indv
    from aic_domain.box_push_v2.maps import MAP_CLEANUP_V3
    from aic_domain.box_push_v3.simulator import BoxPushSimulatorV3
    from aic_domain.box_push_v3.policy import Policy_CleanupV3
    from aic_domain.box_push_v3.mdp import (MDP_CleanupV3_Agent,
                                            MDP_CleanupV3_Task)
    sim = BoxPushSimulatorV3(False)
    TEMPERATURE = 0.3
    GAME_MAP = MAP_CLEANUP_V3
    SAVE_PREFIX = GAME_MAP["name"]
    MDP_TASK = MDP_CleanupV3_Task(**GAME_MAP)
    MDP_AGENT = MDP_CleanupV3_Agent(**GAME_MAP)
    POLICY_1 = Policy_CleanupV3(MDP_TASK, MDP_AGENT, TEMPERATURE, 0)
    POLICY_2 = Policy_CleanupV3(MDP_TASK, MDP_AGENT, TEMPERATURE, 1)
    if po:
      init_states = ([0] * len(GAME_MAP["boxes"]), GAME_MAP["a1_init"],
                     GAME_MAP["a2_init"])
      AGENT_1 = BoxPushAIAgent_PO_Indv(init_states,
                                       POLICY_1,
                                       agent_idx=sim.AGENT1)
      AGENT_2 = BoxPushAIAgent_PO_Indv(init_states,
                                       POLICY_2,
                                       agent_idx=sim.AGENT2)
    else:
      AGENT_1 = BoxPushAIAgent_Indv(POLICY_1, agent_idx=sim.AGENT1)
      AGENT_2 = BoxPushAIAgent_Indv(POLICY_2, agent_idx=sim.AGENT2)
    AGENTS = [AGENT_1, AGENT_2]
    train_data = BoxPushTrajectories(MDP_TASK, MDP_AGENT)
  else:
    raise NotImplementedError

  sim.init_game(**GAME_MAP)
  sim.set_autonomous_agent(*AGENTS)

  # generate data
  ############################################################################
  DATA_DIR = os.path.join(os.path.dirname(__file__), "data/")
  train_dir = os.path.join(DATA_DIR, SAVE_PREFIX + '_train')
  if po:
    train_dir += "_po"

  train_prefix = "train_"
  file_names = glob.glob(os.path.join(train_dir, train_prefix + '*.txt'))
  for fmn in file_names:
    os.remove(fmn)
  sim.run_simulation(num_data, os.path.join(train_dir, train_prefix), "header")


if __name__ == "__main__":
  logging.basicConfig(
      level=logging.INFO,
      format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
      handlers=[logging.StreamHandler()],
      force=True)
  main()
