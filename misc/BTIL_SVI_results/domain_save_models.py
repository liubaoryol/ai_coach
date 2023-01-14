import os
import glob
import click
import logging
import random
import numpy as np
from ai_coach_core.model_learning.BTIL.btil_svi import BTIL_SVI
from ai_coach_domain.helper import TrueModelConverter

import helper


# yapf: disable
@click.command()
@click.option("--domain", type=str, default="movers", help="movers / cleanup_v3 / rescue_2 /rescue_3")  # noqa: E501
@click.option("--num-training-data", type=int, default=500, help="")
@click.option("--gen-trainset", type=bool, default=False, help="")
@click.option("--gem-gamma", type=float, default=3, help="")
@click.option("--tx-alpha", type=float, default=5, help="")
@click.option("--pi-rho", type=float, default=1.01, help="")
@click.option("--num-x", type=int, default=9, help="")
@click.option("--tx-dependency", type=str, default="FTTT",
              help="sequence of T or F indicating dependency on cur_state, actions, and next_state")  # noqa: E501
# yapf: enable
def main(domain, num_training_data, gen_trainset, gem_gamma, tx_alpha, pi_rho,
         num_x, tx_dependency):
  logging.info("domain: %s" % (domain, ))
  logging.info("num training data: %s" % (num_training_data, ))
  logging.info("Gen trainset: %s" % (gen_trainset, ))
  logging.info("GEM gamma: %s" % (gem_gamma, ))
  logging.info("Tx alpha: %s" % (tx_alpha, ))
  logging.info("pi rho: %s" % (pi_rho, ))
  logging.info("num x: %s" % (num_x, ))
  logging.info("Tx dependency: %s" % (tx_dependency, ))

  # define the domain where trajectories were generated
  ##################################################
  if domain == "movers":
    from ai_coach_domain.box_push.utils import BoxPushTrajectories
    from ai_coach_domain.box_push_v2.agent import BoxPushAIAgent_Team
    from ai_coach_domain.box_push_v2.simulator import BoxPushSimulatorV2
    from ai_coach_domain.box_push_v2.maps import MAP_MOVERS
    from ai_coach_domain.box_push_v2.policy import Policy_Movers
    from ai_coach_domain.box_push_v2.mdp import (MDP_Movers_Agent,
                                                 MDP_Movers_Task)
    sim = BoxPushSimulatorV2(0)
    TEMPERATURE = 0.3
    GAME_MAP = MAP_MOVERS
    SAVE_PREFIX = GAME_MAP["name"]
    MDP_TASK = MDP_Movers_Task(**GAME_MAP)
    MDP_AGENT = MDP_Movers_Agent(**GAME_MAP)
    POLICY_1 = Policy_Movers(MDP_TASK, MDP_AGENT, TEMPERATURE, 0)
    POLICY_2 = Policy_Movers(MDP_TASK, MDP_AGENT, TEMPERATURE, 1)
    # init_states = ([0] * len(GAME_MAP["boxes"]), GAME_MAP["a1_init"],
    #                GAME_MAP["a2_init"])
    AGENT_1 = BoxPushAIAgent_Team(POLICY_1, agent_idx=sim.AGENT1)
    AGENT_2 = BoxPushAIAgent_Team(POLICY_2, agent_idx=sim.AGENT2)
    AGENTS = [AGENT_1, AGENT_2]
    train_data = BoxPushTrajectories(MDP_TASK, MDP_AGENT)
  elif domain == "cleanup_v2":
    from ai_coach_domain.box_push.utils import BoxPushTrajectories
    from ai_coach_domain.box_push_v2.agent import BoxPushAIAgent_Indv
    from ai_coach_domain.box_push_v2.simulator import BoxPushSimulatorV2
    from ai_coach_domain.box_push_v2.maps import MAP_CLEANUP_V2
    from ai_coach_domain.box_push_v2.policy import Policy_Cleanup
    from ai_coach_domain.box_push_v2.mdp import (MDP_Cleanup_Agent,
                                                 MDP_Cleanup_Task)
    sim = BoxPushSimulatorV2(0)
    TEMPERATURE = 0.3
    GAME_MAP = MAP_CLEANUP_V2
    SAVE_PREFIX = GAME_MAP["name"]
    MDP_TASK = MDP_Cleanup_Task(**GAME_MAP)
    MDP_AGENT = MDP_Cleanup_Agent(**GAME_MAP)
    POLICY_1 = Policy_Cleanup(MDP_TASK, MDP_AGENT, TEMPERATURE, 0)
    POLICY_2 = Policy_Cleanup(MDP_TASK, MDP_AGENT, TEMPERATURE, 1)
    AGENT_1 = BoxPushAIAgent_Indv(POLICY_1, agent_idx=sim.AGENT1)
    AGENT_2 = BoxPushAIAgent_Indv(POLICY_2, agent_idx=sim.AGENT2)
    AGENTS = [AGENT_1, AGENT_2]
    train_data = BoxPushTrajectories(MDP_TASK, MDP_AGENT)
  elif domain == "cleanup_v3":
    from ai_coach_domain.box_push.utils import BoxPushTrajectories
    from ai_coach_domain.box_push_v2.agent import BoxPushAIAgent_Indv
    from ai_coach_domain.box_push_v2.simulator import BoxPushSimulatorV2
    from ai_coach_domain.box_push_v2.maps import MAP_CLEANUP_V3
    from ai_coach_domain.box_push_v2.policy import Policy_Cleanup
    from ai_coach_domain.box_push_v2.mdp import (MDP_Cleanup_Agent,
                                                 MDP_Cleanup_Task)
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
  elif domain == "rescue_2":
    from ai_coach_domain.rescue.agent import AIAgent_Rescue_PartialObs
    from ai_coach_domain.rescue.simulator import RescueSimulator
    from ai_coach_domain.rescue.maps import MAP_RESCUE
    from ai_coach_domain.rescue.policy import Policy_Rescue
    from ai_coach_domain.rescue.mdp import MDP_Rescue_Agent, MDP_Rescue_Task
    from ai_coach_domain.rescue.utils import RescueTrajectories
    sim = RescueSimulator()
    sim.max_steps = 30
    TEMPERATURE = 0.3

    GAME_MAP = MAP_RESCUE
    SAVE_PREFIX = GAME_MAP["name"]
    MDP_TASK = MDP_Rescue_Task(**GAME_MAP)
    MDP_AGENT = MDP_Rescue_Agent(**GAME_MAP)
    POLICY_1 = Policy_Rescue(MDP_TASK, MDP_AGENT, TEMPERATURE, 0)
    POLICY_2 = Policy_Rescue(MDP_TASK, MDP_AGENT, TEMPERATURE, 1)

    init_states = ([1] * len(GAME_MAP["work_locations"]), GAME_MAP["a1_init"],
                   GAME_MAP["a2_init"])
    AGENT_1 = AIAgent_Rescue_PartialObs(init_states, 0, POLICY_1)
    AGENT_2 = AIAgent_Rescue_PartialObs(init_states, 1, POLICY_2)

    AGENTS = [AGENT_1, AGENT_2]

    def conv_latent_to_idx(agent_idx, latent):
      if agent_idx == 0:
        return AGENT_1.conv_latent_to_idx(latent)
      else:
        return AGENT_2.conv_latent_to_idx(latent)

    train_data = RescueTrajectories(
        MDP_TASK, (MDP_AGENT.num_latents, MDP_AGENT.num_latents),
        conv_latent_to_idx)
  elif domain == "rescue_3":
    from ai_coach_domain.rescue_v2.agent import AIAgent_Rescue_PartialObs
    from ai_coach_domain.rescue_v2.simulator import RescueSimulatorV2
    from ai_coach_domain.rescue_v2.maps import MAP_RESCUE
    from ai_coach_domain.rescue_v2.policy import Policy_Rescue
    from ai_coach_domain.rescue_v2.mdp import MDP_Rescue_Agent, MDP_Rescue_Task
    from ai_coach_domain.rescue_v2.utils import RescueV2Trajectories
    sim = RescueSimulatorV2()
    sim.max_steps = 15

    TEMPERATURE = 0.3
    GAME_MAP = MAP_RESCUE
    SAVE_PREFIX = GAME_MAP["name"]
    MDP_TASK = MDP_Rescue_Task(**GAME_MAP)
    MDP_AGENT = MDP_Rescue_Agent(**GAME_MAP)
    POLICY_1 = Policy_Rescue(MDP_TASK, MDP_AGENT, TEMPERATURE, 0)
    POLICY_2 = Policy_Rescue(MDP_TASK, MDP_AGENT, TEMPERATURE, 1)
    POLICY_3 = Policy_Rescue(MDP_TASK, MDP_AGENT, TEMPERATURE, 2)

    init_states = ([1] * len(GAME_MAP["work_locations"]), GAME_MAP["a1_init"],
                   GAME_MAP["a2_init"], GAME_MAP["a3_init"])
    AGENT_1 = AIAgent_Rescue_PartialObs(init_states, 0, POLICY_1)
    AGENT_2 = AIAgent_Rescue_PartialObs(init_states, 1, POLICY_2)
    AGENT_3 = AIAgent_Rescue_PartialObs(init_states, 2, POLICY_3)

    AGENTS = [AGENT_1, AGENT_2, AGENT_3]

    def conv_latent_to_idx(agent_idx, latent):
      return AGENTS[agent_idx].conv_latent_to_idx(latent)

    train_data = RescueV2Trajectories(
        MDP_TASK,
        (MDP_AGENT.num_latents, MDP_AGENT.num_latents, MDP_AGENT.num_latents),
        conv_latent_to_idx)
  else:
    raise NotImplementedError

  sim.init_game(**GAME_MAP)
  sim.set_autonomous_agent(*AGENTS)

  tuple_tx_dependency = []
  for cha in tx_dependency:
    if cha == "T":
      tuple_tx_dependency.append(True)
    else:
      tuple_tx_dependency.append(False)

  tuple_tx_dependency = tuple(tuple_tx_dependency)

  # generate data
  ############################################################################
  DATA_DIR = os.path.join(os.path.dirname(__file__), "data/")
  TRAIN_DIR = os.path.join(DATA_DIR, SAVE_PREFIX + '_train')

  train_prefix = "train_"
  if gen_trainset:
    file_names = glob.glob(os.path.join(TRAIN_DIR, train_prefix + '*.txt'))
    for fmn in file_names:
      os.remove(fmn)
    sim.run_simulation(num_training_data, os.path.join(TRAIN_DIR, train_prefix),
                       "header")

  # load train set
  ##################################################
  file_names = glob.glob(os.path.join(TRAIN_DIR, '*.txt'))
  random.shuffle(file_names)

  num_train = min(num_training_data, len(file_names))
  logging.info(num_train)

  train_files = file_names[:num_train]

  train_data.load_from_files(train_files)
  traj_labeled_ver = train_data.get_as_row_lists(no_latent_label=False,
                                                 include_terminal=False)
  traj_unlabel_ver = train_data.get_as_row_lists(no_latent_label=True,
                                                 include_terminal=False)

  logging.info(len(traj_labeled_ver))

  # learn policy and transition
  ##################################################
  logging.info("gamma, alpha, rho: %f, %f, %f" % (gem_gamma, tx_alpha, pi_rho))

  joint_action_num = tuple([MDP_AGENT.num_actions] * len(AGENTS))

  logging.info("#########")
  logging.info("BTIL (Unlabeled: %d)" % (num_train, ))
  logging.info("#########")

  # learning models
  btil_models = BTIL_SVI(traj_unlabel_ver,
                         MDP_TASK.num_states,
                         tuple([num_x + 1] * len(AGENTS)),
                         joint_action_num,
                         trans_x_dependency=tuple_tx_dependency,
                         epsilon=0.01,
                         max_iteration=150,
                         forgetting_rate=0.99)
  btil_models.set_prior(gem_gamma, tx_alpha, pi_rho)

  btil_models.do_inference()

  # save models
  save_dir = DATA_DIR + "learned_models/"
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)

  policy_file_name = SAVE_PREFIX + "_btil_svi_policy_"
  policy_file_name += "unsup_"
  policy_file_name += tx_dependency + "_"
  policy_file_name += ("%d_f" % (num_train, )).replace('.', ',')
  policy_file_name = os.path.join(save_dir, policy_file_name)
  for idx in range(len(btil_models.list_np_policy)):
    np.save(policy_file_name + f"_a{idx + 1}", btil_models.list_np_policy[idx])

  tx_file_name = SAVE_PREFIX + "_btil_svi_tx_"
  tx_file_name += "unsup_"
  tx_file_name += tx_dependency + "_"
  tx_file_name += ("%d_" % (num_train, )).replace('.', ',')
  tx_file_name = os.path.join(save_dir, tx_file_name)
  for idx in range(len(btil_models.list_Tx)):
    np.save(tx_file_name + f"_a{idx + 1}", btil_models.list_Tx[idx].np_Tx)

  bx_file_name = SAVE_PREFIX + "_btil_svi_bx_"
  bx_file_name += "unsup_"
  bx_file_name += tx_dependency + "_"
  bx_file_name += ("%d_" % (num_train, )).replace('.', ',')
  bx_file_name = os.path.join(save_dir, bx_file_name)
  for idx in range(len(btil_models.list_bx)):
    np.save(bx_file_name + f"_a{idx + 1}", btil_models.list_bx[idx])


if __name__ == "__main__":
  logging.basicConfig(
      level=logging.INFO,
      format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
      handlers=[logging.StreamHandler()],
      force=True)
  main()
