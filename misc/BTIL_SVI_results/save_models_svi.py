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
@click.option("--supervision", type=float, default=0.3, help="value should be between 0.0 and 1.0")  # noqa: E501
@click.option("--gem-prior", type=float, default=3, help="")
@click.option("--tx-prior", type=float, default=3, help="")
@click.option("--pi-prior", type=float, default=3, help="")
@click.option("--num-x", type=int, default=4, help="")
@click.option("--num-iteration", type=int, default=100, help="")
@click.option("--batch-size", type=int, default=500, help="")
@click.option("--tx-dependency", type=str, default="FTTT",
              help="sequence of T or F indicating dependency on cur_state, actions, and next_state")  # noqa: E501
# yapf: enable
def main(domain, num_training_data, supervision, gem_prior, tx_prior, pi_prior,
         num_x, batch_size, tx_dependency, num_iteration):
  logging.info("domain: %s" % (domain, ))
  logging.info("num training data: %s" % (num_training_data, ))
  logging.info("supervision: %s" % (supervision, ))
  logging.info("GEM gamma: %s" % (gem_prior, ))
  logging.info("Tx alpha: %s" % (tx_prior, ))
  logging.info("pi rho: %s" % (pi_prior, ))
  logging.info("num x: %s" % (num_x, ))
  logging.info("batch size: %s" % (batch_size, ))
  logging.info("Tx dependency: %s" % (tx_dependency, ))

  # define the domain where trajectories were generated
  ##################################################
  if domain == "movers":
    from ai_coach_domain.box_push.utils import BoxPushTrajectories
    from ai_coach_domain.box_push_v2.agent import BoxPushAIAgent_Team
    from ai_coach_domain.box_push_v3.simulator import BoxPushSimulatorV3
    from ai_coach_domain.box_push_v2.maps import MAP_MOVERS
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
  logging.info("gamma, tx,  pi: %f, %f, %f" % (gem_prior, tx_prior, pi_prior))

  joint_action_num = tuple([MDP_AGENT.num_actions] * len(AGENTS))
  labeled_data_idx = int(num_train * supervision)

  logging.info("#########")
  logging.info("BTIL (Unlabeled: %d)" % (num_train, ))
  logging.info("#########")

  # learning models
  btil_models = BTIL_SVI(traj_labeled_ver[0:labeled_data_idx] +
                         traj_unlabel_ver[labeled_data_idx:],
                         MDP_TASK.num_states,
                         tuple([num_x] * len(AGENTS)),
                         joint_action_num,
                         trans_x_dependency=tuple_tx_dependency,
                         epsilon=0.1,
                         max_iteration=num_iteration,
                         lr=1,
                         decay=0,
                         no_gem=True)
  btil_models.set_prior(gem_prior, tx_prior, pi_prior)

  btil_models.initialize_param()
  btil_models.do_inference(batch_size)

  # save models
  save_dir = DATA_DIR + "learned_models/"
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)

  policy_file_name = SAVE_PREFIX + "_btil_svi_policy_"
  policy_file_name += tx_dependency + "_"
  policy_file_name += ("%d_f" % (num_train, )).replace('.', ',')
  policy_file_name = os.path.join(save_dir, policy_file_name)
  for idx in range(len(btil_models.list_np_policy)):
    np.save(policy_file_name + f"_a{idx + 1}", btil_models.list_np_policy[idx])

  tx_file_name = SAVE_PREFIX + "_btil_svi_tx_"
  tx_file_name += tx_dependency + "_"
  tx_file_name += ("%d_" % (num_train, )).replace('.', ',')
  tx_file_name = os.path.join(save_dir, tx_file_name)
  for idx in range(len(btil_models.list_Tx)):
    np.save(tx_file_name + f"_a{idx + 1}", btil_models.list_Tx[idx].np_Tx)

  bx_file_name = SAVE_PREFIX + "_btil_svi_bx_"
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
