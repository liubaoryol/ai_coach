import os
import glob
import click
import logging
import random
import numpy as np
# from ai_coach_core.model_learning.BTIL.btil_for_two import BTILforTwo
from ai_coach_core.model_learning.BTIL import BTIL
from ai_coach_domain.helper import TrueModelConverter
from ai_coach_domain.box_push.agent_model import (
    assumed_initial_mental_distribution)
from ai_coach_domain.box_push.utils import BoxPushTrajectories
from ai_coach_domain.box_push_v2.agent import (BoxPushAIAgent_PO_Indv,
                                               BoxPushAIAgent_PO_Team)
from ai_coach_domain.box_push_v2.simulator import BoxPushSimulatorV2
from ai_coach_domain.box_push_v2.maps import MAP_CLEANUP, MAP_MOVERS
from ai_coach_domain.box_push_v2.policy import Policy_Cleanup, Policy_Movers
from ai_coach_domain.box_push_v2.mdp import (MDP_Cleanup_Agent,
                                             MDP_Cleanup_Task, MDP_Movers_Agent,
                                             MDP_Movers_Task)
import helper


# yapf: disable
@click.command()
@click.option("--is-team", type=bool, default=True, help="team / indv")
@click.option("--synthetic", type=bool, default=True, help="")
@click.option("--num-training-data", type=int, default=100, help="")
@click.option("--supervision", type=float, default=1.0, help="value should be between 0.0 and 1.0")  # noqa: E501
@click.option("--use-true-tx", type=bool, default=False, help="")
@click.option("--gen-trainset", type=bool, default=True, help="")
@click.option("--beta-pi", type=float, default=1.1, help="")
@click.option("--beta-tx", type=float, default=1.1, help="")
# yapf: enable
def main(is_team, synthetic, num_training_data, supervision, use_true_tx,
         gen_trainset, beta_pi, beta_tx):
  logging.info("is_TEAM: %s" % (is_team, ))
  logging.info("synthetic: %s" % (synthetic, ))
  logging.info("num training data: %s" % (num_training_data, ))
  logging.info("supervision: %s" % (supervision, ))
  logging.info("use true Tx: %s" % (use_true_tx, ))
  logging.info("Gen trainset: %s" % (gen_trainset, ))
  logging.info("beta pi: %s" % (beta_pi, ))
  logging.info("beta Tx: %s" % (beta_tx, ))

  assert synthetic or not use_true_tx

  # define the domain where trajectories were generated
  ##################################################
  sim = BoxPushSimulatorV2(0)
  TEMPERATURE = 0.3

  if is_team:
    GAME_MAP = MAP_MOVERS
    SAVE_PREFIX = GAME_MAP["name"] + "_v2"
    MDP_TASK = MDP_Movers_Task(**GAME_MAP)
    MDP_AGENT = MDP_Movers_Agent(**GAME_MAP)
    POLICY_1 = Policy_Movers(MDP_TASK, MDP_AGENT, TEMPERATURE, 0)
    POLICY_2 = Policy_Movers(MDP_TASK, MDP_AGENT, TEMPERATURE, 1)
    init_states = ([0] * len(GAME_MAP["boxes"]), GAME_MAP["a1_init"],
                   GAME_MAP["a2_init"])
    AGENT_1 = BoxPushAIAgent_PO_Team(init_states,
                                     POLICY_1,
                                     agent_idx=sim.AGENT1)
    AGENT_2 = BoxPushAIAgent_PO_Team(init_states,
                                     POLICY_2,
                                     agent_idx=sim.AGENT2)
  else:
    GAME_MAP = MAP_CLEANUP
    SAVE_PREFIX = GAME_MAP["name"] + "_v2"
    MDP_TASK = MDP_Cleanup_Task(**GAME_MAP)
    MDP_AGENT = MDP_Cleanup_Agent(**GAME_MAP)
    POLICY_1 = Policy_Cleanup(MDP_TASK, MDP_AGENT, TEMPERATURE, 0)
    POLICY_2 = Policy_Cleanup(MDP_TASK, MDP_AGENT, TEMPERATURE, 1)
    init_states = ([0] * len(GAME_MAP["boxes"]), GAME_MAP["a1_init"],
                   GAME_MAP["a2_init"])
    AGENT_1 = BoxPushAIAgent_PO_Indv(init_states,
                                     POLICY_1,
                                     agent_idx=sim.AGENT1)
    AGENT_2 = BoxPushAIAgent_PO_Indv(init_states,
                                     POLICY_2,
                                     agent_idx=sim.AGENT2)

  sim.init_game(**GAME_MAP)
  sim.set_autonomous_agent(AGENT_1, AGENT_2)

  true_methods = TrueModelConverter(AGENT_1, AGENT_2, MDP_AGENT.num_latents)

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

  fn_get_bx = None
  fn_get_Tx = None
  if synthetic:
    fn_get_bx = true_methods.get_init_latent_dist
    fn_get_Tx = true_methods.true_Tx_for_var_infer
  else:

    def assumed_init_latent_dist(agent_idx, state_idx):
      if agent_idx == 0:
        return assumed_initial_mental_distribution(0, state_idx, MDP_TASK)
      else:
        return AGENT_2.get_initial_latent_distribution(state_idx)

    fn_get_bx = assumed_init_latent_dist

  # load train set
  ##################################################
  file_names = glob.glob(os.path.join(TRAIN_DIR, '*.txt'))
  random.shuffle(file_names)

  num_train = min(num_training_data, len(file_names))
  logging.info(num_train)

  train_files = file_names[:num_train]

  train_data = BoxPushTrajectories(MDP_TASK, MDP_AGENT)
  train_data.load_from_files(train_files)
  traj_labeled_ver = train_data.get_as_row_lists(no_latent_label=False,
                                                 include_terminal=False)
  traj_unlabel_ver = train_data.get_as_row_lists(no_latent_label=True,
                                                 include_terminal=False)

  logging.info(len(traj_labeled_ver))

  # learn policy and transition
  ##################################################
  logging.info("beta: %f, %f" % (beta_pi, beta_tx))

  joint_action_num = (MDP_AGENT.num_actions, MDP_AGENT.num_actions)
  labeled_data_idx = int(num_train * supervision)

  logging.info("#########")
  logging.info("BTIL (Labeled: %d, Unlabeled: %d)" %
               (labeled_data_idx, num_train - labeled_data_idx))
  logging.info("#########")

  def transition_s(sidx, tup_aidx, sidx_n=None):
    return helper.cached_transition(DATA_DIR, SAVE_PREFIX, MDP_TASK, sidx,
                                    tup_aidx, sidx_n)

  # learning models
  btil_models = BTIL(traj_labeled_ver[0:labeled_data_idx] +
                     traj_unlabel_ver[labeled_data_idx:],
                     MDP_TASK.num_states,
                     (MDP_AGENT.num_latents, MDP_AGENT.num_latents),
                     joint_action_num,
                     transition_s,
                     trans_x_dependency=(True, True, True, False),
                     epsilon=0.01,
                     max_iteration=100)
  btil_models.set_dirichlet_prior(beta_pi, beta_tx)

  if use_true_tx:
    logging.info("Train with true Tx")
    btil_models.set_bx_and_Tx(cb_bx=fn_get_bx, cb_Tx=fn_get_Tx)
  else:
    logging.info("Train without true Tx")
    btil_models.set_bx_and_Tx(cb_bx=fn_get_bx)
  btil_models.do_inference()

  # save models
  save_dir = DATA_DIR + "learned_models/"
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)

  policy_file_name = SAVE_PREFIX + "_btil_policy_"
  policy_file_name += "synth_" if synthetic else "human_"
  policy_file_name += "withTx_" if use_true_tx else "woTx_"
  policy_file_name += "%d_%.2f" % (num_train, supervision)
  policy_file_name = os.path.join(save_dir, policy_file_name)
  np.save(policy_file_name + "_a1", btil_models.list_np_policy[0])
  np.save(policy_file_name + "_a2", btil_models.list_np_policy[1])

  if not use_true_tx:
    tx_file_name = SAVE_PREFIX + "_btil_tx_"
    tx_file_name += "synth_" if synthetic else "human_"
    tx_file_name += "%d_%.2f" % (num_train, supervision)
    tx_file_name = os.path.join(save_dir, tx_file_name)
    np.save(tx_file_name + "_a1", btil_models.list_Tx[0].np_Tx)
    np.save(tx_file_name + "_a2", btil_models.list_Tx[1].np_Tx)


if __name__ == "__main__":
  logging.basicConfig(
      level=logging.INFO,
      format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
      handlers=[logging.StreamHandler()],
      force=True)
  main()
