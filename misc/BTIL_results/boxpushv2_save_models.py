import os
import glob
import click
import logging
import random
import numpy as np
from ai_coach_core.model_learning.BTIL.btil_for_two import BTILforTwo
from ai_coach_domain.box_push.agent_model import (
    assumed_initial_mental_distribution)
from ai_coach_domain.box_push.agent import (BoxPushAIAgent_Indv1,
                                            BoxPushAIAgent_Indv2,
                                            BoxPushAIAgent_Team1,
                                            BoxPushAIAgent_Team2)
from ai_coach_domain.box_push.utils import (TrueModelConverter,
                                            BoxPushTrajectories)
from ai_coach_domain.box_push_v2.simulator import BoxPushSimulatorV2
from ai_coach_domain.box_push_v2.maps import MAP_CLEANUP, MAP_MOVERS
from ai_coach_domain.box_push_v2.policy import Policy_Cleanup, Policy_Movers
from ai_coach_domain.box_push_v2.mdp import (MDP_Cleanup_Agent,
                                             MDP_Cleanup_Task, MDP_Movers_Agent,
                                             MDP_Movers_Task)
import compute_dynamic_data_results as res


# yapf: disable
@click.command()
@click.option("--is-team", type=bool, default=True, help="team / indv")
@click.option("--synthetic", type=bool, default=False, help="")
@click.option("--num-training-data", type=int, default=200, help="")
@click.option("--supervision", type=float, default=1.0, help="value should be between 0.0 and 1.0")  # noqa: E501
@click.option("--use-true-tx", type=bool, default=True, help="")
@click.option("--gen-trainset", type=bool, default=True, help="")
# yapf: enable
def main(is_team, synthetic, num_training_data, supervision, use_true_tx,
         gen_trainset):
  logging.info("is_TEAM: %s" % (is_team, ))
  logging.info("synthetic: %s" % (synthetic, ))
  logging.info("num training data: %s" % (num_training_data, ))
  logging.info("supervision: %s" % (supervision, ))
  logging.info("use true Tx: %s" % (use_true_tx, ))
  logging.info("Gen trainset: %s" % (gen_trainset, ))

  assert synthetic or not use_true_tx

  # define the domain where trajectories were generated
  ##################################################
  sim = BoxPushSimulatorV2(0)
  TEMPERATURE = 0.3

  if is_team:
    GAME_MAP = MAP_MOVERS
    res.SAVE_PREFIX = GAME_MAP["name"] + "_v2"
    res.MDP_TASK = MDP_Movers_Task(**GAME_MAP)
    res.MDP_AGENT = MDP_Movers_Agent(**GAME_MAP)
    POLICY_1 = Policy_Movers(res.MDP_TASK, res.MDP_AGENT, TEMPERATURE, 0)
    POLICY_2 = Policy_Movers(res.MDP_TASK, res.MDP_AGENT, TEMPERATURE, 1)
    AGENT_1 = BoxPushAIAgent_Team1(POLICY_1)
    AGENT_2 = BoxPushAIAgent_Team2(POLICY_2)
    BETA_PI = 1.2
    BETA_TX1 = 1.01
    BETA_TX2 = 1.01
  else:
    GAME_MAP = MAP_CLEANUP
    res.SAVE_PREFIX = GAME_MAP["name"] + "_v2"
    res.MDP_TASK = MDP_Cleanup_Task(**GAME_MAP)
    res.MDP_AGENT = MDP_Cleanup_Agent(**GAME_MAP)
    POLICY_1 = Policy_Cleanup(res.MDP_TASK, res.MDP_AGENT, TEMPERATURE, 0)
    POLICY_2 = Policy_Cleanup(res.MDP_TASK, res.MDP_AGENT, TEMPERATURE, 1)
    AGENT_1 = BoxPushAIAgent_Indv1(POLICY_1)
    AGENT_2 = BoxPushAIAgent_Indv2(POLICY_2)
    BETA_PI = 1.01
    BETA_TX1 = 1.01
    BETA_TX2 = 1.01

  sim.init_game(**GAME_MAP)
  sim.set_autonomous_agent(AGENT_1, AGENT_2)

  true_methods = TrueModelConverter(AGENT_1, AGENT_2, res.MDP_AGENT.num_latents)

  # generate data
  ############################################################################
  DATA_DIR = os.path.join(os.path.dirname(__file__), "data/")
  TRAIN_DIR = os.path.join(DATA_DIR, res.SAVE_PREFIX + '_train')

  train_prefix = "train_"
  if gen_trainset:
    file_names = glob.glob(os.path.join(TRAIN_DIR, train_prefix + '*.txt'))
    for fmn in file_names:
      os.remove(fmn)
    sim.run_simulation(200, os.path.join(TRAIN_DIR, train_prefix), "header")

  def assumed_init_latent_dist(agent_idx, state_idx):
    if agent_idx == 0:
      return assumed_initial_mental_distribution(0, state_idx, res.MDP_TASK)
      # return AGENT_1.init_latent_dist_from_task_mdp_POV(state_idx)
    else:
      return AGENT_2.init_latent_dist_from_task_mdp_POV(state_idx)

  fn_get_bx = None
  fn_get_Tx = None
  if synthetic:
    fn_get_bx = true_methods.get_init_latent_dist
    fn_get_Tx = true_methods.true_Tx_for_var_infer
  else:
    fn_get_bx = assumed_init_latent_dist

  # load train set
  ##################################################
  file_names = glob.glob(os.path.join(TRAIN_DIR, '*.txt'))
  random.shuffle(file_names)

  num_train = min(num_training_data, len(file_names))
  logging.info(num_train)

  train_files = file_names[:num_train]

  train_data = BoxPushTrajectories(res.MDP_TASK, res.MDP_AGENT)
  train_data.load_from_files(train_files)
  traj_labeled_ver = train_data.get_as_row_lists(no_latent_label=False,
                                                 include_terminal=False)
  traj_unlabel_ver = train_data.get_as_row_lists(no_latent_label=True,
                                                 include_terminal=False)

  logging.info(len(traj_labeled_ver))

  # learn policy and transition
  ##################################################
  logging.info("beta: %f, %f, %f" % (BETA_PI, BETA_TX1, BETA_TX2))

  joint_action_num = (res.MDP_AGENT.num_actions, res.MDP_AGENT.num_actions)
  labeled_data_idx = int(num_train * supervision)

  logging.info("#########")
  logging.info("BTIL (Labeled: %d, Unlabeled: %d)" %
               (labeled_data_idx, num_train - labeled_data_idx))
  logging.info("#########")

  # learning models
  btil_models = BTILforTwo(traj_labeled_ver[0:labeled_data_idx] +
                           traj_unlabel_ver[labeled_data_idx:],
                           res.MDP_TASK.num_states,
                           res.MDP_AGENT.num_latents,
                           joint_action_num,
                           res.transition_s,
                           trans_x_dependency=(True, True, True, False),
                           epsilon=0.01,
                           max_iteration=100)
  btil_models.set_dirichlet_prior(BETA_PI, BETA_TX1, BETA_TX2)

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

  policy_file_name = res.SAVE_PREFIX + "_btil_policy_"
  policy_file_name += "synth_" if synthetic else "human_"
  policy_file_name += "withTx_" if use_true_tx else "woTx_"
  policy_file_name += "%d_%.2f" % (num_train, supervision)
  policy_file_name = os.path.join(save_dir, policy_file_name)
  np.save(policy_file_name + "_a1", btil_models.list_np_policy[0])
  np.save(policy_file_name + "_a2", btil_models.list_np_policy[1])

  if not use_true_tx:
    tx_file_name = res.SAVE_PREFIX + "_btil_tx_"
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
