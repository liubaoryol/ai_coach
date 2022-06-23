import os
import glob
import click
import logging
import random
import numpy as np
from ai_coach_core.model_learning.BTIL.btil_for_two import BTILforTwo
import compute_dynamic_data_results as tbp


# yapf: disable
@click.command()
@click.option("--is-team", type=bool, default=True, help="team / indv")
@click.option("--synthetic", type=bool, default=False, help="")
@click.option("--num-training-data", type=int, default=200, help="")
@click.option("--supervision", type=float, default=1.0, help="value should be between 0.0 and 1.0")  # noqa: E501
@click.option("--use-true-tx", type=bool, default=True, help="")
# yapf: enable
def main(is_team, synthetic, num_training_data, supervision, use_true_tx):
  logging.info("is_TEAM: %s" % (is_team, ))
  logging.info("synthetic: %s" % (synthetic, ))
  logging.info("num training data: %s" % (num_training_data, ))
  logging.info("supervision: %s" % (supervision, ))
  logging.info("use true Tx: %s" % (use_true_tx, ))

  GAME_MAP = tbp.bp_maps.EXP1_MAP
  BoxPushPolicyTeam = tbp.bp_policy.BoxPushPolicyTeamExp1
  BoxPushPolicyIndv = tbp.bp_policy.BoxPushPolicyIndvExp1

  if is_team:
    tbp.SAVE_PREFIX = GAME_MAP["name"] + "_team"
    tbp.BoxPushSimulator = tbp.bp_sim.BoxPushSimulator_AlwaysTogether
  else:
    tbp.SAVE_PREFIX = GAME_MAP["name"] + "_indv"
    tbp.BoxPushSimulator = tbp.bp_sim.BoxPushSimulator_AlwaysAlone

  sim = tbp.BoxPushSimulator(0)
  sim.init_game(**GAME_MAP)
  TEMPERATURE = 0.3

  if is_team:
    tbp.MDP_AGENT = tbp.bp_mdp.BoxPushTeamMDP_AlwaysTogether(**GAME_MAP)
    tbp.MDP_TASK = tbp.MDP_AGENT
    policy1 = BoxPushPolicyTeam(tbp.MDP_TASK, TEMPERATURE,
                                tbp.BoxPushSimulator.AGENT1)
    policy2 = BoxPushPolicyTeam(tbp.MDP_TASK, TEMPERATURE,
                                tbp.BoxPushSimulator.AGENT2)
    agent1 = tbp.bp_agent.BoxPushAIAgent_Team1(policy1)
    agent2 = tbp.bp_agent.BoxPushAIAgent_Team2(policy2)
  else:
    tbp.MDP_AGENT = tbp.bp_mdp.BoxPushAgentMDP_AlwaysAlone(**GAME_MAP)
    tbp.MDP_TASK = tbp.bp_mdp.BoxPushTeamMDP_AlwaysAlone(**GAME_MAP)
    policy1 = BoxPushPolicyIndv(tbp.MDP_TASK,
                                tbp.MDP_AGENT,
                                temperature=TEMPERATURE,
                                agent_idx=tbp.BoxPushSimulator.AGENT1)
    policy2 = BoxPushPolicyIndv(tbp.MDP_TASK,
                                tbp.MDP_AGENT,
                                temperature=TEMPERATURE,
                                agent_idx=tbp.BoxPushSimulator.AGENT2)
    agent1 = tbp.bp_agent.BoxPushAIAgent_Indv1(policy1)
    agent2 = tbp.bp_agent.BoxPushAIAgent_Indv2(policy2)

  sim.set_autonomous_agent(agent1, agent2)

  true_methods = tbp.TrueModelConverter(agent1, agent2,
                                        tbp.MDP_AGENT.num_latents)

  train_dir = None
  DATA_DIR = os.path.join(os.path.dirname(__file__), "data/")
  if synthetic:
    train_dir = os.path.join(DATA_DIR, tbp.SAVE_PREFIX + '_box_push_train')
  else:
    aws_dir = os.path.join(os.path.dirname(__file__), "aws_data_test/")

    if is_team:
      train_dir = os.path.join(aws_dir, 'domain1')
    else:
      train_dir = os.path.join(aws_dir, 'domain2')

  file_names = glob.glob(os.path.join(train_dir, '*.txt'))
  random.shuffle(file_names)

  num_train = min(num_training_data, len(file_names))
  logging.info(num_train)

  train_files = file_names[:num_train]

  # load train set
  ##################################################
  train_data = tbp.BoxPushTrajectories(tbp.MDP_AGENT.num_latents, sim)
  train_data.load_from_files(train_files)
  traj_labeled_ver = train_data.get_as_row_lists(no_latent_label=False,
                                                 include_terminal=False)
  traj_unlabel_ver = train_data.get_as_row_lists(no_latent_label=True,
                                                 include_terminal=False)

  logging.info(len(traj_labeled_ver))

  # true policy and transition
  ##################################################
  if is_team:
    BETA_PI = 1.2
    BETA_TX1 = 1.01
    BETA_TX2 = 1.01
  else:
    BETA_PI = 1.01
    BETA_TX1 = 1.01
    BETA_TX2 = 1.01
  logging.info("beta: %f, %f, %f" % (BETA_PI, BETA_TX1, BETA_TX2))

  joint_action_num = ((tbp.MDP_AGENT.a1_a_space.num_actions,
                       tbp.MDP_AGENT.a2_a_space.num_actions) if is_team else
                      (tbp.MDP_AGENT.num_actions, tbp.MDP_AGENT.num_actions))

  labeled_data_idx = int(num_train * supervision)

  logging.info("#########")
  logging.info("BTIL (Labeled: %d, Unlabeled: %d)" %
               (labeled_data_idx, num_train - labeled_data_idx))
  logging.info("#########")

  # learning models
  btil_models = BTILforTwo(traj_labeled_ver[0:labeled_data_idx] +
                           traj_unlabel_ver[labeled_data_idx:],
                           tbp.MDP_TASK.num_states,
                           tbp.MDP_AGENT.num_latents,
                           joint_action_num,
                           tbp.transition_s,
                           trans_x_dependency=(True, True, True, False),
                           epsilon=0.01,
                           max_iteration=100)
  btil_models.set_dirichlet_prior(BETA_PI, BETA_TX1, BETA_TX2)

  if use_true_tx:
    logging.info("Train with true Tx")
    btil_models.set_bx_and_Tx(cb_bx=true_methods.get_init_latent_dist,
                              cb_Tx=true_methods.true_Tx_for_var_infer)
  else:
    logging.info("Train without true Tx")
    btil_models.set_bx_and_Tx(cb_bx=true_methods.get_init_latent_dist)
  btil_models.do_inference()

  # save models
  save_dir = DATA_DIR + "learned_models/"
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)

  policy_file_name = tbp.SAVE_PREFIX + "_btil_policy_"
  policy_file_name += "synth_" if synthetic else "human_"
  policy_file_name += "withTx_" if use_true_tx else "woTx_"
  policy_file_name += "%d_%.2f" % (num_train, supervision)
  policy_file_name = os.path.join(save_dir, policy_file_name)
  np.savez(policy_file_name,
           agent1=btil_models.list_np_policy[0],
           agent2=btil_models.list_np_policy[1])

  if not use_true_tx:
    tx_file_name = tbp.SAVE_PREFIX + "_btil_tx_"
    tx_file_name += "synth_" if synthetic else "human_"
    tx_file_name += "%d_%.2f" % (num_train, supervision)
    tx_file_name = os.path.join(save_dir, tx_file_name)
    np.savez(tx_file_name,
             agent1=btil_models.list_Tx[0].np_Tx,
             agent2=btil_models.list_Tx[1].np_Tx)


if __name__ == "__main__":
  logging.basicConfig(
      level=logging.INFO,
      format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
      handlers=[logging.StreamHandler()],
      force=True)
  main()
