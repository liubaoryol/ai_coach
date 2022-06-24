import os
import glob
import click
import logging
import random
import numpy as np
from ai_coach_core.model_learning.BTIL.btil_for_two import BTILforTwo
from ai_coach_domain.box_push.utils import BoxPushTrajectories
from ai_coach_domain.box_push.agent_model import (
    assumed_initial_mental_distribution)
import compute_dynamic_data_results as tbp


# yapf: disable
@click.command()
@click.option("--is_team", type=bool, default=False, help="team / indv")
@click.option("--show_sl", type=bool, default=True, help="")
@click.option("--show_semi", type=bool, default=False, help="")
@click.option("--show_ul", type=bool, default=False, help="")
@click.option("--num_run", type=int, default=1, help="")
@click.option("--show_random", type=bool, default=False, help="")
@click.option("--suboptimality_stats", type=bool, default=False, help="")
# yapf: enable
def main(is_team, show_sl, show_semi, show_ul, num_run, show_random,
         suboptimality_stats):
  logging.info("is_TEAM: %s" % (is_team, ))

  for dummy_run in range(num_run):
    logging.info("run count: %d" % (dummy_run, ))
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
    sim.max_steps = 200
    TEMPERATURE = 0.3

    if is_team:
      tbp.MDP_AGENT = tbp.bp_mdp.BoxPushTeamMDP_AlwaysTogether(**GAME_MAP)
      tbp.MDP_TASK = tbp.MDP_AGENT
      policy2 = BoxPushPolicyTeam(tbp.MDP_TASK, TEMPERATURE,
                                  tbp.BoxPushSimulator.AGENT2)
      agent2 = tbp.bp_agent.BoxPushAIAgent_Team2(policy2)
    else:
      tbp.MDP_AGENT = tbp.bp_mdp.BoxPushAgentMDP_AlwaysAlone(**GAME_MAP)
      tbp.MDP_TASK = tbp.bp_mdp.BoxPushTeamMDP_AlwaysAlone(**GAME_MAP)
      policy2 = BoxPushPolicyIndv(tbp.MDP_TASK,
                                  tbp.MDP_AGENT,
                                  temperature=TEMPERATURE,
                                  agent_idx=tbp.BoxPushSimulator.AGENT2)
      agent2 = tbp.bp_agent.BoxPushAIAgent_Indv2(policy2)

    def get_init_latent_dist(agent_idx, state_idx):
      if agent_idx == 0:
        return assumed_initial_mental_distribution(0, state_idx, tbp.MDP_TASK)
      else:
        return agent2.init_latent_dist_from_task_mdp_POV(state_idx)

    AWS_DIR = os.path.join(os.path.dirname(__file__), "aws_data_test/")

    if is_team:
      TRAIN_DIR = os.path.join(AWS_DIR, 'domain1')
    else:
      TRAIN_DIR = os.path.join(AWS_DIR, 'domain2')

    file_names = glob.glob(os.path.join(TRAIN_DIR, '*.txt'))
    random.shuffle(file_names)

    num_train = int(len(file_names) * 2 / 3)
    if suboptimality_stats:
      num_train = len(file_names)
    logging.info(num_train)
    train_files = file_names[:num_train]
    test_files = file_names[num_train:]

    # load train set
    ##################################################
    train_data = BoxPushTrajectories(sim, tbp.MDP_TASK, tbp.MDP_AGENT)
    train_data.load_from_files(train_files)
    traj_labeled_ver = train_data.get_as_row_lists(no_latent_label=False,
                                                   include_terminal=False)
    traj_unlabel_ver = train_data.get_as_row_lists(no_latent_label=True,
                                                   include_terminal=False)

    logging.info(len(traj_labeled_ver))

    # load test set
    ##################################################
    test_data = BoxPushTrajectories(sim, tbp.MDP_TASK, tbp.MDP_AGENT)
    test_data.load_from_files(test_files)
    test_traj = test_data.get_as_column_lists(include_terminal=False)
    logging.info(len(test_traj))

    if suboptimality_stats:
      print(tbp.get_suboptimal_stats(traj_labeled_ver, is_team))

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

    list_idx = [int(num_train * 0.2), int(num_train * 0.5), num_train]
    # print(list_idx)

    if show_random:
      logging.info("#########")
      logging.info("Random")
      logging.info("#########")

      def get_uniform_policy(agent_idx, latent_idx, state_idx):
        # return self.agent2.policy_from_task_mdp_POV(state_idx, latent_idx)
        return (np.ones(joint_action_num[agent_idx]) /
                joint_action_num[agent_idx])

      def get_uniform_tx(nidx, xidx, sidx, tuple_aidx, sidx_n):
        return (np.ones(tbp.MDP_AGENT.num_latents) / tbp.MDP_AGENT.num_latents)

      np_results = tbp.get_result(get_uniform_policy, get_uniform_tx,
                                  get_init_latent_dist, test_traj)
      avg1, avg2, avg3 = np.mean(np_results, axis=0)
      std1, std2, std3 = np.std(np_results, axis=0)
      logging.info("%f,%f,%f,%f,%f,%f" % (avg1, std1, avg2, std2, avg3, std3))

    # supervised variational inference
    if show_sl:
      for idx in list_idx:
        logging.info("#########")
        logging.info("SL with %d" % (idx, ))
        logging.info("#########")
        var_inf_sl = BTILforTwo(traj_labeled_ver[0:idx],
                                tbp.MDP_TASK.num_states,
                                tbp.MDP_AGENT.num_latents,
                                joint_action_num,
                                tbp.transition_s,
                                trans_x_dependency=(True, True, True, False))
        var_inf_sl.set_dirichlet_prior(BETA_PI, BETA_TX1, BETA_TX2)

        logging.info("Train without true Tx")
        var_inf_sl.set_bx_and_Tx(cb_bx=get_init_latent_dist)

        var_inf_sl.do_inference()

        var_inf_sl_conv = tbp.BTILConverter(var_inf_sl)
        np_results = tbp.get_result(var_inf_sl_conv.policy_nxs,
                                    var_inf_sl_conv.Tx_nxsas,
                                    get_init_latent_dist, test_traj)
        avg1, avg2, avg3 = np.mean(np_results, axis=0)
        std1, std2, std3 = np.std(np_results, axis=0)
        logging.info("%f,%f,%f,%f,%f,%f" % (avg1, std1, avg2, std2, avg3, std3))

    # semi-supervised
    if show_semi:
      for idx in list_idx[:-1]:
        logging.info("#########")
        logging.info("Semi %d" % (idx, ))
        logging.info("#########")
        # semi-supervised
        var_inf_semi = BTILforTwo(traj_labeled_ver[0:idx] +
                                  traj_unlabel_ver[idx:],
                                  tbp.MDP_TASK.num_states,
                                  tbp.MDP_AGENT.num_latents,
                                  joint_action_num,
                                  tbp.transition_s,
                                  trans_x_dependency=(True, True, True, False),
                                  epsilon=0.01,
                                  max_iteration=100)
        var_inf_semi.set_dirichlet_prior(BETA_PI, BETA_TX1, BETA_TX2)

        logging.info("Train without true Tx")
        var_inf_semi.set_bx_and_Tx(cb_bx=get_init_latent_dist)
        var_inf_semi.do_inference()

        var_inf_semi_conv = tbp.BTILConverter(var_inf_semi)
        np_results = tbp.get_result(var_inf_semi_conv.policy_nxs,
                                    var_inf_semi_conv.Tx_nxsas,
                                    get_init_latent_dist, test_traj)
        avg1, avg2, avg3 = np.mean(np_results, axis=0)
        std1, std2, std3 = np.std(np_results, axis=0)
        logging.info("Prediction of latent with learned Tx")
        logging.info("%f,%f,%f,%f,%f,%f" % (avg1, std1, avg2, std2, avg3, std3))


if __name__ == "__main__":
  import time
  file_prefix = "box_push_aws_results"
  sec, msec = divmod(time.time() * 1000, 1000)
  time_stamp = '-%s' % (time.strftime('%Y%m%d_%H%M%S', time.gmtime(sec)), )
  file_name = (file_prefix + time_stamp + '.log')

  logging.basicConfig(
      level=logging.INFO,
      format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
      handlers=[logging.FileHandler(file_name),
                logging.StreamHandler()],
      force=True)
  logging.info('box push aws results')
  main()
