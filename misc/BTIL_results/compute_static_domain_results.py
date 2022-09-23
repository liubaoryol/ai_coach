import glob
import os
import numpy as np
import pickle
import click
import logging

import ai_coach_core.model_learning.BTIL.btil_static as var_infer
from ai_coach_core.latent_inference.static_inference import (
    bayesian_mental_state_inference)
from ai_coach_core.model_learning.IRL.maxent_irl import MaxEntIRL
from aicoach_baselines.tabular_bc import tabular_behavior_cloning
from ai_coach_core.utils.data_utils import Trajectories
from ai_coach_core.utils.result_utils import cal_latent_policy_error
import aicoach_baselines.ikostrikov_gail as ikostrikov

from ai_coach_domain.box_push.maps import TUTORIAL_MAP
from ai_coach_domain.box_push.simulator import BoxPushSimulator_AloneOrTogether
from ai_coach_domain.box_push_static.mdp import StaticBoxPushMDP
from ai_coach_domain.box_push_static.agent import (StaticBoxPushPolicy,
                                                   StaticBoxPushAgent)
import matplotlib.pyplot as plt

BoxPushSimulator = BoxPushSimulator_AloneOrTogether
BoxPushPolicy = StaticBoxPushPolicy

DATA_DIR = os.path.join(os.path.dirname(__file__), "data/")
GAME_MAP = TUTORIAL_MAP

MDP_AGENT = StaticBoxPushMDP(**GAME_MAP)  # MDP for agent policy
TEMPERATURE = 1


def get_bayesian_infer_result(num_agent, cb_n_xsa_policy, num_lstate,
                              test_full_trajectories, true_latent_labels):

  full_conf = {}
  full_conf[(0, 0)] = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
  full_conf[(0, 1)] = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
  full_conf[(1, 0)] = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
  full_conf[(1, 1)] = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
  tuple_num_lstate = tuple(num_lstate for dummy_i in range(num_agent))

  full_count_correct = 0
  for idx, trj in enumerate(test_full_trajectories):
    infer_lat = bayesian_mental_state_inference(trj, tuple_num_lstate,
                                                cb_n_xsa_policy, num_agent)
    true_lat = true_latent_labels[idx]
    full_conf[true_lat][infer_lat] += 1
    if true_lat == infer_lat:
      full_count_correct += 1
  full_acc = full_count_correct / len(test_full_trajectories) * 100

  return (full_conf, full_acc)


def print_conf(conf):
  ordered_key = [(0, 0), (1, 1), (0, 1), (1, 0)]
  count_all = 0
  sum_corrent = 0
  str_conf = "\n\t;(0, 0)\t;(1, 1)\t;(0, 1)\t;(1, 0)\t\n"
  for key1 in ordered_key:
    # print(key1)
    txt_pred_value = str(key1)
    txt_pred_value = str(key1)
    for key2 in ordered_key:
      # txt_pred_key = txt_pred_key + str(key2) + "; "
      txt_pred_value = txt_pred_value + "\t; " + str(conf[key1][key2])
      count_all += conf[key1][key2]
      if key1 == key2:
        sum_corrent += conf[key1][key2]
    str_conf += txt_pred_value + "\n"

  return str_conf


class StaticBoxPushTrajectories(Trajectories):
  def __init__(self, num_latents: int) -> None:
    super().__init__(num_state_factors=1,
                     num_action_factors=2,
                     num_latent_factors=2,
                     tup_num_latents=(num_latents, num_latents))

  def load_from_files(self, file_names):
    for file_nm in file_names:
      trj = BoxPushSimulator.read_file(file_nm)
      if len(trj) == 0:
        continue

      np_trj = np.zeros((len(trj), self.get_width()), dtype=np.int32)
      for tidx, vec_state_action in enumerate(trj):
        bstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat = vec_state_action

        sidx = MDP_AGENT.conv_sim_states_to_mdp_sidx([bstt, a1pos, a2pos])
        aidx1 = (MDP_AGENT.a1_a_space.action_to_idx[a1act]
                 if a1act is not None else Trajectories.EPISODE_END)
        aidx2 = (MDP_AGENT.a2_a_space.action_to_idx[a2act]
                 if a2act is not None else Trajectories.EPISODE_END)

        xidx1 = (MDP_AGENT.latent_space.state_to_idx[a1lat]
                 if a1lat is not None else Trajectories.EPISODE_END)
        xidx2 = (MDP_AGENT.latent_space.state_to_idx[a2lat]
                 if a2lat is not None else Trajectories.EPISODE_END)

        np_trj[tidx, :] = [sidx, aidx1, aidx2, xidx1, xidx2]

      self.list_np_trajectory.append(np_trj)


# yapf: disable
@click.command()
@click.option("--gen_trainset", type=bool, default=False, help="generate train set")  # noqa: E501
@click.option("--gen_testset", type=bool, default=False, help="generate test set")  # noqa: E501
@click.option("--show_true", type=bool, default=False, help="metrics from true policy")  # noqa: E501
@click.option("--show_bc", type=bool, default=False, help="behavioral cloning results")  # noqa: E501
@click.option("--dnn_bc", type=bool, default=True, help="dnn behavioral cloning")  # noqa: E501
@click.option("--show_sl", type=bool, default=False, help="")
@click.option("--show_semi", type=bool, default=False, help="")
@click.option("--magail", type=bool, default=False, help="")
@click.option("--magail_latent", type=bool, default=False, help="")
@click.option("--num_processes", type=int, default=4, help="")
@click.option("--gail_batch_size", type=int, default=64, help="")
@click.option("--ppo_batch_size", type=int, default=32, help="")
@click.option("--num_iterations", type=int, default=300, help="")
@click.option("--pretrain_steps", type=int, default=100, help="")
@click.option("--use_ce", type=bool, default=False, help="")
@click.option("--num_run", type=int, default=1, help="")
# yapf: enable
def main(gen_trainset, gen_testset, show_true, show_bc, dnn_bc, show_sl,
         show_semi, magail, magail_latent, num_processes, gail_batch_size,
         ppo_batch_size, num_iterations, pretrain_steps, use_ce, num_run):
  logging.info("gail batch size: %d" % (gail_batch_size, ))
  logging.info("ppo batch size: %d" % (ppo_batch_size, ))
  logging.info("num iterations: %d" % (num_iterations, ))
  logging.info("num processes: %d" % (num_processes, ))
  logging.info("pretrain steps: %d" % (pretrain_steps, ))
  logging.info("use_ce: %s" % (use_ce, ))

  for dummy_run in range(num_run):
    logging.info("run count: %d" % (dummy_run, ))

    # set simulator
    ############################################################################
    sim = BoxPushSimulator(0)
    sim.init_game(**GAME_MAP)
    sim.max_steps = 200

    policy1 = BoxPushPolicy(MDP_AGENT, TEMPERATURE, BoxPushSimulator.AGENT1)
    policy2 = BoxPushPolicy(MDP_AGENT, TEMPERATURE, BoxPushSimulator.AGENT2)
    agent1 = StaticBoxPushAgent(policy1, BoxPushSimulator.AGENT1)
    agent2 = StaticBoxPushAgent(policy2, BoxPushSimulator.AGENT2)

    sim.set_autonomous_agent(agent1, agent2)

    def get_true_policy(agent_idx, latent_idx, state_idx):
      if agent_idx == 0:
        return agent1.policy_from_task_mdp_POV(state_idx, latent_idx)
      else:
        return agent2.policy_from_task_mdp_POV(state_idx, latent_idx)

    # generate data
    ############################################################################

    TRAIN_DIR = os.path.join(DATA_DIR, 'static_bp_train')
    TEST_DIR = os.path.join(DATA_DIR, 'static_bp_test')

    train_prefix = "train_"
    test_prefix = "test_"
    if gen_trainset:
      file_names = glob.glob(os.path.join(TRAIN_DIR, train_prefix + '*.txt'))
      for fmn in file_names:
        os.remove(fmn)

      sim.run_simulation(100, os.path.join(TRAIN_DIR, train_prefix), "header")

    if gen_testset:
      file_names = glob.glob(os.path.join(TEST_DIR, test_prefix + '*.txt'))
      for fmn in file_names:
        os.remove(fmn)

      sim.run_simulation(100, os.path.join(TEST_DIR, test_prefix), "header")

    # load train set
    ##################################################
    file_names = glob.glob(os.path.join(TRAIN_DIR, train_prefix + '*.txt'))

    train_data = StaticBoxPushTrajectories(MDP_AGENT.num_latents, sim)
    train_data.load_from_files(file_names)
    train_data.shuffle()

    trajectories, latent_labels = train_data.get_as_row_lists_for_static_x(
        include_terminal=False)
    sax_trajs = train_data.get_as_row_lists(no_latent_label=False,
                                            include_terminal=False)

    logging.info(len(trajectories))

    # partial trajectory? for each traj, traj[sidx:eidx]

    # load test set
    ##################################################
    test_file_names = glob.glob(os.path.join(TEST_DIR, test_prefix + '*.txt'))

    test_data = StaticBoxPushTrajectories(MDP_AGENT.num_latents, sim)
    test_data.load_from_files(test_file_names)

    test_traj, test_labels = test_data.get_as_row_lists_for_static_x(
        include_terminal=False)

    logging.info(len(test_traj))

    num_agents = sim.get_num_agents()

    list_idx = [int(len(trajectories) / 5), len(trajectories)]

    BETA_PI = 1.5
    joint_num_action = (MDP_AGENT.a1_a_space.num_actions,
                        MDP_AGENT.a2_a_space.num_actions)
    if show_bc:
      for idx in list_idx:
        logging.info("#########")
        logging.info("BC %d" % (idx, ))
        logging.info("#########")
        pi_a1 = np.zeros(
            (MDP_AGENT.num_latents, MDP_AGENT.num_states, joint_num_action[0]))
        pi_a2 = np.zeros(
            (MDP_AGENT.num_latents, MDP_AGENT.num_states, joint_num_action[1]))

        if dnn_bc:
          logging.info("BC by DNN")
          train_data.set_num_samples_to_use(idx)
          trajectories_bc = train_data.get_trajectories_fragmented_by_latent(
              include_next_state=False)

          for xidx in range(MDP_AGENT.num_latents):
            pi_a1[xidx] = ikostrikov.bc_dnn(MDP_AGENT.num_states,
                                            joint_num_action[0],
                                            trajectories_bc[0][xidx],
                                            demo_batch_size=gail_batch_size,
                                            ppo_batch_size=ppo_batch_size,
                                            bc_pretrain_steps=pretrain_steps)
            pi_a2[xidx] = ikostrikov.bc_dnn(MDP_AGENT.num_states,
                                            joint_num_action[1],
                                            trajectories_bc[1][xidx],
                                            demo_batch_size=gail_batch_size,
                                            ppo_batch_size=ppo_batch_size,
                                            bc_pretrain_steps=pretrain_steps)
        else:
          logging.info("Tabular BC")
          train_data.set_num_samples_to_use(idx)
          trajectories_bc = train_data.get_trajectories_fragmented_by_latent(
              include_next_state=False)

          for xidx in range(MDP_AGENT.num_latents):
            pi_a1[xidx] = tabular_behavior_cloning(trajectories_bc[0][xidx],
                                                   MDP_AGENT.num_states,
                                                   joint_num_action[0])
            pi_a2[xidx] = tabular_behavior_cloning(trajectories_bc[1][xidx],
                                                   MDP_AGENT.num_states,
                                                   joint_num_action[1])
        list_pi_bc = [pi_a1, pi_a2]
        # print(np.sum(pi_a1 != 1 / 6))

        sup_conf_full1, full_acc1 = get_bayesian_infer_result(
            num_agents, (lambda m, x, s, joint: list_pi_bc[m][x, s, joint[m]]),
            MDP_AGENT.num_latents, test_traj, test_labels)

        logging.info(print_conf(sup_conf_full1))
        logging.info("4by4(Full) Acc: " + str(full_acc1))

        policy_errors = cal_latent_policy_error(
            num_agents, MDP_AGENT.num_states, MDP_AGENT.num_latents, sax_trajs,
            lambda ai, x, s: get_true_policy(ai, x, s),
            lambda ai, x, s: list_pi_bc[ai][x, s, :])

        logging.info(policy_errors)

    # train base line
    ###########################################################################

    if show_true:
      logging.info("#########")
      logging.info("True")
      logging.info("#########")

      def policy_true(agent_idx, x_idx, state_idx, joint_action):
        return get_true_policy(agent_idx, x_idx,
                               state_idx)[joint_action[agent_idx]]

      sup_conf_true, full_acc_true = get_bayesian_infer_result(
          num_agents, policy_true, MDP_AGENT.num_latents, test_traj,
          test_labels)

      logging.info(print_conf(sup_conf_true))
      logging.info("4by4(Full) Acc: " + str(full_acc_true))

    ##############################################
    # supervised policy learning
    if show_sl:
      for idx in list_idx:
        logging.info("#########")
        logging.info("SL with %d" % (idx, ))
        logging.info("#########")
        var_inf_sl = var_infer.VarInferStaticX_SL(
            trajectories[0:idx], latent_labels[0:idx], num_agents,
            MDP_AGENT.num_states, MDP_AGENT.num_latents, joint_num_action)

        var_inf_sl.set_dirichlet_prior(BETA_PI)
        var_inf_sl.do_inference()

        sup_conf_full1, full_acc1 = get_bayesian_infer_result(
            num_agents,
            (lambda m, x, s, joint: var_inf_sl.list_np_policy[m][x, s, joint[m]]
             ), MDP_AGENT.num_latents, test_traj, test_labels)

        logging.info(print_conf(sup_conf_full1))
        logging.info("4by4(Full) Acc: " + str(full_acc1))

        policy_errors = cal_latent_policy_error(
            num_agents, MDP_AGENT.num_states, MDP_AGENT.num_latents, sax_trajs,
            lambda ai, x, s: get_true_policy(ai, x, s),
            lambda ai, x, s: var_inf_sl.list_np_policy[ai][x, s, :])

        logging.info(policy_errors)

    # ##############################################
    # # semisupervised policy learning
    full_acc_history = []

    def accuracy_history(num_agent, pi_hyper):
      list_np_policies = [None for dummy_i in range(num_agents)]
      # if np.isnan(pi_hyper[0].sum()):
      #   print("Nan acc_hist 1-1")
      for idx in range(num_agents):
        numerator = pi_hyper[idx] - 1
        action_sums = np.sum(numerator, axis=2)
        list_np_policies[idx] = numerator / action_sums[:, :, np.newaxis]
      conf_full, full_acc = get_bayesian_infer_result(
          num_agent,
          (lambda m, x, s, joint: list_np_policies[m][x, s, joint[m]]),
          MDP_AGENT.num_latents, test_traj, test_labels)
      # full_acc_history.append(full_acc)
      # part_acc_history.append(part_acc)
      full_acc_history.append(full_acc)

    if show_semi:
      for idx in list_idx[:-1]:
        logging.info("#########")
        logging.info("Semi %d" % (idx, ))
        logging.info("#########")
        # semi-supervised
        var_infer_semi = var_infer.VarInferStaticX_SemiSL(
            trajectories[idx:len(trajectories)],
            trajectories[0:idx],
            latent_labels[0:idx],
            num_agents,
            MDP_AGENT.num_states,
            MDP_AGENT.num_latents,
            joint_num_action,
            max_iteration=100,
            epsilon=0.001)

        var_infer_semi.set_dirichlet_prior(BETA_PI)

        var_infer_semi.do_inference(callback=accuracy_history)

        semi_conf_full, semi_full_acc = get_bayesian_infer_result(
            num_agents, (lambda m, x, s, joint: var_infer_semi.list_np_policy[m]
                         [x, s, joint[m]]), MDP_AGENT.num_latents, test_traj,
            test_labels)
        logging.info(print_conf(semi_conf_full))
        logging.info("4by4(Full) Acc: " + str(semi_full_acc))

        policy_errors = cal_latent_policy_error(
            num_agents, MDP_AGENT.num_states, MDP_AGENT.num_latents, sax_trajs,
            lambda ai, x, s: get_true_policy(ai, x, s),
            lambda ai, x, s: var_infer_semi.list_np_policy[ai][x, s, :])

        logging.info(policy_errors)

        # fig = plt.figure(figsize=(3, 3))
        # ax1 = fig.add_subplot(111)
        # ax1.grid(True)
        # ax1.plot(full_acc_history,
        #          '.-',
        #          label="SemiSL",
        #          clip_on=False,
        #          fillstyle='none')
        # plt.show()

    init_state = MDP_AGENT.conv_sim_states_to_mdp_sidx(
        [[0] * len(sim.box_states), sim.a1_init, sim.a2_init])

    if magail:
      from aicoach_baselines.magail import magail_w_ppo

      def magail_by_latent(trajectories, latent_labels, init_state):
        n_traj = len(trajectories)
        list_disc_loss = []
        list_value_loss = []
        list_action_loss = []
        list_entropy = []

        def get_loss_each_round(disc_loss, value_loss, action_loss, entropy):
          if disc_loss is not None:
            list_disc_loss.append(disc_loss)
          if value_loss is not None:
            list_value_loss.append(value_loss)
          if action_loss is not None:
            list_action_loss.append(action_loss)
          if entropy is not None:
            list_entropy.append(entropy)

        trajectories_by_latent = [[], [], [], []]
        X_00, X_01, X_10, X_11 = 0, 1, 2, 3

        for idx, latents in enumerate(latent_labels):
          xidx1, xidx2 = latents
          if xidx1 == 0 and xidx2 == 0:
            trajectories_by_latent[X_00].append(trajectories[idx])
          elif xidx1 == 0 and xidx2 == 1:
            trajectories_by_latent[X_01].append(trajectories[idx])
          elif xidx1 == 1 and xidx2 == 0:
            trajectories_by_latent[X_10].append(trajectories[idx])
          elif xidx1 == 1 and xidx2 == 1:
            trajectories_by_latent[X_11].append(trajectories[idx])
          else:
            raise ValueError

        num_00 = len(trajectories_by_latent[X_00])
        num_01 = len(trajectories_by_latent[X_01])
        num_10 = len(trajectories_by_latent[X_10])
        num_11 = len(trajectories_by_latent[X_11])

        ratio_a1_0 = num_00 / (num_00 + num_01)
        ratio_a1_1 = num_10 / (num_10 + num_11)
        ratio_a2_0 = num_00 / (num_00 + num_10)
        ratio_a2_1 = num_01 / (num_01 + num_11)

        pi_by_combination = []
        for idx, trajs in enumerate(trajectories_by_latent):
          list_disc_loss.clear()
          list_value_loss.clear()
          list_action_loss.clear()
          list_entropy.clear()
          list_policies = magail_w_ppo(MDP_AGENT, [init_state],
                                       trajs,
                                       num_processes=num_processes,
                                       demo_batch_size=gail_batch_size,
                                       ppo_batch_size=ppo_batch_size,
                                       num_iterations=num_iterations,
                                       do_pretrain=True,
                                       bc_pretrain_steps=pretrain_steps,
                                       use_ce=use_ce,
                                       callback_loss=get_loss_each_round)
          pi_by_combination.append(list_policies)

          f = plt.figure(figsize=(15, 5))
          ax0 = f.add_subplot(141)
          ax0.plot(list_disc_loss)
          ax0.set_ylabel('disc_loss')
          ax1 = f.add_subplot(142)
          ax1.plot(list_value_loss)
          ax1.set_ylabel('value_loss')
          ax2 = f.add_subplot(143)
          ax2.plot(list_action_loss)
          ax2.set_ylabel('action_loss')
          ax3 = f.add_subplot(144)
          ax3.plot(list_entropy)
          ax3.set_ylabel('entropy')
          # plt.show()
          plt.savefig('loss (%d) %d.png' % (n_traj, idx))

        pi_a1 = np.zeros(
            (MDP_AGENT.num_latents, MDP_AGENT.num_states, joint_num_action[0]))
        pi_a2 = np.zeros(
            (MDP_AGENT.num_latents, MDP_AGENT.num_states, joint_num_action[1]))

        pi_a1[0] = (pi_by_combination[X_00][0] * ratio_a1_0 +
                    pi_by_combination[X_01][0] * (1 - ratio_a1_0))
        pi_a1[1] = (pi_by_combination[X_10][0] * ratio_a1_1 +
                    pi_by_combination[X_11][0] * (1 - ratio_a1_1))
        pi_a2[0] = (pi_by_combination[X_00][1] * ratio_a2_0 +
                    pi_by_combination[X_10][1] * (1 - ratio_a2_0))
        pi_a2[1] = (pi_by_combination[X_01][1] * ratio_a2_1 +
                    pi_by_combination[X_11][1] * (1 - ratio_a2_1))

        list_pi_magail = [pi_a1, pi_a2]
        return list_pi_magail

      for idx in list_idx:
        logging.info("#########")
        logging.info("MAGAIL %d" % (idx, ))
        logging.info("#########")
        list_pi_magail = magail_by_latent(trajectories[:idx],
                                          latent_labels[:idx], init_state)

        sup_conf_full1, full_acc1 = get_bayesian_infer_result(
            num_agents,
            (lambda m, x, s, joint: list_pi_magail[m][x, s, joint[m]]),
            MDP_AGENT.num_latents, test_traj, test_labels)

        logging.info(print_conf(sup_conf_full1))
        logging.info("4by4(Full) Acc: " + str(full_acc1))

        policy_errors = cal_latent_policy_error(
            num_agents, MDP_AGENT.num_states, MDP_AGENT.num_latents, sax_trajs,
            lambda ai, x, s: get_true_policy(ai, x, s),
            lambda ai, x, s: list_pi_magail[ai][x, s, :])

        logging.info(policy_errors)

    if magail_latent:
      from aicoach_baselines.latent_magail import lmagail_w_ppo
      for idx in list_idx:
        logging.info("#########")
        logging.info("LatentMAGAIL %d" % (idx, ))
        logging.info("#########")

        list_disc_loss = []
        list_value_loss = []
        list_action_loss = []
        list_entropy = []

        def get_loss_each_round(disc_loss, value_loss, action_loss, entropy):
          if disc_loss is not None:
            list_disc_loss.append(disc_loss)
          if value_loss is not None:
            list_value_loss.append(value_loss)
          if action_loss is not None:
            list_action_loss.append(action_loss)
          if entropy is not None:
            list_entropy.append(entropy)

        list_pi_magail = lmagail_w_ppo(MDP_AGENT, [init_state],
                                       [agent1.agent_model, agent2.agent_model],
                                       sax_trajs[:idx],
                                       num_processes=num_processes,
                                       demo_batch_size=gail_batch_size,
                                       ppo_batch_size=ppo_batch_size,
                                       num_iterations=num_iterations,
                                       do_pretrain=True,
                                       bc_pretrain_steps=pretrain_steps,
                                       use_ce=use_ce,
                                       callback_loss=get_loss_each_round)

        sup_conf_full1, full_acc1 = get_bayesian_infer_result(
            num_agents,
            (lambda m, x, s, joint: list_pi_magail[m][x, s, joint[m]]),
            MDP_AGENT.num_latents, test_traj, test_labels)

        logging.info(print_conf(sup_conf_full1))
        logging.info("4by4(Full) Acc: " + str(full_acc1))

        policy_errors = cal_latent_policy_error(
            num_agents, MDP_AGENT.num_states, MDP_AGENT.num_latents, sax_trajs,
            lambda ai, x, s: get_true_policy(ai, x, s),
            lambda ai, x, s: list_pi_magail[ai][x, s, :])

        logging.info(policy_errors)

        f = plt.figure(figsize=(15, 5))
        ax0 = f.add_subplot(141)
        ax0.plot(list_disc_loss)
        ax0.set_ylabel('disc_loss')
        ax1 = f.add_subplot(142)
        ax1.plot(list_value_loss)
        ax1.set_ylabel('value_loss')
        ax2 = f.add_subplot(143)
        ax2.plot(list_action_loss)
        ax2.set_ylabel('action_loss')
        ax3 = f.add_subplot(144)
        ax3.plot(list_entropy)
        ax3.set_ylabel('entropy')
        # plt.show()
        plt.savefig('latent_magail loss (%d).png' % (idx, ))

    # ------------------ deprecated baseline
    IRL = False
    if IRL:
      trajectories_x1 = []
      trajectories_x2 = []
      for idx, latents in enumerate(latent_labels):
        xidx1, xidx2 = latents
        if xidx1 == 0 and xidx2 == 0:
          trajectories_x1.append(trajectories[idx])
        if xidx1 == 1 and xidx2 == 1:
          trajectories_x2.append(trajectories[idx])

      print(len(trajectories_x1))
      print(len(trajectories_x2))

      def feature_extract_full_state(mdp, s_idx, a_idx):
        np_feature = np.zeros(mdp.num_states)
        np_feature[s_idx] = 100
        return np_feature

      init_prop = np.zeros((MDP_AGENT.num_states))
      sid = MDP_AGENT.conv_sim_states_to_mdp_sidx([[0] * len(GAME_MAP["boxes"]),
                                                   GAME_MAP["a1_init"],
                                                   GAME_MAP["a2_init"]])
      init_prop[sid] = 1

      list_pi_est = []
      list_w_est = []
      irl_x1 = MaxEntIRL(trajectories_x1,
                         MDP_AGENT,
                         feature_extractor=feature_extract_full_state,
                         max_value_iter=100,
                         initial_prop=init_prop)
      print("Do irl1")
      irl_x1.do_inverseRL(epsilon=0.001, n_max_run=100)
      list_pi_est.append(irl_x1.pi_est)
      list_w_est.append(irl_x1.weights)

      save_name = os.path.join(DATA_DIR, 'static_box_irl_weight_1.pickle')
      with open(save_name, 'wb') as f:
        pickle.dump(irl_x1.weights, f, pickle.HIGHEST_PROTOCOL)
      save_name2 = os.path.join(DATA_DIR, 'static_box_irl_pi_1.pickle')
      with open(save_name2, 'wb') as f:
        pickle.dump(irl_x1.pi_est, f, pickle.HIGHEST_PROTOCOL)

      irl_x2 = MaxEntIRL(trajectories_x2,
                         MDP_AGENT,
                         feature_extractor=feature_extract_full_state,
                         max_value_iter=100,
                         initial_prop=init_prop)

      print("Do irl2")
      irl_x2.do_inverseRL(epsilon=0.001, n_max_run=100)
      list_pi_est.append(irl_x2.pi_est)
      list_w_est.append(irl_x2.weights)

      save_name = os.path.join(DATA_DIR, 'static_box_irl_weight_2.pickle')
      with open(save_name, 'wb') as f:
        pickle.dump(irl_x2.weights, f, pickle.HIGHEST_PROTOCOL)
      save_name2 = os.path.join(DATA_DIR, 'static_box_irl_pi_2.pickle')
      with open(save_name2, 'wb') as f:
        pickle.dump(irl_x2.pi_est, f, pickle.HIGHEST_PROTOCOL)

      print("joint policy")
      # joint policy to individual policy
      irl_np_policy = []
      for idx in range(2):
        irl_np_policy.append(
            np.zeros((MDP_AGENT.num_latents, MDP_AGENT.num_states,
                      joint_num_action[idx])))

      for x_idx in range(MDP_AGENT.num_latents):
        for a_idx in range(MDP_AGENT.num_actions):
          a_cn_i, a_sn_i = MDP_AGENT.np_idx_to_action[a_idx]
          irl_np_policy[0][x_idx, :, a_cn_i] += list_pi_est[x_idx][:, a_idx]
          irl_np_policy[1][x_idx, :, a_sn_i] += list_pi_est[x_idx][:, a_idx]

      save_name2 = os.path.join(DATA_DIR, 'static_box_irl_pi_list.pickle')
      with open(save_name2, 'wb') as f:
        pickle.dump(irl_np_policy, f, pickle.HIGHEST_PROTOCOL)

      def irl_pol(ag, x, s, joint_action):
        return irl_np_policy[ag][x, s, joint_action[ag]]

      sup_conf_true, full_acc_true = (get_bayesian_infer_result(
          num_agents, irl_pol, MDP_AGENT.num_latents, test_traj, test_labels))
      # policy: joint

      print("Full - IRL")
      print_conf(sup_conf_true)
      print("4by4(Full) Acc: " + str(full_acc_true))

      policy_errors = cal_latent_policy_error(
          num_agents, MDP_AGENT.num_states, MDP_AGENT.num_latents, sax_trajs,
          lambda ai, x, s: get_true_policy(ai, x, s),
          lambda ai, x, s: irl_np_policy[ai][x, s, :])

      print(policy_errors)
    IRL = False
    if IRL:
      trajectories_x1 = []
      trajectories_x2 = []
      for idx, latents in enumerate(latent_labels):
        xidx1, xidx2 = latents
        if xidx1 == 0 and xidx2 == 0:
          trajectories_x1.append(trajectories[idx])
        if xidx1 == 1 and xidx2 == 1:
          trajectories_x2.append(trajectories[idx])

      print(len(trajectories_x1))
      print(len(trajectories_x2))

      def feature_extract_full_state(mdp, s_idx, a_idx):
        np_feature = np.zeros(mdp.num_states)
        np_feature[s_idx] = 100
        return np_feature

      init_prop = np.zeros((MDP_AGENT.num_states))
      sid = MDP_AGENT.conv_sim_states_to_mdp_sidx([[0] * len(GAME_MAP["boxes"]),
                                                   GAME_MAP["a1_init"],
                                                   GAME_MAP["a2_init"]])
      init_prop[sid] = 1

      list_pi_est = []
      list_w_est = []
      irl_x1 = MaxEntIRL(trajectories_x1,
                         MDP_AGENT,
                         feature_extractor=feature_extract_full_state,
                         max_value_iter=100,
                         initial_prop=init_prop)
      print("Do irl1")
      irl_x1.do_inverseRL(epsilon=0.001, n_max_run=100)
      list_pi_est.append(irl_x1.pi_est)
      list_w_est.append(irl_x1.weights)

      save_name = os.path.join(DATA_DIR, 'static_box_irl_weight_1.pickle')
      with open(save_name, 'wb') as f:
        pickle.dump(irl_x1.weights, f, pickle.HIGHEST_PROTOCOL)
      save_name2 = os.path.join(DATA_DIR, 'static_box_irl_pi_1.pickle')
      with open(save_name2, 'wb') as f:
        pickle.dump(irl_x1.pi_est, f, pickle.HIGHEST_PROTOCOL)

      irl_x2 = MaxEntIRL(trajectories_x2,
                         MDP_AGENT,
                         feature_extractor=feature_extract_full_state,
                         max_value_iter=100,
                         initial_prop=init_prop)

      print("Do irl2")
      irl_x2.do_inverseRL(epsilon=0.001, n_max_run=100)
      list_pi_est.append(irl_x2.pi_est)
      list_w_est.append(irl_x2.weights)

      save_name = os.path.join(DATA_DIR, 'static_box_irl_weight_2.pickle')
      with open(save_name, 'wb') as f:
        pickle.dump(irl_x2.weights, f, pickle.HIGHEST_PROTOCOL)
      save_name2 = os.path.join(DATA_DIR, 'static_box_irl_pi_2.pickle')
      with open(save_name2, 'wb') as f:
        pickle.dump(irl_x2.pi_est, f, pickle.HIGHEST_PROTOCOL)

      print("joint policy")
      # joint policy to individual policy
      irl_np_policy = []
      for idx in range(2):
        irl_np_policy.append(
            np.zeros((MDP_AGENT.num_latents, MDP_AGENT.num_states,
                      joint_num_action[idx])))

      for x_idx in range(MDP_AGENT.num_latents):
        for a_idx in range(MDP_AGENT.num_actions):
          a_cn_i, a_sn_i = MDP_AGENT.np_idx_to_action[a_idx]
          irl_np_policy[0][x_idx, :, a_cn_i] += list_pi_est[x_idx][:, a_idx]
          irl_np_policy[1][x_idx, :, a_sn_i] += list_pi_est[x_idx][:, a_idx]

      save_name2 = os.path.join(DATA_DIR, 'static_box_irl_pi_list.pickle')
      with open(save_name2, 'wb') as f:
        pickle.dump(irl_np_policy, f, pickle.HIGHEST_PROTOCOL)

      def irl_pol(ag, x, s, joint_action):
        return irl_np_policy[ag][x, s, joint_action[ag]]

      sup_conf_true, full_acc_true = (get_bayesian_infer_result(
          num_agents, irl_pol, MDP_AGENT.num_latents, test_traj, test_labels))
      # policy: joint

      print("Full - IRL")
      print_conf(sup_conf_true)
      print("4by4(Full) Acc: " + str(full_acc_true))

      policy_errors = cal_latent_policy_error(
          num_agents, MDP_AGENT.num_states, MDP_AGENT.num_latents, sax_trajs,
          lambda ai, x, s: get_true_policy(ai, x, s),
          lambda ai, x, s: irl_np_policy[ai][x, s, :])

      print(policy_errors)


if __name__ == "__main__":
  logging.basicConfig(
      level=logging.INFO,
      format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
      handlers=[
          logging.FileHandler("box_push_static_results.log"),
          logging.StreamHandler()
      ],
      force=True)
  logging.info('box push static results')
  main()
