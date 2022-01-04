import glob
import os
import numpy as np
import pickle

import ai_coach_core.model_inference.var_infer.var_infer_static_x as var_infer
from ai_coach_core.latent_inference.bayesian_inference import (
    bayesian_mind_inference)
from ai_coach_core.model_inference.IRL.maxent_irl import CMaxEntIRL
from ai_coach_core.model_inference.behavior_cloning import behavior_cloning
from ai_coach_core.utils.data_utils import Trajectories
from ai_coach_core.utils.result_utils import cal_policy_kl_error
from ai_coach_core.model_inference.sb3_algorithms import behavior_cloning_sb3

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
    infer_lat = bayesian_mind_inference(trj, tuple_num_lstate, cb_n_xsa_policy,
                                        num_agent)
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
  print("\t;(0, 0)\t;(1, 1)\t;(0, 1)\t;(1, 0)\t")
  for key1 in ordered_key:
    # print(key1)
    txt_pred_value = str(key1)
    for key2 in ordered_key:
      # txt_pred_key = txt_pred_key + str(key2) + "; "
      txt_pred_value = txt_pred_value + "\t; " + str(conf[key1][key2])
      count_all += conf[key1][key2]
      if key1 == key2:
        sum_corrent += conf[key1][key2]
    print(txt_pred_value)


class StaticBoxPushTrajectories(Trajectories):
  def __init__(self, num_latents) -> None:
    super().__init__(num_state_factors=1,
                     num_action_factors=2,
                     num_latent_factors=2,
                     num_latents=num_latents)

  def load_from_files(self, file_names):
    for file_nm in file_names:
      trj = BoxPushSimulator.read_file(file_nm)
      if len(trj) == 0:
        continue

      np_trj = np.zeros((len(trj), self.get_width()), dtype=np.int32)
      for tidx, vec_state_action in enumerate(trj):
        bstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat = vec_state_action

        sidx = MDP_AGENT.conv_sim_states_to_mdp_sidx(a1pos, a2pos, bstt)
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


if __name__ == "__main__":
  # set simulator
  #############################################################################
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
  #############################################################################
  GEN_TRAIN_SET = False
  GEN_TEST_SET = False

  SHOW_TRUE = True
  SHOW_SL = False
  SHOW_SEMI = True
  IRL = False
  BC = True
  VI_TRAIN = (SHOW_TRUE or SHOW_SL or SHOW_SEMI or IRL or BC)

  TRAIN_DIR = os.path.join(DATA_DIR, 'static_bp_train')
  TEST_DIR = os.path.join(DATA_DIR, 'static_bp_test')

  train_prefix = "train_"
  test_prefix = "test_"
  if GEN_TRAIN_SET:
    file_names = glob.glob(os.path.join(TRAIN_DIR, train_prefix + '*.txt'))
    for fmn in file_names:
      os.remove(fmn)

    sim.run_simulation(100, os.path.join(TRAIN_DIR, train_prefix), "header")

  if GEN_TEST_SET:
    file_names = glob.glob(os.path.join(TEST_DIR, test_prefix + '*.txt'))
    for fmn in file_names:
      os.remove(fmn)

    sim.run_simulation(100, os.path.join(TEST_DIR, test_prefix), "header")

  # train variational inference
  #############################################################################
  if VI_TRAIN:

    # load train set
    ##################################################
    file_names = glob.glob(os.path.join(TRAIN_DIR, train_prefix + '*.txt'))

    train_data = StaticBoxPushTrajectories(MDP_AGENT.num_latents)
    train_data.load_from_files(file_names)
    trajectories, latent_labels = train_data.get_as_row_lists_for_static_x(
        include_terminal=False)
    sax_trajs = train_data.get_as_row_lists(no_latent_label=False,
                                            include_terminal=False)

    trajectories_x1 = []
    trajectories_x2 = []
    for idx, latents in enumerate(latent_labels):
      xidx1, xidx2 = latents
      if xidx1 == 0 and xidx2 == 0:
        trajectories_x1.append(trajectories[idx])
      if xidx1 == 1 and xidx2 == 1:
        trajectories_x2.append(trajectories[idx])

    print(len(trajectories))
    print(len(trajectories_x1))
    print(len(trajectories_x2))
    print(len(latent_labels))

    # partial trajectory? for each traj, traj[sidx:eidx]

    # load test set
    ##################################################
    test_file_names = glob.glob(os.path.join(TEST_DIR, test_prefix + '*.txt'))

    test_data = StaticBoxPushTrajectories(MDP_AGENT.num_latents)
    test_data.load_from_files(test_file_names)

    test_traj, test_labels = test_data.get_as_row_lists_for_static_x(
        include_terminal=False)

    print(len(test_traj))

    num_agents = sim.get_num_agents()

    list_idx = [int(len(trajectories) / 5), len(trajectories)]
    print(list_idx)

    BETA_PI = 1.5
    joint_num_action = (MDP_AGENT.a1_a_space.num_actions,
                        MDP_AGENT.a2_a_space.num_actions)
    if BC:
      for idx in list_idx:
        print("#########")
        print("BC %d" % (idx, ))
        print("#########")
        pi_a1 = np.zeros(
            (MDP_AGENT.num_latents, MDP_AGENT.num_states, joint_num_action[0]))
        pi_a2 = np.zeros(
            (MDP_AGENT.num_latents, MDP_AGENT.num_states, joint_num_action[1]))

        TEST_SB3_BC = True
        if TEST_SB3_BC:
          train_data.set_num_samples_to_use(idx)
          trajectories_bc = train_data.get_trajectories_fragmented_by_latent(
              include_next_state=True)

          for xidx in range(MDP_AGENT.num_latents):
            pi_a1[xidx] = behavior_cloning_sb3(trajectories_bc[0][xidx],
                                               MDP_AGENT.num_states,
                                               joint_num_action[0])
            pi_a2[xidx] = behavior_cloning_sb3(trajectories_bc[1][xidx],
                                               MDP_AGENT.num_states,
                                               joint_num_action[1])
        else:
          train_data.set_num_samples_to_use(idx)
          trajectories_bc = train_data.get_trajectories_fragmented_by_latent(
              include_next_state=False)

          for xidx in range(MDP_AGENT.num_latents):
            pi_a1[xidx] = behavior_cloning(trajectories_bc[0][xidx],
                                           MDP_AGENT.num_states,
                                           joint_num_action[0])
            pi_a2[xidx] = behavior_cloning(trajectories_bc[1][xidx],
                                           MDP_AGENT.num_states,
                                           joint_num_action[1])
        list_pi_bc = [pi_a1, pi_a2]
        # print(np.sum(pi_a1 != 1 / 6))

        sup_conf_full1, full_acc1 = get_bayesian_infer_result(
            num_agents, (lambda m, x, s, joint: list_pi_bc[m][x, s, joint[m]]),
            MDP_AGENT.num_latents, test_traj, test_labels)

        print_conf(sup_conf_full1)
        print("4by4(Full) Acc: " + str(full_acc1))

        kl1, kl2 = cal_policy_kl_error(
            num_agents, MDP_AGENT.num_states, MDP_AGENT.num_latents, sax_trajs,
            lambda ai, x, s: get_true_policy(ai, x, s),
            lambda ai, x, s: list_pi_bc[ai][x, s, :])

        print("kl1, kl2: %f,%f" % (kl1, kl2))

    # train base line
    ###########################################################################
    if IRL:

      def feature_extract_full_state(mdp, s_idx, a_idx):
        np_feature = np.zeros(mdp.num_states)
        np_feature[s_idx] = 100
        return np_feature

      init_prop = np.zeros((MDP_AGENT.num_states))
      sid = MDP_AGENT.conv_sim_states_to_mdp_sidx(GAME_MAP["a1_init"],
                                                  GAME_MAP["a2_init"],
                                                  [0] * len(GAME_MAP["boxes"]))
      init_prop[sid] = 1

      list_pi_est = []
      list_w_est = []
      irl_x1 = CMaxEntIRL(trajectories_x1,
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

      irl_x2 = CMaxEntIRL(trajectories_x2,
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
      print("4by4 Acc: " + str(full_acc_true))

      for ai in range(2):
        rel_freq = compute_relative_freq(MDP_AGENT.num_states,
                                         MDP_AGENT.num_latents, trajectories,
                                         latent_labels, ai)
        kl1 = cal_policy_error(rel_freq, MDP_AGENT,
                               lambda x, s: irl_np_policy[ai][x, s, :],
                               lambda x, s: get_true_policy(ai, x, s))
        print("agent %d: KL %f" % (ai, kl1))

    if SHOW_TRUE:
      print("#########")
      print("True")
      print("#########")

      def policy_true(agent_idx, x_idx, state_idx, joint_action):
        return get_true_policy(agent_idx, x_idx,
                               state_idx)[joint_action[agent_idx]]

      sup_conf_true, full_acc_true = get_bayesian_infer_result(
          num_agents, policy_true, MDP_AGENT.num_latents, test_traj,
          test_labels)

      print_conf(sup_conf_true)
      print("4by4(Full) Acc: " + str(full_acc_true))

    ##############################################
    # supervised policy learning
    if SHOW_SL:
      for idx in list_idx:
        print("#########")
        print("SL with %d" % (idx, ))
        print("#########")
        var_inf_sl = var_infer.VarInferStaticX_SL(
            trajectories[0:idx], latent_labels[0:idx], num_agents,
            MDP_AGENT.num_states, MDP_AGENT.num_latents, joint_num_action)

        var_inf_sl.set_dirichlet_prior(BETA_PI)
        var_inf_sl.do_inference()

        sup_conf_full1, full_acc1 = get_bayesian_infer_result(
            num_agents,
            (lambda m, x, s, joint: var_inf_sl.list_np_policy[m][x, s, joint[m]]
             ), MDP_AGENT.num_latents, test_traj, test_labels)

        print_conf(sup_conf_full1)
        print("4by4(Full) Acc: " + str(full_acc1))

        kl1, kl2 = cal_policy_kl_error(
            num_agents, MDP_AGENT.num_states, MDP_AGENT.num_latents, sax_trajs,
            lambda ai, x, s: get_true_policy(ai, x, s),
            lambda ai, x, s: var_inf_sl.list_np_policy[ai][x, s, :])

        print("kl1, kl2: %f,%f" % (kl1, kl2))

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

    if SHOW_SEMI:
      for idx in list_idx[:-1]:
        print("#########")
        print("Semi %d" % (idx, ))
        print("#########")
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
        print_conf(semi_conf_full)
        print("4by4(Full) Acc: " + str(semi_full_acc))

        kl1, kl2 = cal_policy_kl_error(
            num_agents, MDP_AGENT.num_states, MDP_AGENT.num_latents, sax_trajs,
            lambda ai, x, s: get_true_policy(ai, x, s),
            lambda ai, x, s: var_infer_semi.list_np_policy[ai][x, s, :])

        print("kl1, kl2: %f,%f" % (kl1, kl2))

        fig = plt.figure(figsize=(3, 3))
        # str_title = (
        #     "hyperparam: " + str(SEMISUPER_HYPERPARAM) +
        #     ", # labeled: " + str(len(trajectories)) +
        #     ", # unlabeled: " + str(len(unlabeled_traj)))
        # fig.suptitle(str_title)
        ax1 = fig.add_subplot(111)
        ax1.grid(True)
        ax1.plot(full_acc_history,
                 '.-',
                 label="SemiSL",
                 clip_on=False,
                 fillstyle='none')
        plt.show()
      # if SHOW_SL_SMALL:
      #   ax1.axhline(
      # y=full_align_acc1, color='r', linestyle='-', label="SL-Small")
      # if SHOW_SL_LARGE:
      #   ax1.axhline(
      # y=full_align_acc2, color='g', linestyle='-', label="SL-Large")
      # FONT_SIZE = 16
      # TITLE_FONT_SIZE = 12
      # LEGENT_FONT_SIZE = 12
      # ax1.set_ylabel("Accuracy (%)", fontsize=FONT_SIZE)
      # ax1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
      # # ax1.set_ylim([70, 100])
      # # ax1.set_xlim([0, 16])
      # ax1.set_title("Full Sequence", fontsize=TITLE_FONT_SIZE)