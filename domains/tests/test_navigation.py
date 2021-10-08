import glob
import os
import pickle
import numpy as np
import ai_coach_core.model_inference.var_infer.var_infer_static_x as var_infer
from ai_coach_core.latent_inference.bayesian_inference import (
    bayesian_mind_inference)
from ai_coach_core.model_inference.IRL.maxent_irl import CMaxEntIRL

from ai_coach_domain.navigation.mdp import NavigationMDP, transition_navi
from ai_coach_domain.navigation.maps import NAVI_MAP
from ai_coach_domain.navigation.policy import (get_static_policy,
                                               get_static_action)
from ai_coach_domain.navigation.simulator import NavigationSimulator

DATA_DIR = os.path.join(os.path.dirname(__file__), "data/")
GAME_MAP = NAVI_MAP

MDP_AGENT = NavigationMDP(**GAME_MAP)  # MDP for agent policy
TEMPERATURE = 0.3
LIST_POLICY = get_static_policy(MDP_AGENT, TEMPERATURE)


def cal_policy_kl_error(mdp_agent: NavigationMDP, trajectories, latent_labels,
                        cb_pi_true, cb_pi_infer):
  def compute_relative_freq(num_latents, num_states, trajs):
    rel_freq_a1 = np.zeros((num_latents, num_states))
    rel_freq_a2 = np.zeros((num_latents, num_states))
    count = 0
    for traj in trajs:
      for idx, (s, joint_a) in enumerate(traj):
        rel_freq_a1[latent_labels[idx][0], s] += 1
        rel_freq_a2[latent_labels[idx][1], s] += 1
        count += 1

    rel_freq_a1 = rel_freq_a1 / count
    rel_freq_a2 = rel_freq_a2 / count

    return rel_freq_a1, rel_freq_a2

  def compute_kl(agent_idx, x_idx, s_idx):
    np_p_inf = cb_pi_infer(agent_idx, x_idx, s_idx)
    np_p_true = cb_pi_true(agent_idx, x_idx, s_idx)

    sum_val = 0
    for idx in range(len(np_p_inf)):
      if np_p_inf[idx] != 0 and np_p_true[idx] != 0:
        sum_val += np_p_inf[idx] * (np.log(np_p_inf[idx]) -
                                    np.log(np_p_true[idx]))
    return sum_val

  rel_freq_a1, rel_freq_a2 = compute_relative_freq(mdp_agent.num_latents,
                                                   mdp_agent.num_states,
                                                   trajectories)

  sum_kl1 = 0
  sum_kl2 = 0
  for xidx in range(mdp_agent.num_latents):
    for sidx in range(mdp_agent.num_states):
      if rel_freq_a1[xidx, sidx] > 0:
        sum_kl1 += rel_freq_a1[xidx, sidx] * compute_kl(0, xidx, sidx)

      if rel_freq_a2[xidx, sidx] > 0:
        sum_kl2 += rel_freq_a2[xidx, sidx] * compute_kl(1, xidx, sidx)

  return sum_kl1, sum_kl2


def get_bayesian_infer_result(num_agent, cb_n_xsa_policy, num_lstate,
                              test_full_trajectories, true_latent_labels):

  full_conf = {}
  for idx1 in range(4):
    for idx2 in range(4):
      full_conf[(idx1, idx2)] = {}
      for idxa in range(4):
        for idxb in range(4):
          full_conf[(idx1, idx2)][(idxa, idxb)] = 0

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
  ordered_key = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3),
                 (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]
  count_all = 0
  sum_corrent = 0
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


#  (MDP_AGENT.a1_a_space.num_actions,
#                         MDP_AGENT.a2_a_space.num_actions)
def policy_true_dist(agent_idx, x_idx, state_idx):
  return np.sum(LIST_POLICY[x_idx][state_idx].reshape(
      (MDP_AGENT.a1_a_space.num_actions, MDP_AGENT.a2_a_space.num_actions)),
                axis=(1 - agent_idx))


if __name__ == "__main__":
  # set simulator
  #############################################################################
  sim = NavigationSimulator(0, transition_navi)
  sim.init_game(**GAME_MAP)
  sim.max_steps = 200

  def get_a1_action(**kwargs):
    return get_static_action(MDP_AGENT, NavigationSimulator.AGENT1, TEMPERATURE,
                             **kwargs)

  def get_a2_action(**kwargs):
    return get_static_action(MDP_AGENT, NavigationSimulator.AGENT2, TEMPERATURE,
                             **kwargs)

  def get_init_x(box_states, a1_pos, a2_pos):
    a1 = "GH1" if np.random.randint(2) == 0 else "GH2"
    a2 = 1 if np.random.randint(2) == 0 else 2
    a3 = "GH1" if np.random.randint(2) == 0 else "GH2"
    a4 = 1 if np.random.randint(2) == 0 else 2
    return (a1, a2), (a3, a4)

  sim.set_autonomous_agent(cb_get_A1_action=get_a1_action,
                           cb_get_A2_action=get_a2_action,
                           cb_get_A1_mental_state=None,
                           cb_get_A2_mental_state=None,
                           cb_get_init_mental_state=get_init_x)

  # generate data
  #############################################################################
  GEN_TRAIN_SET = False
  GEN_TEST_SET = False

  SHOW_TRUE = False
  SHOW_SL_SMALL = False
  SHOW_SL_LARGE = False
  SHOW_SEMI = False
  BASE_LINE = True

  VI_TRAIN = SHOW_TRUE or SHOW_SL_SMALL or SHOW_SL_LARGE or SHOW_SEMI or BASE_LINE

  TRAIN_DIR = os.path.join(DATA_DIR, 'navigation_train')
  TEST_DIR = os.path.join(DATA_DIR, 'navigation_test')

  train_prefix = "train_"
  test_prefix = "test_"
  if GEN_TRAIN_SET:
    file_names = glob.glob(os.path.join(TRAIN_DIR, train_prefix + '*.txt'))
    for fmn in file_names:
      os.remove(fmn)

    sim.run_simulation(1000, os.path.join(TRAIN_DIR, train_prefix), "header")

  if GEN_TEST_SET:
    file_names = glob.glob(os.path.join(TEST_DIR, test_prefix + '*.txt'))
    for fmn in file_names:
      os.remove(fmn)

    sim.run_simulation(300, os.path.join(TEST_DIR, test_prefix), "header")

  # train variational inference
  #############################################################################
  if VI_TRAIN:

    # load train set
    ##################################################
    trajectories = []
    list_trajectories_same_x = [[], [], [], []]
    latent_labels = []
    file_names = glob.glob(os.path.join(TRAIN_DIR, train_prefix + '*.txt'))
    for idx, file_nm in enumerate(file_names):
      trj = NavigationSimulator.read_file(file_nm)
      traj = []
      for bstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat in trj:
        a1s = MDP_AGENT.a1_pos_space.state_to_idx[a1pos]
        a2s = MDP_AGENT.a2_pos_space.state_to_idx[a2pos]
        sidx = MDP_AGENT.conv_state_to_idx((a1s, a2s))

        aidx1 = MDP_AGENT.a1_a_space.action_to_idx[a1act]
        aidx2 = MDP_AGENT.a2_a_space.action_to_idx[a2act]

        traj.append((sidx, (aidx1, aidx2)))

        xidx1 = MDP_AGENT.latent_space.state_to_idx[a1lat]
        xidx2 = MDP_AGENT.latent_space.state_to_idx[a2lat]

      trajectories.append(traj)
      latent_labels.append((xidx1, xidx2))
      if xidx1 == 0 and xidx2 == 0:
        list_trajectories_same_x[0].append(traj)
      if xidx1 == 1 and xidx2 == 1:
        list_trajectories_same_x[1].append(traj)
      if xidx1 == 2 and xidx2 == 2:
        list_trajectories_same_x[2].append(traj)
      if xidx1 == 3 and xidx2 == 3:
        list_trajectories_same_x[3].append(traj)

    print(len(trajectories))
    print(len(latent_labels))

    # partial trajectory? for each traj, traj[sidx:eidx]

    # load test set
    ##################################################
    test_file_names = glob.glob(os.path.join(TEST_DIR, test_prefix + '*.txt'))
    # test_file_names = [test_file_names[0]]
    test_traj = []
    test_labels = []
    for idx, file_nm in enumerate(test_file_names):
      trj = NavigationSimulator.read_file(file_nm)
      traj = []
      for bstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat in trj:
        a1s = MDP_AGENT.a1_pos_space.state_to_idx[a1pos]
        a2s = MDP_AGENT.a2_pos_space.state_to_idx[a2pos]
        sidx = MDP_AGENT.conv_state_to_idx((a1s, a2s))

        aidx1 = MDP_AGENT.a1_a_space.action_to_idx[a1act]
        aidx2 = MDP_AGENT.a2_a_space.action_to_idx[a2act]
        traj.append((sidx, (aidx1, aidx2)))

        xidx1 = MDP_AGENT.latent_space.state_to_idx[a1lat]
        xidx2 = MDP_AGENT.latent_space.state_to_idx[a2lat]

      test_traj.append(traj)
      test_labels.append((xidx1, xidx2))
    print(len(test_traj))

    BETA_PI = 1.1
    joint_num_action = (MDP_AGENT.a1_a_space.num_actions,
                        MDP_AGENT.a2_a_space.num_actions)

    idx_small = int(len(trajectories) / 5)
    print(idx_small)
    num_agents = sim.get_num_agents()

    # train base line
    ###########################################################################
    if BASE_LINE:

      def feature_extract_full_state(mdp, s_idx, a_idx):
        np_feature = np.zeros(mdp.num_states)
        np_feature[s_idx] = 100
        return np_feature

      init_prop = np.zeros((MDP_AGENT.num_states))

      a1s = MDP_AGENT.a1_pos_space.state_to_idx[GAME_MAP["a1_init"]]
      a2s = MDP_AGENT.a2_pos_space.state_to_idx[GAME_MAP["a2_init"]]
      sid = MDP_AGENT.conv_state_to_idx((a1s, a2s))
      init_prop[sid] = 1

      list_pi_est = []
      list_w_est = []
      for idx in range(4):
        print(len(list_trajectories_same_x[idx]))
        irl_x1 = CMaxEntIRL(list_trajectories_same_x[idx],
                            MDP_AGENT,
                            feature_extractor=feature_extract_full_state,
                            max_value_iter=500,
                            initial_prop=init_prop)

        irl_x1.do_inverseRL(epsilon=0.001, n_max_run=1)
        list_pi_est.append(irl_x1.pi_est)
        list_w_est.append(irl_x1.weights)

        save_name = os.path.join(DATA_DIR, 'nv_irl_weight_%d.pickle' % (idx, ))
        with open(save_name, 'wb') as f:
          pickle.dump(irl_x1.weights, f, pickle.HIGHEST_PROTOCOL)
        save_name2 = os.path.join(DATA_DIR, 'nv_irl_pi_%d.pickle' % (idx, ))
        with open(save_name2, 'wb') as f:
          pickle.dump(irl_x1.pi_est, f, pickle.HIGHEST_PROTOCOL)

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

      def irl_pol(ag, x, s, a):
        return irl_np_policy[ag][x, s, a]

      sup_conf_true, full_acc_true = (get_bayesian_infer_result(
          num_agents, irl_pol, MDP_AGENT.num_latents, test_traj, test_labels))
      # policy: joint

      print("Full - IRL")
      print_conf(sup_conf_true)
      print("16by16 Acc: " + str(full_acc_true))

    if SHOW_TRUE:

      def policy_true(agent_idx, x_idx, state_idx, joint_action):
        return np.sum(LIST_POLICY[x_idx][state_idx].reshape(joint_num_action),
                      axis=(1 - agent_idx))[joint_action[agent_idx]]

      sup_conf_true, full_acc_true = get_bayesian_infer_result(
          num_agents, policy_true, MDP_AGENT.num_latents, test_traj,
          test_labels)

      print("Full - True")
      print_conf(sup_conf_true)
      print("16by16 Acc: " + str(full_acc_true))

      # def inferred_policy_dist(m, x, s):
      #   return var_infer_small.list_np_policy[m][x, s]

      # kl1, kl2 = cal_policy_kl_error(MDP_AGENT, trajectories, policy_true_dist,
      #                                inferred_policy_dist)
      # print("kl1: %f, kl2: %f" % (kl1, kl2))

    ##############################################
    # supervised policy learning
    if SHOW_SL_SMALL:
      var_infer_small = var_infer.VarInferStaticX_SL(
          trajectories[0:idx_small], latent_labels[0:idx_small], num_agents,
          MDP_AGENT.num_states, MDP_AGENT.num_latents, joint_num_action)

      var_infer_small.set_dirichlet_prior(BETA_PI)
      var_infer_small.do_inference()

      sup_conf_full1, full_acc1 = get_bayesian_infer_result(
          num_agents, (lambda m, x, s, joint: var_infer_small.list_np_policy[m][
              x, s, joint[m]]), MDP_AGENT.num_latents, test_traj, test_labels)

      print("Full - SL_SMALL")
      print_conf(sup_conf_full1)
      print("16by16 Acc: " + str(full_acc1))

      def inferred_policy_dist(m, x, s):
        return var_infer_small.list_np_policy[m][x, s]

      kl1, kl2 = cal_policy_kl_error(MDP_AGENT, trajectories, latent_labels,
                                     policy_true_dist, inferred_policy_dist)

      print("kl1: %f, kl2: %f" % (kl1, kl2))

    if SHOW_SL_LARGE:
      var_infer_large = var_infer.VarInferStaticX_SL(trajectories,
                                                     latent_labels, num_agents,
                                                     MDP_AGENT.num_states,
                                                     MDP_AGENT.num_latents,
                                                     joint_num_action)
      var_infer_large.set_dirichlet_prior(BETA_PI)
      var_infer_large.do_inference()
      sup_conf_full2, full_acc2 = get_bayesian_infer_result(
          num_agents, (lambda m, x, s, joint: var_infer_large.list_np_policy[m][
              x, s, joint[m]]), MDP_AGENT.num_latents, test_traj, test_labels)

      print("Full - SL_LARGE")
      print_conf(sup_conf_full2)
      print("16by16 Acc: " + str(full_acc2))

      def inferred_policy_dist(m, x, s):
        return var_infer_large.list_np_policy[m][x, s]

      kl1, kl2 = cal_policy_kl_error(MDP_AGENT, trajectories, latent_labels,
                                     policy_true_dist, inferred_policy_dist)

      print("kl1: %f, kl2: %f" % (kl1, kl2))

    # ##############################################
    # # semisupervised policy learning
    if SHOW_SEMI:
      var_infer_semi = var_infer.VarInferStaticX_SemiSL(
          trajectories[idx_small:len(trajectories)],
          trajectories[0:idx_small],
          latent_labels[0:idx_small],
          num_agents,
          MDP_AGENT.num_states,
          MDP_AGENT.num_latents,
          joint_num_action,
          max_iteration=100,
          epsilon=0.001)

      var_infer_semi.set_dirichlet_prior(BETA_PI)

      var_infer_semi.do_inference()

      semi_conf_full, semi_full_acc = get_bayesian_infer_result(
          num_agents, (lambda m, x, s, joint: var_infer_semi.list_np_policy[m][
              x, s, joint[m]]), MDP_AGENT.num_latents, test_traj, test_labels)
      print("Full - SEMI")
      print_conf(semi_conf_full)
      print("16by16 Acc: " + str(semi_full_acc))

      def inferred_policy_dist(m, x, s):
        return var_infer_semi.list_np_policy[m][x, s]

      kl1, kl2 = cal_policy_kl_error(MDP_AGENT, trajectories, latent_labels,
                                     policy_true_dist, inferred_policy_dist)

      print("kl1: %f, kl2: %f" % (kl1, kl2))