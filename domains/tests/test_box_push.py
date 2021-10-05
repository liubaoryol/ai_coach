import glob
import os
import random
import pickle

import ai_coach_domain.box_push.maps as bp_maps
import ai_coach_domain.box_push.simulator as bp_sim
import ai_coach_domain.box_push.mdp as bp_mdp
import ai_coach_domain.box_push.policy as bp_policy
import ai_coach_domain.box_push.transition_x as bp_tx
from ai_coach_core.model_inference.var_infer.var_infer_dynamic_x import (
    VarInferDuo)
from ai_coach_core.latent_inference.most_probable_sequence import (
    most_probable_sequence)
from ai_coach_core.utils.result_utils import (norm_hamming_distance,
                                              alignment_sequence)
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data/")
GAME_MAP = bp_maps.EXP1_MAP
IS_TEAM = False
if IS_TEAM:
  BoxPushSimulator = bp_sim.BoxPushSimulator_AlwaysTogether
  BoxPushAgentMDP = bp_mdp.BoxPushTeamMDP_AlwaysTogether
  BoxPushTaskMDP = bp_mdp.BoxPushTeamMDP_AlwaysTogether
  get_box_push_action = bp_policy.get_exp1_action
  get_box_push_Tx = bp_tx.get_Tx_team
  get_np_Tx = bp_tx.get_np_Tx_team
  SAVED_TRANSITION = "mdp_exp1"
else:
  BoxPushSimulator = bp_sim.BoxPushSimulator_AlwaysAlone
  BoxPushAgentMDP = bp_mdp.BoxPushAgentMDP_AlwaysAlone
  BoxPushTaskMDP = bp_mdp.BoxPushTeamMDP_AlwaysAlone
  get_box_push_action = bp_policy.get_indv_action
  get_box_push_Tx = bp_tx.get_Tx_indv
  get_np_Tx = bp_tx.get_np_Tx_indv
  SAVED_TRANSITION = "mdp_indv"

MDP_AGENT = BoxPushAgentMDP(**GAME_MAP)  # MDP for agent policy
MDP_TASK = BoxPushTaskMDP(**GAME_MAP)  # MDP for task environment

TEMPERATURE = 1

if IS_TEAM:
  LIST_POLICY = bp_policy.get_exp1_policy(MDP_AGENT, TEMPERATURE)
else:
  LIST_POLICY = bp_policy.get_indv_policy(MDP_AGENT, TEMPERATURE)


def cal_policy_kl_error(mdp_agent: BoxPushAgentMDP, trajectories, cb_pi_true,
                        cb_pi_infer):
  def compute_relative_freq(num_latents, num_states, trajs):
    rel_freq_a1 = np.zeros((num_latents, num_states))
    rel_freq_a2 = np.zeros((num_latents, num_states))
    count = 0
    for traj in trajs:
      for s, joint_a, joint_x in traj:
        rel_freq_a1[joint_x[0], s] += 1
        rel_freq_a2[joint_x[1], s] += 1
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


def np_true_policy(agent_idx, latent_idx, state_idx):
  if IS_TEAM:
    np_joint_act_p = LIST_POLICY[latent_idx][state_idx].reshape(
        MDP_AGENT.a1_a_space.num_actions, MDP_AGENT.a2_a_space.num_actions)
    if agent_idx == 0:
      return np.sum(np_joint_act_p, axis=1)
    else:
      return np.sum(np_joint_act_p, axis=0)
  else:
    if agent_idx == 0:
      return LIST_POLICY[latent_idx][state_idx]
    else:
      pos1, pos2, bstates = MDP_AGENT.conv_mdp_sidx_to_sim_states(state_idx)
      bstates_2 = bp_mdp.get_agent_switched_boxstates(bstates,
                                                      len(MDP_AGENT.drops),
                                                      len(MDP_AGENT.goals))
      sidx_2 = MDP_AGENT.conv_sim_states_to_mdp_sidx(pos2, pos1, bstates_2)
      return LIST_POLICY[latent_idx][sidx_2]


def true_policy(agent_idx, latent_idx, state_idx, tuple_action_idx):
  if IS_TEAM:
    indices = None
    if agent_idx == 0:
      indices = MDP_TASK.np_action_to_idx[tuple_action_idx[agent_idx], :]
    else:
      indices = MDP_TASK.np_action_to_idx[:, tuple_action_idx[agent_idx]]
    return np.sum(LIST_POLICY[latent_idx][state_idx, indices])
  else:
    if agent_idx == 0:
      return LIST_POLICY[latent_idx][state_idx, tuple_action_idx[agent_idx]]
    else:
      pos1, pos2, bstates = MDP_AGENT.conv_mdp_sidx_to_sim_states(state_idx)
      bstates_2 = bp_mdp.get_agent_switched_boxstates(bstates,
                                                      len(MDP_AGENT.drops),
                                                      len(MDP_AGENT.goals))
      sidx_2 = MDP_AGENT.conv_sim_states_to_mdp_sidx(pos2, pos1, bstates_2)
      return LIST_POLICY[latent_idx][sidx_2, tuple_action_idx[agent_idx]]


def true_Tx(agent_idx, latent_idx, state_idx, tuple_action_idx, next_state_idx,
            latent_next_idx):
  np_Tx = get_np_Tx(MDP_AGENT, agent_idx, latent_idx, state_idx,
                    tuple_action_idx, next_state_idx)
  return np_Tx[latent_next_idx]


def true_Tx_for_var_infer(agent_idx, latent_idx, state_idx, action1_idx,
                          action2_idx, next_state_idx, latent_next_idx):
  joint_action = (action1_idx, action2_idx)
  return true_Tx(agent_idx, latent_idx, state_idx, joint_action, next_state_idx,
                 latent_next_idx)


def initial_latent_pr(agent_idx, state_idx, latent_idx):
  if IS_TEAM:
    np_bx = bp_tx.get_team_np_bx(MDP_AGENT, agent_idx, state_idx)
  else:
    np_bx = bp_tx.get_indv_np_bx(MDP_AGENT, agent_idx, state_idx)

  return np_bx[latent_idx]


def get_result(var_inf_obj: VarInferDuo, test_samples):
  def var_inf_policy(agent_idx, latent_idx, state_idx, tuple_action_idx):
    p = var_inf_obj.list_np_policy[agent_idx][latent_idx, state_idx,
                                              tuple_action_idx[agent_idx]]
    return p

  def var_inf_Tx(agent_idx, latent_idx, state_idx, tuple_action_idx,
                 next_state_idx, latent_next_idx):
    p = var_inf_obj.get_Tx_pr(agent_idx, latent_idx, state_idx,
                              tuple_action_idx[0], tuple_action_idx[1],
                              next_state_idx, latent_next_idx)
    return p

  np_results = np.zeros((len(test_samples), 3))
  for idx, sample in enumerate(test_samples):
    mpseq_x_infer = most_probable_sequence(sample[0], sample[1], 2,
                                           MDP_AGENT.num_latents,
                                           var_inf_policy, var_inf_Tx,
                                           initial_latent_pr)
    seq_x_per_agent = list(zip(*sample[2]))
    res1 = norm_hamming_distance(seq_x_per_agent[0], mpseq_x_infer[0])
    res2 = norm_hamming_distance(seq_x_per_agent[1], mpseq_x_infer[1])

    align_true = alignment_sequence(seq_x_per_agent[0], seq_x_per_agent[1])
    align_infer = alignment_sequence(mpseq_x_infer[0], mpseq_x_infer[1])
    res3 = norm_hamming_distance(align_true, align_infer)

    np_results[idx, :] = [res1, res2, res3]

  return np_results


def get_result_with_true(test_samples):
  np_results = np.zeros((len(test_samples), 3))
  for idx, sample in enumerate(test_samples):
    mpseq_x_infer = most_probable_sequence(sample[0], sample[1], 2,
                                           MDP_AGENT.num_latents, true_policy,
                                           true_Tx, initial_latent_pr)
    seq_x_per_agent = list(zip(*sample[2]))
    # print(mpseq_x_infer)
    # print(seq_x_per_agent)
    res1 = norm_hamming_distance(seq_x_per_agent[0], mpseq_x_infer[0])
    res2 = norm_hamming_distance(seq_x_per_agent[1], mpseq_x_infer[1])

    align_true = alignment_sequence(seq_x_per_agent[0], seq_x_per_agent[1])
    align_infer = alignment_sequence(mpseq_x_infer[0], mpseq_x_infer[1])
    res3 = norm_hamming_distance(align_true, align_infer)

    np_results[idx, :] = [res1, res2, res3]

  return np_results


loaded_transition_model = None


def transition_s(sidx, aidx1, aidx2, sidx_n):
  global loaded_transition_model
  if loaded_transition_model is None:
    pickle_trans_s = os.path.join(DATA_DIR, SAVED_TRANSITION + ".pickle")
    if os.path.exists(pickle_trans_s):
      with open(pickle_trans_s, 'rb') as handle:
        loaded_transition_model = pickle.load(handle)
      print("transition_s loaded by pickle")
    else:
      loaded_transition_model = MDP_TASK.np_transition_model
      print("save transition_s by pickle")
      with open(pickle_trans_s, 'wb') as handle:
        pickle.dump(loaded_transition_model,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

  # mdp_task.np_transition_model
  aidx_team = MDP_TASK.np_action_to_idx[aidx1, aidx2]
  p = loaded_transition_model[sidx, aidx_team, sidx_n]
  # p = MDP_TASK.np_transition_model[sidx, aidx_team, sidx_n]
  return p


if __name__ == "__main__":

  # set simulator
  #############################################################################
  sim = BoxPushSimulator(0)
  sim.init_game(**GAME_MAP)
  sim.max_steps = 200

  def get_a1_action(**kwargs):
    return get_box_push_action(MDP_AGENT, BoxPushSimulator.AGENT1, TEMPERATURE,
                               **kwargs)

  def get_a2_action(**kwargs):
    return get_box_push_action(MDP_AGENT, BoxPushSimulator.AGENT2, TEMPERATURE,
                               **kwargs)

  def get_a1_latent(cur_state, a1_action, a2_action, a1_latent, next_state):
    return get_box_push_Tx(MDP_AGENT, 0, a1_latent, cur_state, a1_action,
                           a2_action, next_state)

  def get_a2_latent(cur_state, a1_action, a2_action, a2_latent, next_state):
    return get_box_push_Tx(MDP_AGENT, 1, a2_latent, cur_state, a1_action,
                           a2_action, next_state)

  def get_init_x(box_states, a1_pos, a2_pos):
    return bp_tx.get_init_x(MDP_AGENT, box_states, a1_pos, a2_pos, IS_TEAM)

  sim.set_autonomous_agent(cb_get_A1_action=get_a1_action,
                           cb_get_A2_action=get_a2_action,
                           cb_get_A1_mental_state=get_a1_latent,
                           cb_get_A2_mental_state=get_a2_latent,
                           cb_get_init_mental_state=get_init_x)

  # generate data
  #############################################################################
  GEN_TRAIN_SET = False
  GEN_TEST_SET = False

  SHOW_TRUE = True
  SHOW_SL_SMALL = True
  SHOW_SL_LARGE = True
  SHOW_SEMI = True
  VI_TRAIN = SHOW_TRUE or SHOW_SL_SMALL or SHOW_SL_LARGE or SHOW_SEMI

  TRAIN_DIR = os.path.join(DATA_DIR, 'box_push_train')
  TEST_DIR = os.path.join(DATA_DIR, 'box_push_test')

  train_prefix = "train_"
  test_prefix = "test_"
  if GEN_TRAIN_SET:
    file_names = glob.glob(os.path.join(TRAIN_DIR, train_prefix + '*.txt'))
    for fmn in file_names:
      os.remove(fmn)

    sim.run_simulation(500, os.path.join(TRAIN_DIR, train_prefix), "header")

  if GEN_TEST_SET:
    file_names = glob.glob(os.path.join(TEST_DIR, test_prefix + '*.txt'))
    for fmn in file_names:
      os.remove(fmn)

    sim.run_simulation(100, os.path.join(TEST_DIR, test_prefix), "header")

  # train variational inference
  #############################################################################
  if VI_TRAIN:
    # import matplotlib.pyplot as plt

    # load train set
    ##################################################
    traj_labeled_ver = []  # labels
    traj_unlabel_ver = []  # no labels
    traj_random_ver = []  # each time step is randomly labeled
    # traj_critical_ver = []  # labeled only at the critical points
    file_names = glob.glob(os.path.join(TRAIN_DIR, train_prefix + '*.txt'))
    for idx, file_nm in enumerate(file_names):
      trj = BoxPushSimulator.read_file(file_nm)
      trj_labeled = []
      trj_unlabel = []
      trj_random = []
      # traj_critical = []
      for bstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat in trj:
        sidx = MDP_TASK.conv_sim_states_to_mdp_sidx(a1pos, a2pos, bstt)
        aidx1 = MDP_TASK.a1_a_space.action_to_idx[a1act]
        aidx2 = MDP_TASK.a2_a_space.action_to_idx[a2act]

        # # to remove labels
        xidx1 = MDP_AGENT.latent_space.state_to_idx[a1lat]
        xidx2 = MDP_AGENT.latent_space.state_to_idx[a2lat]

        trj_labeled.append([sidx, (aidx1, aidx2), (xidx1, xidx2)])
        trj_unlabel.append([sidx, (aidx1, aidx2), (None, None)])
        xidx1_r = xidx1
        xidx2_r = xidx2
        if random.uniform(0, 1) < 0.5:
          xidx1_r = None
          xidx2_r = None
        trj_random.append([sidx, (aidx1, aidx2), (xidx1_r, xidx2_r)])

      traj_labeled_ver.append(trj_labeled)
      traj_unlabel_ver.append(trj_unlabel)
      traj_random_ver.append(trj_random)
    print(len(traj_labeled_ver))

    # load test set
    ##################################################
    test_file_names = glob.glob(os.path.join(TEST_DIR, test_prefix + '*.txt'))
    # test_file_names = [test_file_names[0]]
    test_traj = []
    for idx, file_nm in enumerate(test_file_names):
      trj = BoxPushSimulator.read_file(file_nm)
      traj_states = []
      traj_actions = []
      traj_labels = []
      # traj_critical = []
      for bstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat in trj:
        sidx = MDP_TASK.conv_sim_states_to_mdp_sidx(a1pos, a2pos, bstt)
        traj_states.append(sidx)

        aidx1 = MDP_TASK.a1_a_space.action_to_idx[a1act]
        aidx2 = MDP_TASK.a2_a_space.action_to_idx[a2act]
        traj_actions.append((aidx1, aidx2))

        xidx1 = MDP_AGENT.latent_space.state_to_idx[a1lat]
        xidx2 = MDP_AGENT.latent_space.state_to_idx[a2lat]
        traj_labels.append((xidx1, xidx2))

      test_traj.append([traj_states, traj_actions, traj_labels])
    print(len(test_traj))

    # true policy and transition
    ##################################################
    BETA_PI = 1.01
    BETA_TX1 = 1.01
    BETA_TX2 = 1.01
    print("beta: %f, %f, %f" % (BETA_PI, BETA_TX1, BETA_TX2))

    num_a1_action, num_a2_action = ((MDP_AGENT.a1_a_space.num_actions,
                                     MDP_AGENT.a2_a_space.num_actions)
                                    if IS_TEAM else (MDP_AGENT.num_actions,
                                                     MDP_AGENT.num_actions))
    joint_action = (num_a1_action, num_a2_action)

    # True policy
    if SHOW_TRUE:
      np_results = get_result_with_true(test_traj)
      avg1, avg2, avg3 = np.mean(np_results, axis=0)
      print("%f, %f, %f" % (avg1, avg2, avg3))
      # kl1, kl2 = cal_policy_kl_error(MDP_AGENT, traj_labeled_ver,
      #                                np_true_policy, np_true_policy)
      # print("kl1: %f, kl2: %f" % (kl1, kl2))

    # fig1 = plt.figure(figsize=(8, 3))
    # ax1 = fig1.add_subplot(131)
    # ax2 = fig1.add_subplot(132)
    # ax3 = fig1.add_subplot(133)

    idx_small = int(len(traj_labeled_ver) / 5)
    print(idx_small)
    # supervised with small samples
    if SHOW_SL_SMALL:
      print("Small")
      var_inf_small = VarInferDuo(traj_labeled_ver[0:idx_small],
                                  MDP_TASK.num_states, MDP_AGENT.num_latents,
                                  joint_action, transition_s)
      var_inf_small.set_dirichlet_prior(BETA_PI, BETA_TX1, BETA_TX2)
      var_inf_small.set_bx_and_Tx(cb_bx=initial_latent_pr,
                                  cb_Tx=true_Tx_for_var_infer)
      var_inf_small.do_inference()

      np_results = get_result(var_inf_small, test_traj)
      avg1, avg2, avg3 = np.mean(np_results, axis=0)
      print("%f, %f, %f" % (avg1, avg2, avg3))

      def np_infer_policy(agent_idx, xidx, sidx):
        return var_inf_small.list_np_policy[agent_idx][xidx, sidx]

      kl1, kl2 = cal_policy_kl_error(MDP_AGENT, traj_labeled_ver,
                                     np_true_policy, np_infer_policy)
      print("kl1: %f, kl2: %f" % (kl1, kl2))

      # ax1.plot(list_res1, 'r')
      # ax2.plot(list_res2, 'r')
      # ax3.plot(list_res3, 'r')
      # plt.show()

    # supervised with large samples
    if SHOW_SL_LARGE:
      print("Large")
      var_inf_large = VarInferDuo(traj_labeled_ver, MDP_TASK.num_states,
                                  MDP_AGENT.num_latents, joint_action,
                                  transition_s)
      var_inf_large.set_dirichlet_prior(BETA_PI, BETA_TX1, BETA_TX2)
      var_inf_large.set_bx_and_Tx(cb_bx=initial_latent_pr,
                                  cb_Tx=true_Tx_for_var_infer)
      var_inf_large.do_inference()

      np_results = get_result(var_inf_large, test_traj)
      avg1, avg2, avg3 = np.mean(np_results, axis=0)
      print("%f, %f, %f" % (avg1, avg2, avg3))

      def np_infer_policy(agent_idx, xidx, sidx):
        return var_inf_large.list_np_policy[agent_idx][xidx, sidx]

      kl1, kl2 = cal_policy_kl_error(MDP_AGENT, traj_labeled_ver,
                                     np_true_policy, np_infer_policy)
      print("kl1: %f, kl2: %f" % (kl1, kl2))

      # ax1.plot(list_res1)
      # ax2.plot(list_res2)
      # ax3.plot(list_res3)
      # plt.show()

      # print(mdp_agent.num_latents)

    # supervised with large samples
    if SHOW_SEMI:
      print("Semi")
      # semi-supervised with 50:50 labeled and unlabeled samples
      var_inf_semi = VarInferDuo(
          traj_labeled_ver[0:idx_small] + traj_unlabel_ver[idx_small:],
          MDP_TASK.num_states, MDP_AGENT.num_latents, joint_action,
          transition_s)
      var_inf_semi.set_dirichlet_prior(BETA_PI, BETA_TX1, BETA_TX2)
      var_inf_semi.set_bx_and_Tx(cb_bx=initial_latent_pr,
                                 cb_Tx=true_Tx_for_var_infer)
      save_path = os.path.join(
          DATA_DIR, "indv_semi_%f_%f_%f.npz" % (BETA_PI, BETA_TX1, BETA_TX2))
      var_inf_semi.set_load_save_file_name(save_path)
      var_inf_semi.do_inference()

      np_results = get_result(var_inf_semi, test_traj)
      avg1, avg2, avg3 = np.mean(np_results, axis=0)
      print("%f, %f, %f" % (avg1, avg2, avg3))

      def np_infer_policy(agent_idx, xidx, sidx):
        return var_inf_semi.list_np_policy[agent_idx][xidx, sidx]

      kl1, kl2 = cal_policy_kl_error(MDP_AGENT, traj_labeled_ver,
                                     np_true_policy, np_infer_policy)
      print("kl1: %f, kl2: %f" % (kl1, kl2))

      # ax1.plot(list_res1)
      # ax2.plot(list_res2)
      # ax3.plot(list_res3)
      # plt.show()
