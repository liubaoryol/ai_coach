from typing import List
import glob
import os
import pickle
import tempfile
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from ai_coach_core.model_inference.var_infer.var_infer_dynamic_x import (
    VarInferDuo)
from ai_coach_core.latent_inference.most_probable_sequence import (
    most_probable_sequence)
from ai_coach_core.utils.result_utils import (norm_hamming_distance,
                                              alignment_sequence)
from ai_coach_core.model_inference.behavior_cloning import behavior_cloning
from ai_coach_core.model_inference.sb3_algorithms import (behavior_cloning_sb3,
                                                          gail_on_agent)

import ai_coach_domain.box_push.maps as bp_maps
import ai_coach_domain.box_push.simulator as bp_sim
import ai_coach_domain.box_push.mdp as bp_mdp
import ai_coach_domain.box_push.mdppolicy as bp_policy
import ai_coach_domain.box_push.agent as bp_agent

DATA_DIR = os.path.join(os.path.dirname(__file__), "data/")
TEMPERATURE = 1

IS_TEST = False
IS_TEAM = False

if IS_TEST:
  GAME_MAP = bp_maps.TEST_MAP
  BoxPushPolicyTeam = bp_policy.BoxPushPolicyTeamTest
  BoxPushPolicyIndv = bp_policy.BoxPushPolicyIndvTest
else:
  GAME_MAP = bp_maps.EXP1_MAP
  BoxPushPolicyTeam = bp_policy.BoxPushPolicyTeamExp1
  BoxPushPolicyIndv = bp_policy.BoxPushPolicyIndvExp1

if IS_TEAM:
  SAVE_PREFIX = GAME_MAP["name"] + "_team"
  BoxPushSimulator = bp_sim.BoxPushSimulator_AlwaysTogether
  BoxPushAgentMDP = bp_mdp.BoxPushTeamMDP_AlwaysTogether
  BoxPushTaskMDP = bp_mdp.BoxPushTeamMDP_AlwaysTogether
else:
  SAVE_PREFIX = GAME_MAP["name"] + "_indv"
  BoxPushSimulator = bp_sim.BoxPushSimulator_AlwaysAlone
  BoxPushAgentMDP = bp_mdp.BoxPushAgentMDP_AlwaysAlone
  BoxPushTaskMDP = bp_mdp.BoxPushTeamMDP_AlwaysAlone

MDP_AGENT = BoxPushAgentMDP(**GAME_MAP)  # MDP for agent policy
MDP_TASK = BoxPushTaskMDP(**GAME_MAP)  # MDP for task environment


def cal_policy_kl_error(mdp_agent: BoxPushAgentMDP, trajectories, cb_pi_true,
                        cb_pi_infer):
  def compute_relative_freq(num_latents, num_states, trajs):
    rel_freq_a1 = np.zeros((num_latents, num_states))
    rel_freq_a2 = np.zeros((num_latents, num_states))
    count = 0
    for traj in trajs:
      for s, joint_a, joint_x in traj:
        if joint_x[0] is None or joint_x[1] is None:
          continue
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


def get_result(cb_get_np_policy_nxs, cb_get_np_Tx_nxsas,
               cb_get_np_init_latent_ns, test_samples):
  def policy_nxsa(nidx, xidx, sidx, tuple_aidx):
    return cb_get_np_policy_nxs(nidx, xidx, sidx)[tuple_aidx[nidx]]

  def Tx_nxsasx(nidx, xidx, sidx, tuple_aidx, sidx_n, xidx_n):
    return cb_get_np_Tx_nxsas(nidx, xidx, sidx, tuple_aidx, sidx_n)[xidx_n]

  def init_latent_nxs(nidx, xidx, sidx):
    return cb_get_np_init_latent_ns(nidx, sidx)[xidx]

  np_results = np.zeros((len(test_samples), 3))
  for idx, sample in enumerate(test_samples):
    mpseq_x_infer = most_probable_sequence(sample[0], sample[1], 2,
                                           MDP_AGENT.num_latents, policy_nxsa,
                                           Tx_nxsasx, init_latent_nxs)
    seq_x_per_agent = list(zip(*sample[2]))
    res1 = norm_hamming_distance(seq_x_per_agent[0], mpseq_x_infer[0])
    res2 = norm_hamming_distance(seq_x_per_agent[1], mpseq_x_infer[1])

    align_true = alignment_sequence(seq_x_per_agent[0], seq_x_per_agent[1])
    align_infer = alignment_sequence(mpseq_x_infer[0], mpseq_x_infer[1])
    res3 = norm_hamming_distance(align_true, align_infer)

    np_results[idx, :] = [res1, res2, res3]

  return np_results


g_loaded_transition_model = None


def transition_s(sidx, aidx1, aidx2, sidx_n=None):
  global g_loaded_transition_model
  if g_loaded_transition_model is None:
    pickle_trans_s = os.path.join(DATA_DIR, SAVE_PREFIX + "_mdp.pickle")
    if os.path.exists(pickle_trans_s):
      with open(pickle_trans_s, 'rb') as handle:
        g_loaded_transition_model = pickle.load(handle)
      print("transition_s loaded by pickle")
    else:
      g_loaded_transition_model = MDP_TASK.np_transition_model
      print("save transition_s by pickle")
      with open(pickle_trans_s, 'wb') as handle:
        pickle.dump(g_loaded_transition_model,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

  # mdp_task.np_transition_model
  aidx_team = MDP_TASK.np_action_to_idx[aidx1, aidx2]
  if sidx_n is None:
    return g_loaded_transition_model[sidx, aidx_team].todense()
  else:
    p = g_loaded_transition_model[sidx, aidx_team, sidx_n]
    # p = MDP_TASK.np_transition_model[sidx, aidx_team, sidx_n]
    return p


class TrueModelConverter:
  def __init__(self, agent1: bp_agent.BoxPushAIAgent_Abstract,
               agent2: bp_agent.BoxPushAIAgent_Abstract, num_latents) -> None:
    self.agent1 = agent1
    self.agent2 = agent2
    self.num_latents = num_latents

  def get_true_policy(self, agent_idx, latent_idx, state_idx):
    if agent_idx == 0:
      return self.agent1.policy_from_task_mdp_POV(state_idx, latent_idx)
    else:
      return self.agent2.policy_from_task_mdp_POV(state_idx, latent_idx)

  def get_true_Tx_nxsas(self, agent_idx, latent_idx, state_idx,
                        tuple_action_idx, next_state_idx):
    if agent_idx == 0:
      return self.agent1.transition_model_from_task_mdp_POV(
          latent_idx, state_idx, tuple_action_idx, next_state_idx)
    else:
      return self.agent2.transition_model_from_task_mdp_POV(
          latent_idx, state_idx, tuple_action_idx, next_state_idx)

  def get_init_latent_dist(self, agent_idx, state_idx):
    if agent_idx == 0:
      return self.agent1.init_latent_dist_from_task_mdp_POV(state_idx)
    else:
      return self.agent2.init_latent_dist_from_task_mdp_POV(state_idx)

  def true_Tx_for_var_infer(self, agent_idx, state_idx, action1_idx,
                            action2_idx, next_state_idx):
    joint_action = (action1_idx, action2_idx)
    np_Txx = np.zeros((self.num_latents, self.num_latents))
    for xidx in range(self.num_latents):
      np_Txx[xidx, :] = self.get_true_Tx_nxsas(agent_idx, xidx, state_idx,
                                               joint_action, next_state_idx)

    return np_Txx


class VarInfConverter:
  def __init__(self, var_inf_obj: VarInferDuo) -> None:
    self.var_inf_obj = var_inf_obj

  def policy_nxs(self, agent_idx, latent_idx, state_idx):
    return self.var_inf_obj.list_np_policy[agent_idx][latent_idx, state_idx]

  def Tx_nxsas(self, agent_idx, latent_idx, state_idx, tuple_action_idx,
               next_state_idx):
    return self.var_inf_obj.get_Tx(agent_idx, state_idx, tuple_action_idx[0],
                                   tuple_action_idx[1],
                                   next_state_idx)[latent_idx]


class Trajectories:
  EPISODE_END = -1
  DUMMY = -2

  def __init__(self, tuple_num_sax_factors, num_latents) -> None:
    self.list_np_trajectory = []  # type: List[np.ndarray]
    self.num_latents = num_latents

    # TODO: make more generic
    self.num_state_factors = tuple_num_sax_factors[0]
    self.num_action_factors = tuple_num_sax_factors[1]
    self.num_latent_factors = tuple_num_sax_factors[2]

  def get_width(self):
    return (self.num_state_factors + self.num_action_factors +
            self.num_latent_factors)

  def is_episode_end(self, np_row):
    '''
    can be the terminal of episode or
    just a normal state in case episode ended early due to max step count
    '''
    return np_row[self.num_state_factors] == Trajectories.EPISODE_END

  def get_as_row_lists(self, no_latent_label: bool, include_terminal: bool):
    list_list_SAX = []
    for np_trj in self.list_np_trajectory:
      list_SAX = []
      for idx in range(np_trj.shape[0]):
        list_tmp = []
        # TODO: make more generic
        # state
        start_idx, end_idx = 0, self.num_state_factors
        if self.num_state_factors == 1:
          list_tmp.append(np_trj[idx][start_idx:end_idx][0])
        else:
          list_tmp.append(tuple(np_trj[idx][start_idx:end_idx]))

        if self.is_episode_end(np_trj[idx]):
          if include_terminal:
            list_tmp.append(None)
            list_tmp.append(None)
          else:
            break
        else:
          # action
          start_idx, end_idx = end_idx, end_idx + self.num_action_factors
          if self.num_action_factors == 1:
            list_tmp.append(np_trj[idx][start_idx:end_idx][0])
          else:
            list_tmp.append(tuple(np_trj[idx][start_idx:end_idx]))
          # latent
          if no_latent_label:
            if self.num_latent_factors == 1:
              list_tmp.append(None)
            else:
              list_tmp.append((None, ) * self.num_latent_factors)
          else:
            start_idx, end_idx = end_idx, end_idx + self.num_latent_factors
            if self.num_latent_factors == 1:
              list_tmp.append(np_trj[idx][start_idx:end_idx][0])
            else:
              list_tmp.append(tuple(np_trj[idx][start_idx:end_idx]))

        list_SAX.append(list_tmp)
      list_list_SAX.append(list_SAX)
    return list_list_SAX

  def get_as_column_lists(self, include_terminal: bool):
    list_trajectories = []
    for np_trj in self.list_np_trajectory:
      is_terminal = self.is_episode_end(np_trj[-1, :])

      row_end_idx = (-1 if is_terminal and not include_terminal else
                     np_trj.shape[0])
      start_idx, end_idx = 0, self.num_state_factors
      if self.num_state_factors == 1:
        list_states = list(np_trj[:row_end_idx, start_idx:end_idx].squeeze())
      else:
        list_states = list(map(tuple, np_trj[:row_end_idx, start_idx:end_idx]))

      row_end_idx = -1 if is_terminal else np_trj.shape[0]
      start_idx, end_idx = end_idx, end_idx + self.num_action_factors
      if self.num_action_factors == 1:
        list_actions = list(np_trj[:row_end_idx, start_idx:end_idx].squeeze())
      else:
        list_actions = list(map(tuple, np_trj[:row_end_idx, start_idx:end_idx]))

      start_idx, end_idx = end_idx, end_idx + self.num_latent_factors
      if self.num_latent_factors == 1:
        list_latents = list(np_trj[:row_end_idx, start_idx:end_idx].squeeze())
      else:
        list_latents = list(map(tuple, np_trj[:row_end_idx, start_idx:end_idx]))

      list_trajectories.append([list_states, list_actions, list_latents])

    return list_trajectories

  def get_trajectories_fragmented_by_latent(self,
                                            include_next_state: bool,
                                            num_training_samples=None):
    '''
    num_training_samples: set the number of trajectories to use for training
                          if None, all data are assumed to be used
    return: trajectories fragmented by latents for each agent
            can be accessed as list_name[agent][latent][trajectory][time_step]
    only works when num_action_factors == num_latent_factors
    '''
    num_samples = (len(self.list_np_trajectory)
                   if num_training_samples is None else num_training_samples)

    list_by_agent = []
    idx_lat_start = self.num_state_factors + self.num_action_factors
    for nidx in range(self.num_action_factors):
      list_lat = [list() for dummy in range(self.num_latents)]
      for np_trj in self.list_np_trajectory[:num_samples]:
        if np_trj.shape[0] < 2:
          continue

        list_frag_trj = []
        for tidx in range(np_trj.shape[0] - 1):
          tidx_n = tidx + 1

          np_sidx = np_trj[tidx, 0:self.num_state_factors]
          sidx = np_sidx[0] if self.num_state_factors == 1 else tuple(np_sidx)

          np_sidx_n = np_trj[tidx_n, 0:self.num_state_factors]
          sidx_n = (np_sidx_n[0]
                    if self.num_state_factors == 1 else tuple(np_sidx_n))

          aidx = np_trj[tidx, self.num_state_factors + nidx]
          aidx_n = np_trj[tidx_n, self.num_state_factors + nidx]

          xidx = np_trj[tidx, idx_lat_start + nidx]
          xidx_n = np_trj[tidx_n, idx_lat_start + nidx]

          list_frag_trj.append((sidx, aidx))
          if xidx != xidx_n:
            if include_next_state:
              if self.is_episode_end(np_trj[tidx_n]):
                list_frag_trj.append((sidx_n, Trajectories.EPISODE_END))
              else:
                list_frag_trj.append((sidx_n, Trajectories.DUMMY))

            list_lat[xidx].append(list_frag_trj)
            list_frag_trj = []
          # handle the last element if not handled yet
          # this cannot be terminal as the case is already handled above
          elif tidx == np_trj.shape[0] - 2:
            if include_next_state:
              list_frag_trj.append((sidx_n, Trajectories.DUMMY))
            else:
              list_frag_trj.append((sidx_n, aidx_n))
            list_lat[xidx].append(list_frag_trj)

      list_by_agent.append(list_lat)
    return list_by_agent


class BoxPushTrajectories(Trajectories):
  def __init__(self, num_latents) -> None:
    super().__init__(tuple_num_sax_factors=(1, 2, 2), num_latents=num_latents)

  def load_from_files(self, file_names):
    for file_nm in file_names:
      trj = BoxPushSimulator.read_file(file_nm)
      if len(trj) == 0:
        continue

      np_trj = np.zeros((len(trj), self.get_width()), dtype=np.int32)
      for tidx, vec_state_action in enumerate(trj):
        bstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat = vec_state_action

        sidx = MDP_TASK.conv_sim_states_to_mdp_sidx(a1pos, a2pos, bstt)
        aidx1 = (MDP_TASK.a1_a_space.action_to_idx[a1act]
                 if a1act is not None else Trajectories.EPISODE_END)
        aidx2 = (MDP_TASK.a2_a_space.action_to_idx[a2act]
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

  if IS_TEAM:
    policy1 = BoxPushPolicyTeam(MDP_AGENT, TEMPERATURE, BoxPushSimulator.AGENT1)
    policy2 = BoxPushPolicyTeam(MDP_AGENT, TEMPERATURE, BoxPushSimulator.AGENT2)
    agent1 = bp_agent.BoxPushAIAgent_Team1(policy1)
    agent2 = bp_agent.BoxPushAIAgent_Team2(policy2)
  else:
    policy = BoxPushPolicyIndv(MDP_AGENT, TEMPERATURE)
    agent1 = bp_agent.BoxPushAIAgent_Indv1(policy)
    agent2 = bp_agent.BoxPushAIAgent_Indv2(policy)

  sim.set_autonomous_agent(agent1, agent2)

  true_methods = TrueModelConverter(agent1, agent2, MDP_AGENT.num_latents)

  # generate data
  #############################################################################
  GEN_TRAIN_SET = False
  GEN_TEST_SET = False

  SHOW_TRUE = False

  BC = False
  TEST_SB3_BC = False

  SHOW_SL = True
  SL_TRUE_TX = False

  SHOW_SEMI = False
  SEMI_TRUE_TX = False

  GAIL = False

  VI_TRAIN = SHOW_TRUE or SHOW_SL or SHOW_SEMI or BC or GAIL

  PLOT = False

  TRAIN_DIR = os.path.join(DATA_DIR, SAVE_PREFIX + '_box_push_train')
  TEST_DIR = os.path.join(DATA_DIR, SAVE_PREFIX + '_box_push_test')

  train_prefix = "train_"
  test_prefix = "test_"
  if GEN_TRAIN_SET:
    file_names = glob.glob(os.path.join(TRAIN_DIR, train_prefix + '*.txt'))
    for fmn in file_names:
      os.remove(fmn)
    sim.run_simulation(200, os.path.join(TRAIN_DIR, train_prefix), "header")

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
    file_names = glob.glob(os.path.join(TRAIN_DIR, train_prefix + '*.txt'))

    train_data = BoxPushTrajectories(MDP_AGENT.num_latents)
    train_data.load_from_files(file_names)
    traj_labeled_ver = train_data.get_as_row_lists(no_latent_label=False,
                                                   include_terminal=False)
    traj_unlabel_ver = train_data.get_as_row_lists(no_latent_label=True,
                                                   include_terminal=False)

    print(len(traj_labeled_ver))

    # load test set
    ##################################################
    test_file_names = glob.glob(os.path.join(TEST_DIR, test_prefix + '*.txt'))

    test_data = BoxPushTrajectories(MDP_AGENT.num_latents)
    test_data.load_from_files(test_file_names)
    test_traj = test_data.get_as_column_lists(include_terminal=False)
    print(len(test_traj))

    # true policy and transition
    ##################################################
    if IS_TEAM:
      BETA_PI = 1.2
      BETA_TX1 = 1.01
      BETA_TX2 = 1.01
    else:
      BETA_PI = 1.01
      BETA_TX1 = 1.01
      BETA_TX2 = 1.01
    print("beta: %f, %f, %f" % (BETA_PI, BETA_TX1, BETA_TX2))

    joint_action_num = ((MDP_AGENT.a1_a_space.num_actions,
                         MDP_AGENT.a2_a_space.num_actions) if IS_TEAM else
                        (MDP_AGENT.num_actions, MDP_AGENT.num_actions))

    # True policy
    if SHOW_TRUE:
      print("#########")
      print("True")
      print("#########")
      np_results = get_result(true_methods.get_true_policy,
                              true_methods.get_true_Tx_nxsas,
                              true_methods.get_init_latent_dist, test_traj)
      avg1, avg2, avg3 = np.mean(np_results, axis=0)
      std1, std2, std3 = np.std(np_results, axis=0)

      kl1, kl2 = cal_policy_kl_error(MDP_AGENT, traj_labeled_ver,
                                     true_methods.get_true_policy,
                                     true_methods.get_true_policy)

      print("%f,%f,%f,%f,%f,%f" % (avg1, std1, avg2, std2, avg3, std3))
      print("kl1, kl2: %f,%f" % (kl1, kl2))

    # fig1 = plt.figure(figsize=(8, 3))
    # ax1 = fig1.add_subplot(131)
    # ax2 = fig1.add_subplot(132)
    # ax3 = fig1.add_subplot(133)

    list_idx = [20, 50, 100, len(traj_labeled_ver)]

    print(list_idx)
    if BC:
      for idx in list_idx:
        print("#########")
        print("BC %d" % (idx, ))
        print("#########")

        pi_a1 = np.zeros(
            (MDP_AGENT.num_latents, MDP_AGENT.num_states, joint_action_num[0]))
        pi_a2 = np.zeros(
            (MDP_AGENT.num_latents, MDP_AGENT.num_states, joint_action_num[1]))

        if TEST_SB3_BC:
          list_frag_traj = train_data.get_trajectories_fragmented_by_latent(
              include_next_state=True, num_training_samples=idx)

          for xidx in range(MDP_AGENT.num_latents):
            pi_a1[xidx] = behavior_cloning_sb3(list_frag_traj[0][xidx],
                                               MDP_AGENT.num_states,
                                               joint_action_num[0])
            pi_a2[xidx] = behavior_cloning_sb3(list_frag_traj[1][xidx],
                                               MDP_AGENT.num_states,
                                               joint_action_num[1])
        else:
          list_frag_traj = train_data.get_trajectories_fragmented_by_latent(
              include_next_state=False, num_training_samples=idx)

          for xidx in range(MDP_AGENT.num_latents):
            pi_a1[xidx] = behavior_cloning(list_frag_traj[0][xidx],
                                           MDP_AGENT.num_states,
                                           joint_action_num[0])
            pi_a2[xidx] = behavior_cloning(list_frag_traj[1][xidx],
                                           MDP_AGENT.num_states,
                                           joint_action_num[1])
        list_pi_bc = [pi_a1, pi_a2]

        def bc_policy_nxs(agent_idx, latent_idx, state_idx):
          return list_pi_bc[agent_idx][latent_idx, state_idx]

        np_results = get_result(bc_policy_nxs, true_methods.get_true_Tx_nxsas,
                                true_methods.get_init_latent_dist, test_traj)
        avg1, avg2, avg3 = np.mean(np_results, axis=0)
        std1, std2, std3 = np.std(np_results, axis=0)

        kl1, kl2 = cal_policy_kl_error(MDP_AGENT, traj_labeled_ver,
                                       true_methods.get_true_policy,
                                       bc_policy_nxs)
        print("%f,%f,%f,%f,%f,%f" % (avg1, std1, avg2, std2, avg3, std3))
        print("kl1, kl2: %f,%f" % (kl1, kl2))

    if GAIL:
      tempdir = tempfile.TemporaryDirectory(prefix="gail")
      tempdir_path = pathlib.Path(tempdir.name)

      list_frag_traj = train_data.get_trajectories_fragmented_by_latent(
          include_next_state=True, num_training_samples=None)

      def transition_for_gail_agent_1(sidx, aidx):
        aidx2 = np.random.choice(joint_action_num[1])
        next_sidx_dist = transition_s(sidx, aidx, aidx2)
        sidx_n = np.random.choice(MDP_TASK.num_states, p=next_sidx_dist)
        return sidx_n

      def transition_for_gail_agent_2(sidx, aidx):
        aidx1 = np.random.choice(joint_action_num[0])
        next_sidx_dist = transition_s(sidx, aidx1, aidx)
        sidx_n = np.random.choice(MDP_TASK.num_states, p=next_sidx_dist)
        return sidx_n

      def is_terminal(sidx):
        return MDP_TASK.is_terminal(sidx)

      init_sidx = MDP_TASK.conv_sim_states_to_mdp_sidx(
          sim.a1_init, sim.a2_init, [0] * len(GAME_MAP["boxes"]))

      pi_a1 = np.zeros(
          (MDP_AGENT.num_latents, MDP_AGENT.num_states, joint_action_num[0]))
      pi_a2 = np.zeros(
          (MDP_AGENT.num_latents, MDP_AGENT.num_states, joint_action_num[1]))
      for xidx in range(MDP_AGENT.num_latents):
        log_path = tempdir_path / ("A1_x" + str(xidx))
        pi_a1[xidx] = gail_on_agent(MDP_AGENT.num_states, joint_action_num[0],
                                    transition_for_gail_agent_1, is_terminal,
                                    init_sidx, list_frag_traj[0][xidx],
                                    Trajectories.EPISODE_END, log_path)
        log_path = tempdir_path / ("A2_x" + str(xidx))
        pi_a2[xidx] = gail_on_agent(MDP_AGENT.num_states, joint_action_num[1],
                                    transition_for_gail_agent_2, is_terminal,
                                    init_sidx, list_frag_traj[1][xidx],
                                    Trajectories.EPISODE_END, log_path)
        print("Done %d-th latent" % xidx)
      list_pi_gail = [pi_a1, pi_a2]

      def gail_policy_nxs(agent_idx, latent_idx, state_idx):
        return list_pi_gail[agent_idx][latent_idx, state_idx]

      np_results = get_result(gail_policy_nxs, true_methods.get_true_Tx_nxsas,
                              true_methods.get_init_latent_dist, test_traj)
      avg1, avg2, avg3 = np.mean(np_results, axis=0)
      std1, std2, std3 = np.std(np_results, axis=0)

      kl1, kl2 = cal_policy_kl_error(MDP_AGENT, traj_labeled_ver,
                                     true_methods.get_true_policy,
                                     gail_policy_nxs)
      print("%f,%f,%f,%f,%f,%f" % (avg1, std1, avg2, std2, avg3, std3))
      print("kl1, kl2: %f,%f" % (kl1, kl2))

    # supervised variational inference
    if SHOW_SL:
      for idx in list_idx:
        print("#########")
        print("SL with %d" % (idx, ))
        print("#########")
        var_inf_sl = VarInferDuo(traj_labeled_ver[0:idx],
                                 MDP_TASK.num_states,
                                 MDP_AGENT.num_latents,
                                 joint_action_num,
                                 transition_s,
                                 trans_x_dependency=(True, True, True, False))
        var_inf_sl.set_dirichlet_prior(BETA_PI, BETA_TX1, BETA_TX2)
        if SL_TRUE_TX:
          print("Train with true Tx")
          var_inf_sl.set_bx_and_Tx(cb_bx=true_methods.get_init_latent_dist,
                                   cb_Tx=true_methods.true_Tx_for_var_infer)
        else:
          print("Train without true Tx")
          var_inf_sl.set_bx_and_Tx(cb_bx=true_methods.get_init_latent_dist)

        var_inf_sl.do_inference()

        var_inf_sl_conv = VarInfConverter(var_inf_sl)
        np_results = get_result(var_inf_sl_conv.policy_nxs,
                                var_inf_sl_conv.Tx_nxsas,
                                true_methods.get_init_latent_dist, test_traj)
        avg1, avg2, avg3 = np.mean(np_results, axis=0)
        std1, std2, std3 = np.std(np_results, axis=0)

        kl1, kl2 = cal_policy_kl_error(MDP_AGENT, traj_labeled_ver,
                                       true_methods.get_true_policy,
                                       var_inf_sl_conv.policy_nxs)

        print("%f,%f,%f,%f,%f,%f" % (avg1, std1, avg2, std2, avg3, std3))
        print("kl1, kl2: %f,%f" % (kl1, kl2))

      # ax1.plot(list_res1, 'r')
      # ax2.plot(list_res2, 'r')
      # ax3.plot(list_res3, 'r')
      # plt.show()

    # semi-supervised
    if SHOW_SEMI:
      for idx in list_idx[:-1]:
        print("#########")
        print("Semi %d" % (idx, ))
        print("#########")
        # semi-supervised
        var_inf_semi = VarInferDuo(traj_labeled_ver[0:idx] +
                                   traj_unlabel_ver[idx:],
                                   MDP_TASK.num_states,
                                   MDP_AGENT.num_latents,
                                   joint_action_num,
                                   transition_s,
                                   trans_x_dependency=(True, True, True, False),
                                   epsilon=0.01,
                                   max_iteration=100)
        var_inf_semi.set_dirichlet_prior(BETA_PI, BETA_TX1, BETA_TX2)

        if SEMI_TRUE_TX:
          print("Train with true Tx")
          var_inf_semi.set_bx_and_Tx(cb_bx=true_methods.get_init_latent_dist,
                                     cb_Tx=true_methods.true_Tx_for_var_infer)
        else:
          print("Train without true Tx")
          var_inf_semi.set_bx_and_Tx(cb_bx=true_methods.get_init_latent_dist)

          save_name = SAVE_PREFIX + "_semi_%f_%f_%f.npz" % (BETA_PI, BETA_TX1,
                                                            BETA_TX2)
          save_path = os.path.join(DATA_DIR, save_name)
          var_inf_semi.set_load_save_file_name(save_path)

        var_inf_semi.do_inference()

        var_inf_semi_conv = VarInfConverter(var_inf_semi)
        if not SEMI_TRUE_TX:
          np_results = get_result(var_inf_semi_conv.policy_nxs,
                                  var_inf_semi_conv.Tx_nxsas,
                                  true_methods.get_init_latent_dist, test_traj)
          avg1, avg2, avg3 = np.mean(np_results, axis=0)
          std1, std2, std3 = np.std(np_results, axis=0)
          print("Prediction of latent with learned Tx")
          print("%f,%f,%f,%f,%f,%f" % (avg1, std1, avg2, std2, avg3, std3))

        np_results = get_result(var_inf_semi_conv.policy_nxs,
                                true_methods.get_true_Tx_nxsas,
                                true_methods.get_init_latent_dist, test_traj)
        avg1, avg2, avg3 = np.mean(np_results, axis=0)
        std1, std2, std3 = np.std(np_results, axis=0)
        print("Prediction of latent with true Tx")
        print("%f,%f,%f,%f,%f,%f" % (avg1, std1, avg2, std2, avg3, std3))

        kl1, kl2 = cal_policy_kl_error(MDP_AGENT, traj_labeled_ver,
                                       true_methods.get_true_policy,
                                       var_inf_semi_conv.policy_nxs)
        print("kl1, kl2: %f,%f" % (kl1, kl2))

  if PLOT:
    # ##############################################
    # # results

    fig = plt.figure(figsize=(7.2, 3))
    # str_title = (
    #     "hyperparam: " + str(SEMISUPER_HYPERPARAM) +
    #     ", # labeled: " + str(len(trajectories)) +
    #     ", # unlabeled: " + str(len(unlabeled_traj)))
    str_title = ("KL over beta")
    list_kl1_team = [
        0.211022, 0.209135, 0.198055, 0.164204, 0.157211, 0.157834, 0.166335,
        0.196487
    ]
    list_kl2_team = [
        0.207553, 0.205707, 0.194994, 0.163914, 0.159222, 0.161656, 0.172828,
        0.20674
    ]
    list_kl_idx_team = [1.0001, 1.001, 1.01, 1.1, 1.2, 1.3, 1.5, 2.0]

    list_kl1_indv = [0.188086, 0.185471, 0.183754, 0.225749, 0.340395]
    list_kl2_indv = [0.153631, 0.151331, 0.152605, 0.203895, 0.33548]

    list_kl_idx_indv = [1.0001, 1.001, 1.01, 1.04, 1.1]

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.grid(True)
    ax2.grid(True)
    ax1.plot(list_kl_idx_team,
             list_kl1_team,
             '.-',
             label="Human",
             clip_on=False,
             fillstyle='none')
    ax1.plot(list_kl_idx_team,
             list_kl2_team,
             '.-',
             label="Robot",
             clip_on=False,
             fillstyle='none')
    ax2.plot(list_kl_idx_indv,
             list_kl1_indv,
             '.-',
             label="Human",
             clip_on=False,
             fillstyle='none')
    ax2.plot(list_kl_idx_indv,
             list_kl2_indv,
             '.-',
             label="Robot",
             clip_on=False,
             fillstyle='none')
    # ax1.axhline(y=full_align_acc1, color='r', linestyle='-', label="SL-Small")
    # ax1.axhline(y=full_align_acc2, color='g', linestyle='-', label="SL-Large")
    FONT_SIZE = 16
    # TITLE_FONT_SIZE = 12
    # LEGENT_FONT_SIZE = 12
    ax1.set_ylabel("KL-Divergence", fontsize=FONT_SIZE)
    ax1.set_xlabel("Beta", fontsize=FONT_SIZE)
    ax1.legend()
    ax1.set_title("Movers and Packers", fontsize=FONT_SIZE)
    ax2.set_ylabel("KL-Divergence", fontsize=FONT_SIZE)
    ax2.set_xlabel("Beta", fontsize=FONT_SIZE)
    ax2.legend()
    ax2.set_title("Cleanup", fontsize=FONT_SIZE)

    # ax1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    # # ax1.set_ylim([70, 100])
    # # ax1.set_xlim([0, 16])
    # ax1.set_title("Full Sequence", fontsize=TITLE_FONT_SIZE)

    #   ax2.plot(part_acc_history,
    #            '.-',
    #            label="SemiSL",
    #            clip_on=False,
    #            fillstyle='none')
    #   if do_sup_infer:
    #     ax2.axhline(
    # y=part_align_acc1, color='r', linestyle='-', label="SL-Small")
    #     ax2.axhline(
    # y=part_align_acc2, color='g', linestyle='-', label="SL-Large")
    #   ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    #   # ax2.set_ylim([50, 80])
    #   # ax2.set_xlim([0, 16])
    #   ax2.set_title("Partial Sequence (5 Steps)", fontsize=TITLE_FONT_SIZE)
    #   handles, labels = ax2.get_legend_handles_labels()
    #   fig.legend(handles,
    #              labels,
    #              loc='center right',
    #              prop={'size': LEGENT_FONT_SIZE})
    #   fig.text(0.45, 0.04, 'Iteration', ha='center', fontsize=FONT_SIZE)
    #   fig.tight_layout(pad=2.0)
    #   fig.subplots_adjust(right=0.8, bottom=0.2)
    plt.show()
