import glob
import os
import pickle
import click
import tempfile
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from ai_coach_core.model_inference.var_infer.var_infer_dynamic_x import (
    VarInferDuo)
from ai_coach_core.latent_inference.most_probable_sequence import (
    most_probable_sequence)
from ai_coach_core.utils.result_utils import (norm_hamming_distance,
                                              alignment_sequence,
                                              cal_latent_policy_error)
from ai_coach_core.model_inference.behavior_cloning import behavior_cloning
from ai_coach_core.utils.data_utils import Trajectories

import ai_coach_domain.box_push.maps as bp_maps
import ai_coach_domain.box_push.simulator as bp_sim
import ai_coach_domain.box_push.mdp as bp_mdp
import ai_coach_domain.box_push.mdppolicy as bp_policy
import ai_coach_domain.box_push.agent as bp_agent

DATA_DIR = os.path.join(os.path.dirname(__file__), "data/")
TEMPERATURE = 1

MDP_AGENT = None  # type: bp_mdp.BoxPushMDP  # MDP for agent policy
MDP_TASK = None  # type: bp_mdp.BoxPushMDP   # MDP for task environment
SAVE_PREFIX = None
BoxPushSimulator = None  # type: bp_sim.BoxPushSimulator


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


def get_result_ul(cb_get_np_policy_nxs, cb_get_np_Tx_nxsas,
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
    raise NotImplementedError
    # need a method to infer meaning of x...
    res1 = norm_hamming_distance(seq_x_per_agent[0], mpseq_x_infer[0])
    res2 = norm_hamming_distance(seq_x_per_agent[1], mpseq_x_infer[1])

    align_true = alignment_sequence(seq_x_per_agent[0], seq_x_per_agent[1])
    align_infer = alignment_sequence(mpseq_x_infer[0], mpseq_x_infer[1])
    res3 = norm_hamming_distance(align_true, align_infer)

    np_results[idx, :] = [res1, res2, res3]

  return np_results


def match_policy(cb_learned_policy, cb_true_policy):
  pass


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


class BoxPushTrajectories(Trajectories):
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


# yapf: disable
@click.command()
@click.option("--is_team", type=bool, default=False, help="team / indv")
@click.option("--is_test", type=bool, default=False, help="exp1 / test")
@click.option("--gen_trainset", type=bool, default=False, help="generate train set")
@click.option("--gen_testset", type=bool, default=False, help="generate test set")
@click.option("--show_true", type=bool, default=False, help="metrics from true policy")
@click.option("--show_bc", type=bool, default=False, help="behavioral cloning results")
@click.option("--dnn_bc", type=bool, default=True, help="dnn behavioral cloning")
@click.option("--show_sl", type=bool, default=False, help="")
@click.option("--show_semi", type=bool, default=False, help="")
@click.option("--show_ul", type=bool, default=False, help="")
@click.option("--use_true_tx", type=bool, default=True, help="")
@click.option("--magail", type=bool, default=False, help="")
@click.option("--num_processes", type=int, default=4, help="")
@click.option("--gail_batch_size", type=int, default=64, help="")
@click.option("--ppo_batch_size", type=int, default=32, help="")
@click.option("--num_iterations", type=int, default=300, help="")
@click.option("--pretrain_steps", type=int, default=100, help="")
@click.option("--use_ce", type=bool, default=False, help="")
# yapf: enable
def main(is_team, is_test, gen_trainset, gen_testset, show_true, show_bc,
         dnn_bc, show_sl, show_semi, show_ul, use_true_tx, magail,
         num_processes, gail_batch_size, ppo_batch_size, num_iterations,
         pretrain_steps, use_ce):
  global MDP_AGENT, MDP_TASK, SAVE_PREFIX, BoxPushSimulator

  if is_test:
    GAME_MAP = bp_maps.TEST_MAP
    BoxPushPolicyTeam = bp_policy.BoxPushPolicyTeamTest
    BoxPushPolicyIndv = bp_policy.BoxPushPolicyIndvTest
  else:
    GAME_MAP = bp_maps.EXP1_MAP
    BoxPushPolicyTeam = bp_policy.BoxPushPolicyTeamExp1
    BoxPushPolicyIndv = bp_policy.BoxPushPolicyIndvExp1

  if is_team:
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

  # set simulator
  #############################################################################
  sim = BoxPushSimulator(0)
  sim.init_game(**GAME_MAP)
  sim.max_steps = 200
  NUM_AGENT = 2

  if is_team:
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

  TRAIN_DIR = os.path.join(DATA_DIR, SAVE_PREFIX + '_box_push_train')
  TEST_DIR = os.path.join(DATA_DIR, SAVE_PREFIX + '_box_push_test')

  train_prefix = "train_"
  test_prefix = "test_"
  if gen_trainset:
    file_names = glob.glob(os.path.join(TRAIN_DIR, train_prefix + '*.txt'))
    for fmn in file_names:
      os.remove(fmn)
    sim.run_simulation(200, os.path.join(TRAIN_DIR, train_prefix), "header")

  if gen_testset:
    file_names = glob.glob(os.path.join(TEST_DIR, test_prefix + '*.txt'))
    for fmn in file_names:
      os.remove(fmn)
    sim.run_simulation(100, os.path.join(TEST_DIR, test_prefix), "header")

  # train variational inference
  #############################################################################
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
  if is_team:
    BETA_PI = 1.2
    BETA_TX1 = 1.01
    BETA_TX2 = 1.01
  else:
    BETA_PI = 1.01
    BETA_TX1 = 1.01
    BETA_TX2 = 1.01
  print("beta: %f, %f, %f" % (BETA_PI, BETA_TX1, BETA_TX2))

  joint_action_num = ((MDP_AGENT.a1_a_space.num_actions,
                       MDP_AGENT.a2_a_space.num_actions) if is_team else
                      (MDP_AGENT.num_actions, MDP_AGENT.num_actions))

  # True policy
  if show_true:
    print("#########")
    print("True")
    print("#########")
    np_results = get_result(true_methods.get_true_policy,
                            true_methods.get_true_Tx_nxsas,
                            true_methods.get_init_latent_dist, test_traj)
    avg1, avg2, avg3 = np.mean(np_results, axis=0)
    std1, std2, std3 = np.std(np_results, axis=0)

    policy_errors = cal_latent_policy_error(NUM_AGENT, MDP_AGENT.num_states,
                                            MDP_AGENT.num_latents,
                                            traj_labeled_ver,
                                            true_methods.get_true_policy,
                                            true_methods.get_true_policy)

    print("%f,%f,%f,%f,%f,%f" % (avg1, std1, avg2, std2, avg3, std3))
    print(policy_errors)

  # fig1 = plt.figure(figsize=(8, 3))
  # ax1 = fig1.add_subplot(131)
  # ax2 = fig1.add_subplot(132)
  # ax3 = fig1.add_subplot(133)

  list_idx = [20, 50, 100, len(traj_labeled_ver)]

  print(list_idx)
  if show_bc:
    for idx in list_idx:
      print("#########")
      print("BC %d" % (idx, ))
      print("#########")

      pi_a1 = np.zeros(
          (MDP_AGENT.num_latents, MDP_AGENT.num_states, joint_action_num[0]))
      pi_a2 = np.zeros(
          (MDP_AGENT.num_latents, MDP_AGENT.num_states, joint_action_num[1]))

      if dnn_bc:
        import ai_coach_core.model_inference.ikostrikov_gail as ikostrikov
        print("BC by DNN")
        train_data.set_num_samples_to_use(idx)
        list_frag_traj = train_data.get_trajectories_fragmented_by_latent(
            include_next_state=False)

        for xidx in range(MDP_AGENT.num_latents):
          pi_a1[xidx] = ikostrikov.bc_dnn(MDP_AGENT.num_states,
                                          joint_action_num[0],
                                          list_frag_traj[0][xidx],
                                          demo_batch_size=gail_batch_size,
                                          ppo_batch_size=ppo_batch_size,
                                          bc_pretrain_steps=pretrain_steps)
          pi_a2[xidx] = ikostrikov.bc_dnn(MDP_AGENT.num_states,
                                          joint_action_num[1],
                                          list_frag_traj[1][xidx],
                                          demo_batch_size=gail_batch_size,
                                          ppo_batch_size=ppo_batch_size,
                                          bc_pretrain_steps=pretrain_steps)

        # for xidx in range(MDP_AGENT.num_latents):
        #   pi_a1[xidx] = behavior_cloning_sb3(list_frag_traj[0][xidx],
        #                                      MDP_AGENT.num_states,
        #                                      joint_action_num[0])
        #   pi_a2[xidx] = behavior_cloning_sb3(list_frag_traj[1][xidx],
        #                                      MDP_AGENT.num_states,
        #                                      joint_action_num[1])
      else:
        train_data.set_num_samples_to_use(idx)
        list_frag_traj = train_data.get_trajectories_fragmented_by_latent(
            include_next_state=False)

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

      policy_errors = cal_latent_policy_error(NUM_AGENT, MDP_AGENT.num_states,
                                              MDP_AGENT.num_latents,
                                              traj_labeled_ver,
                                              true_methods.get_true_policy,
                                              bc_policy_nxs)
      print("%f,%f,%f,%f,%f,%f" % (avg1, std1, avg2, std2, avg3, std3))
      print(policy_errors)

  # supervised variational inference
  if show_sl:
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
      if use_true_tx:
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

      policy_errors = cal_latent_policy_error(NUM_AGENT, MDP_AGENT.num_states,
                                              MDP_AGENT.num_latents,
                                              traj_labeled_ver,
                                              true_methods.get_true_policy,
                                              var_inf_sl_conv.policy_nxs)

      print("%f,%f,%f,%f,%f,%f" % (avg1, std1, avg2, std2, avg3, std3))
      print(policy_errors)

    # ax1.plot(list_res1, 'r')
    # ax2.plot(list_res2, 'r')
    # ax3.plot(list_res3, 'r')
    # plt.show()

  # semi-supervised
  if show_semi:
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

      if use_true_tx:
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
      if not use_true_tx:
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

      policy_errors = cal_latent_policy_error(NUM_AGENT, MDP_AGENT.num_states,
                                              MDP_AGENT.num_latents,
                                              traj_labeled_ver,
                                              true_methods.get_true_policy,
                                              var_inf_semi_conv.policy_nxs)
      print(policy_errors)

  if show_ul:
    print("#########")
    print("UL %d" % (len(traj_labeled_ver), ))
    print("#########")
    # unsupervised
    var_inf_ul = VarInferDuo(traj_unlabel_ver,
                             MDP_TASK.num_states,
                             MDP_AGENT.num_latents,
                             joint_action_num,
                             transition_s,
                             trans_x_dependency=(True, True, True, False),
                             epsilon=0.01,
                             max_iteration=100)
    var_inf_ul.set_dirichlet_prior(BETA_PI, BETA_TX1, BETA_TX2)

    if use_true_tx:
      print("Train with true Tx")
      var_inf_ul.set_bx_and_Tx(cb_bx=true_methods.get_init_latent_dist,
                               cb_Tx=true_methods.true_Tx_for_var_infer)
    else:
      print("Train without true Tx")
      var_inf_ul.set_bx_and_Tx(cb_bx=true_methods.get_init_latent_dist)

      save_name = SAVE_PREFIX + "_ul_%f_%f_%f.npz" % (BETA_PI, BETA_TX1,
                                                      BETA_TX2)
      save_path = os.path.join(DATA_DIR, save_name)
      var_inf_ul.set_load_save_file_name(save_path)

    var_inf_ul.do_inference()

    var_inf_ul_conv = VarInfConverter(var_inf_ul)
    if not use_true_tx:
      np_results = get_result_ul(var_inf_ul_conv.policy_nxs,
                                 var_inf_ul_conv.Tx_nxsas,
                                 true_methods.get_init_latent_dist, test_traj)
      avg1, avg2, avg3 = np.mean(np_results, axis=0)
      std1, std2, std3 = np.std(np_results, axis=0)
      print("Prediction of latent with learned Tx")
      print("%f,%f,%f,%f,%f,%f" % (avg1, std1, avg2, std2, avg3, std3))

    np_results = get_result_ul(var_inf_ul_conv.policy_nxs,
                               true_methods.get_true_Tx_nxsas,
                               true_methods.get_init_latent_dist, test_traj)
    avg1, avg2, avg3 = np.mean(np_results, axis=0)
    std1, std2, std3 = np.std(np_results, axis=0)
    print("Prediction of latent with true Tx")
    print("%f,%f,%f,%f,%f,%f" % (avg1, std1, avg2, std2, avg3, std3))

    policy_errors = cal_latent_policy_error(NUM_AGENT, MDP_AGENT.num_states,
                                            MDP_AGENT.num_latents,
                                            traj_labeled_ver,
                                            true_methods.get_true_policy,
                                            var_inf_ul_conv.policy_nxs)
    print(policy_errors)


if __name__ == "__main__":
  main()