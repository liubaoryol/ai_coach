import glob
import os
import pickle
import click
import logging
import numpy as np
import warnings
import matplotlib.pyplot as plt
import time

from aic_ml.BTIL.btil_for_two import BTILforTwo
from aic_core.utils.decoding import (most_probable_sequence)
from aic_core.utils.result_utils import (norm_hamming_distance,
                                         alignment_sequence,
                                         cal_latent_policy_error)

from aic_domain.helper import TrueModelConverter
import aic_domain.box_push.maps as bp_maps
import aic_domain.box_push.simulator as bp_sim
import aic_domain.box_push.mdp as bp_mdp
import aic_domain.box_push.policy as bp_policy
import aic_domain.box_push.agent as bp_agent
from aic_domain.box_push.utils import BoxPushTrajectories
from aic_domain.box_push.agent_model import get_holding_box_and_floor_boxes

DATA_DIR = os.path.join(os.path.dirname(__file__), "data/")

MDP_AGENT = None  # type: bp_mdp.BoxPushMDP  # MDP for agent policy
MDP_TASK = None  # type: bp_mdp.BoxPushMDP   # MDP for task environment
SAVE_PREFIX = None
BoxPushSimulator = None  # type: type[bp_sim.BoxPushSimulator]

g_loaded_transition_model = None


def get_suboptimal_stats(traj_labeled_ver, is_team):
  n_total = 0
  n_aligned = 0
  for traj in traj_labeled_ver:
    for s, a, x in traj:
      n_total += 1
      if is_team:
        n_aligned += 1 if x[0] == x[1] else 0
      else:
        bstates, pos1, pos2 = MDP_TASK.conv_mdp_sidx_to_sim_states(s)
        a1_box, a2_box, valid_box = get_holding_box_and_floor_boxes(
            bstates, len(MDP_TASK.drops), len(MDP_TASK.goals))
        if len(valid_box) > 1:
          a1_latent = MDP_AGENT.latent_space.idx_to_state[x[0]]
          a2_latent = MDP_AGENT.latent_space.idx_to_state[x[1]]
          if (a1_latent[0] == "pickup" and a2_latent[0] == "pickup"
              and a1_latent[1] == a2_latent[1]):
            continue
          elif (a1_latent[0] == "pickup" and a2_latent[0] != "pickup"
                and a1_latent[1] == a2_box):
            continue
          elif (a1_latent[0] != "pikcup" and a2_latent[0] == "pikcup"
                and a2_latent[1] == a1_box):
            continue
          elif (a1_latent[0] == "origin" or a2_latent[0] == "origin"):
            continue
          else:
            n_aligned += 1
        elif len(valid_box) == 1:
          if (a1_latent[0] == "pickup" and a2_latent[0] != "pickup"
              and a1_latent[1] == a2_box):
            continue
          elif (a1_latent[0] != "pikcup" and a2_latent[0] == "pikcup"
                and a2_latent[1] == a1_box):
            continue
          elif (a1_latent[0] == "origin" or a2_latent[0] == "origin"):
            continue
          else:
            n_aligned += 1
        else:
          if (a1_latent[0] == "origin" or a2_latent[0] == "origin"):
            continue
          else:
            n_aligned += 1
  return n_aligned / n_total


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


def transition_s(sidx, aidx1, aidx2, sidx_n=None):
  global g_loaded_transition_model
  if g_loaded_transition_model is None:
    file_name = SAVE_PREFIX + "_transition_" + MDP_TASK.map_to_str()
    pickle_trans_s = os.path.join(DATA_DIR, file_name + ".pickle")
    if os.path.exists(pickle_trans_s):
      with open(pickle_trans_s, 'rb') as handle:
        g_loaded_transition_model = pickle.load(handle)
      logging.info("transition_s loaded by pickle")
      warnings.warn(
          "The transition has been loaded from a file ({}). "
          "If any related implementation is changed, "
          "be sure to delete the saved file and regenerate it.".format(
              os.path.basename(pickle_trans_s)),
          stacklevel=2)
    else:
      g_loaded_transition_model = MDP_TASK.np_transition_model
      dir_name = os.path.dirname(pickle_trans_s)
      if not os.path.exists(dir_name):
        os.makedirs(dir_name)
      logging.info("save transition_s by pickle")
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


class BTILConverter:

  def __init__(self, var_inf_obj: BTILforTwo) -> None:
    self.var_inf_obj = var_inf_obj

  def policy_nxs(self, agent_idx, latent_idx, state_idx):
    return self.var_inf_obj.list_np_policy[agent_idx][latent_idx, state_idx]

  def Tx_nxsas(self, agent_idx, latent_idx, state_idx, tuple_action_idx,
               next_state_idx):
    return self.var_inf_obj.get_Tx(agent_idx, state_idx, tuple_action_idx[0],
                                   tuple_action_idx[1],
                                   next_state_idx)[latent_idx]


# yapf: disable
@click.command()
@click.option("--is_team", type=bool, default=True, help="team / indv")
@click.option("--is_test", type=bool, default=False, help="exp1 / test")
@click.option("--gen_trainset", type=bool, default=False, help="generate train set")  # noqa: E501
@click.option("--gen_testset", type=bool, default=False, help="generate test set")  # noqa: E501
@click.option("--show_random", type=bool, default=False, help="results of uniform policy")  # noqa: E501
@click.option("--show_bc", type=bool, default=False, help="behavioral cloning results")  # noqa: E501
@click.option("--dnn_bc", type=bool, default=True, help="dnn behavioral cloning")  # noqa: E501
@click.option("--show_sl", type=bool, default=False, help="")
@click.option("--show_semi", type=bool, default=False, help="")
@click.option("--show_ul", type=bool, default=False, help="")
@click.option("--use_true_tx", type=bool, default=True, help="")
@click.option("--magail", type=bool, default=False, help="")
@click.option("--num_processes", type=int, default=16, help="")
@click.option("--gail_batch_size", type=int, default=128, help="")
@click.option("--ppo_batch_size", type=int, default=64, help="")
@click.option("--num_iterations", type=int, default=300, help="")
@click.option("--pretrain_steps", type=int, default=100, help="")
@click.option("--use_ce", type=bool, default=False, help="")
@click.option("--num_run", type=int, default=1, help="")
@click.option("--only_20", type=bool, default=False, help="")
@click.option("--suboptimality_stats", type=bool, default=False, help="")
# yapf: enable
def main(is_team, is_test, gen_trainset, gen_testset, show_random, show_bc,
         dnn_bc, show_sl, show_semi, show_ul, use_true_tx, magail,
         num_processes, gail_batch_size, ppo_batch_size, num_iterations,
         pretrain_steps, use_ce, num_run, only_20, suboptimality_stats):
  global MDP_AGENT, MDP_TASK, SAVE_PREFIX, BoxPushSimulator

  str_options = "\n"
  str_options += "--is_team=%s\n" % is_team
  str_options += "--is_test=%s\n" % is_test
  str_options += "--gen_trainset=%s\n" % gen_trainset
  str_options += "--gen_testset=%s\n" % gen_testset
  str_options += "--show_random=%s\n" % show_random
  str_options += "--show_bc=%s\n" % show_bc
  str_options += "--dnn_bc=%s\n" % dnn_bc
  str_options += "--show_sl=%s\n" % show_sl
  str_options += "--show_semi=%s\n" % show_semi
  str_options += "--show_ul=%s\n" % show_ul
  str_options += "--use_true_tx=%s\n" % use_true_tx
  str_options += "--magail=%s\n" % magail
  str_options += "--num_processes=%s\n" % num_processes
  str_options += "--gail_batch_size=%s\n" % gail_batch_size
  str_options += "--ppo_batch_size=%s\n" % ppo_batch_size
  str_options += "--num_iterations=%s\n" % num_iterations
  str_options += "--pretrain_steps=%s\n" % pretrain_steps
  str_options += "--use_ce=%s\n" % use_ce
  str_options += "--num_run=%s\n" % num_run
  str_options += "--only_20=%s\n" % only_20
  logging.info(str_options)

  for dummy_run in range(num_run):
    logging.info("run count: %d" % (dummy_run, ))

    if is_test:
      GAME_MAP = bp_maps.TEST_MAP
      BoxPushPolicyTeam = bp_policy.BoxPushPolicyTeamTest
      BoxPushPolicyIndv = bp_policy.BoxPushPolicyIndvTest_New
    else:
      GAME_MAP = bp_maps.EXP1_MAP
      BoxPushPolicyTeam = bp_policy.BoxPushPolicyTeamExp1
      BoxPushPolicyIndv = bp_policy.BoxPushPolicyIndvExp1

    if is_team:
      SAVE_PREFIX = GAME_MAP["name"] + "_team"
      BoxPushSimulator = bp_sim.BoxPushSimulator_AlwaysTogether
    else:
      SAVE_PREFIX = GAME_MAP["name"] + "_indv"
      BoxPushSimulator = bp_sim.BoxPushSimulator_AlwaysAlone

    # set simulator
    ############################################################################
    sim = BoxPushSimulator(0)
    sim.init_game(**GAME_MAP)
    sim.max_steps = 200
    NUM_AGENT = 2
    TEMPERATURE = 1

    if is_team:
      MDP_AGENT = bp_mdp.BoxPushTeamMDP_AlwaysTogether(**GAME_MAP)
      MDP_TASK = MDP_AGENT
      policy1 = BoxPushPolicyTeam(MDP_TASK, TEMPERATURE,
                                  BoxPushSimulator.AGENT1)
      policy2 = BoxPushPolicyTeam(MDP_TASK, TEMPERATURE,
                                  BoxPushSimulator.AGENT2)
      agent1 = bp_agent.BoxPushAIAgent_Team1(policy1)
      agent2 = bp_agent.BoxPushAIAgent_Team2(policy2)
    else:
      MDP_AGENT = bp_mdp.BoxPushAgentMDP_AlwaysAlone(**GAME_MAP)
      MDP_TASK = bp_mdp.BoxPushTeamMDP_AlwaysAlone(**GAME_MAP)
      policy1 = BoxPushPolicyIndv(MDP_TASK,
                                  MDP_AGENT,
                                  temperature=TEMPERATURE,
                                  agent_idx=BoxPushSimulator.AGENT1)
      policy2 = BoxPushPolicyIndv(MDP_TASK,
                                  MDP_AGENT,
                                  temperature=TEMPERATURE,
                                  agent_idx=BoxPushSimulator.AGENT2)
      agent1 = bp_agent.BoxPushAIAgent_Indv1(policy1)
      agent2 = bp_agent.BoxPushAIAgent_Indv2(policy2)

    sim.set_autonomous_agent(agent1, agent2)

    init_state = MDP_TASK.conv_sim_states_to_mdp_sidx(
        [[0] * len(sim.box_states), sim.a1_init, sim.a2_init])

    true_methods = TrueModelConverter([agent1, agent2], MDP_AGENT.num_latents)

    # generate data
    ############################################################################

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
    ############################################################################
    # import matplotlib.pyplot as plt

    # load train set
    ##################################################
    file_names = glob.glob(os.path.join(TRAIN_DIR, train_prefix + '*.txt'))

    train_data = BoxPushTrajectories(MDP_TASK, MDP_AGENT)
    train_data.load_from_files(file_names)
    if num_run > 1:
      train_data.shuffle()
    traj_labeled_ver = train_data.get_as_row_lists(no_latent_label=False,
                                                   include_terminal=False)
    traj_unlabel_ver = train_data.get_as_row_lists(no_latent_label=True,
                                                   include_terminal=False)

    logging.info(len(traj_labeled_ver))

    # load test set
    ##################################################
    test_file_names = glob.glob(os.path.join(TEST_DIR, test_prefix + '*.txt'))

    test_data = BoxPushTrajectories(MDP_TASK, MDP_AGENT)
    test_data.load_from_files(test_file_names)
    test_traj = test_data.get_as_column_lists(include_terminal=False)
    logging.info(len(test_traj))

    if is_team:
      BETA_PI = 1.2
      BETA_TX1 = 1.01
      BETA_TX2 = 1.01
    else:
      BETA_PI = 1.01
      BETA_TX1 = 1.01
      BETA_TX2 = 1.01
    logging.info("beta: %f, %f, %f" % (BETA_PI, BETA_TX1, BETA_TX2))

    if suboptimality_stats:
      print(get_suboptimal_stats(traj_labeled_ver, is_team))

    joint_action_num = ((MDP_AGENT.a1_a_space.num_actions,
                         MDP_AGENT.a2_a_space.num_actions) if is_team else
                        (MDP_AGENT.num_actions, MDP_AGENT.num_actions))

    if show_random:
      logging.info("#########")
      logging.info("Random")
      logging.info("#########")

      def get_uniform_policy(agent_idx, latent_idx, state_idx):
        # return self.agent2.policy_from_task_mdp_POV(state_idx, latent_idx)
        return (np.ones(joint_action_num[agent_idx]) /
                joint_action_num[agent_idx])

      def get_uniform_tx(nidx, xidx, sidx, tuple_aidx, sidx_n):
        return (np.ones(MDP_AGENT.num_latents) / MDP_AGENT.num_latents)

      np_results = get_result(get_uniform_policy, get_uniform_tx,
                              true_methods.get_init_latent_dist, test_traj)
      avg1, avg2, avg3 = np.mean(np_results, axis=0)
      std1, std2, std3 = np.std(np_results, axis=0)
      logging.info("%f,%f,%f,%f,%f,%f" % (avg1, std1, avg2, std2, avg3, std3))

    if only_20:
      list_idx = [20]
    else:
      list_idx = [20, 50, 100, len(traj_labeled_ver)]

    logging.info(list_idx)
    if show_bc:
      for idx in list_idx:
        logging.info("#########")
        logging.info("BC %d" % (idx, ))
        logging.info("#########")

        pi_a1 = np.zeros(
            (MDP_AGENT.num_latents, MDP_AGENT.num_states, joint_action_num[0]))
        pi_a2 = np.zeros(
            (MDP_AGENT.num_latents, MDP_AGENT.num_states, joint_action_num[1]))

        if dnn_bc:
          import aic_ml.baselines.ikostrikov_gail as ikostrikov
          logging.info("BC by DNN")
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
          from aic_ml.baselines.tabular_bc import tabular_behavior_cloning
          train_data.set_num_samples_to_use(idx)
          list_frag_traj = train_data.get_trajectories_fragmented_by_latent(
              include_next_state=False)

          for xidx in range(MDP_AGENT.num_latents):
            pi_a1[xidx] = tabular_behavior_cloning(list_frag_traj[0][xidx],
                                                   MDP_AGENT.num_states,
                                                   joint_action_num[0])
            pi_a2[xidx] = tabular_behavior_cloning(list_frag_traj[1][xidx],
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
        logging.info("%f,%f,%f,%f,%f,%f" % (avg1, std1, avg2, std2, avg3, std3))
        logging.info(policy_errors)

    # supervised variational inference
    if show_sl:
      for idx in list_idx:
        logging.info("#########")
        logging.info("SL with %d" % (idx, ))
        logging.info("#########")
        var_inf_sl = BTILforTwo(traj_labeled_ver[0:idx],
                                MDP_TASK.num_states,
                                MDP_AGENT.num_latents,
                                joint_action_num,
                                transition_s,
                                trans_x_dependency=(True, True, True, False))
        var_inf_sl.set_dirichlet_prior(BETA_PI, BETA_TX1, BETA_TX2)
        if use_true_tx:
          logging.info("Train with true Tx")
          var_inf_sl.set_bx_and_Tx(cb_bx=true_methods.get_init_latent_dist,
                                   cb_Tx=true_methods.true_Tx_for_var_infer)
        else:
          logging.info("Train without true Tx")
          var_inf_sl.set_bx_and_Tx(cb_bx=true_methods.get_init_latent_dist)

        var_inf_sl.do_inference()

        var_inf_sl_conv = BTILConverter(var_inf_sl)
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

        logging.info("%f,%f,%f,%f,%f,%f" % (avg1, std1, avg2, std2, avg3, std3))
        logging.info(policy_errors)

      # ax1.plot(list_res1, 'r')
      # ax2.plot(list_res2, 'r')
      # ax3.plot(list_res3, 'r')
      # plt.show()

    # semi-supervised
    if show_semi:
      for idx in list_idx:
        if idx == len(traj_labeled_ver):
          continue
        logging.info("#########")
        logging.info("Semi %d" % (idx, ))
        logging.info("#########")
        # semi-supervised
        var_inf_semi = BTILforTwo(traj_labeled_ver[0:idx] +
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
          logging.info("Train with true Tx")
          var_inf_semi.set_bx_and_Tx(cb_bx=true_methods.get_init_latent_dist,
                                     cb_Tx=true_methods.true_Tx_for_var_infer)
        else:
          logging.info("Train without true Tx")
          var_inf_semi.set_bx_and_Tx(cb_bx=true_methods.get_init_latent_dist)

        var_inf_semi.do_inference()

        var_inf_semi_conv = BTILConverter(var_inf_semi)
        if not use_true_tx:
          np_results = get_result(var_inf_semi_conv.policy_nxs,
                                  var_inf_semi_conv.Tx_nxsas,
                                  true_methods.get_init_latent_dist, test_traj)
          avg1, avg2, avg3 = np.mean(np_results, axis=0)
          std1, std2, std3 = np.std(np_results, axis=0)
          logging.info("Prediction of latent with learned Tx")
          logging.info("%f,%f,%f,%f,%f,%f" %
                       (avg1, std1, avg2, std2, avg3, std3))

        np_results = get_result(var_inf_semi_conv.policy_nxs,
                                true_methods.get_true_Tx_nxsas,
                                true_methods.get_init_latent_dist, test_traj)
        avg1, avg2, avg3 = np.mean(np_results, axis=0)
        std1, std2, std3 = np.std(np_results, axis=0)
        logging.info("Prediction of latent with true Tx")
        logging.info("%f,%f,%f,%f,%f,%f" % (avg1, std1, avg2, std2, avg3, std3))

        policy_errors = cal_latent_policy_error(NUM_AGENT, MDP_AGENT.num_states,
                                                MDP_AGENT.num_latents,
                                                traj_labeled_ver,
                                                true_methods.get_true_policy,
                                                var_inf_semi_conv.policy_nxs)
        logging.info(policy_errors)

    if show_ul:
      logging.info("#########")
      logging.info("UL %d" % (len(traj_labeled_ver), ))
      logging.info("#########")
      # unsupervised
      var_inf_ul = BTILforTwo(traj_unlabel_ver,
                              MDP_TASK.num_states,
                              MDP_AGENT.num_latents,
                              joint_action_num,
                              transition_s,
                              trans_x_dependency=(True, True, True, False),
                              epsilon=0.01,
                              max_iteration=100)
      var_inf_ul.set_dirichlet_prior(BETA_PI, BETA_TX1, BETA_TX2)

      if use_true_tx:
        logging.info("Train with true Tx")
        var_inf_ul.set_bx_and_Tx(cb_bx=true_methods.get_init_latent_dist,
                                 cb_Tx=true_methods.true_Tx_for_var_infer)
      else:
        logging.info("Train without true Tx")
        var_inf_ul.set_bx_and_Tx(cb_bx=true_methods.get_init_latent_dist)

      var_inf_ul.do_inference()

      var_inf_ul_conv = BTILConverter(var_inf_ul)
      if not use_true_tx:
        np_results = get_result_ul(var_inf_ul_conv.policy_nxs,
                                   var_inf_ul_conv.Tx_nxsas,
                                   true_methods.get_init_latent_dist, test_traj)
        avg1, avg2, avg3 = np.mean(np_results, axis=0)
        std1, std2, std3 = np.std(np_results, axis=0)
        logging.info("Prediction of latent with learned Tx")
        logging.info("%f,%f,%f,%f,%f,%f" % (avg1, std1, avg2, std2, avg3, std3))

      np_results = get_result_ul(var_inf_ul_conv.policy_nxs,
                                 true_methods.get_true_Tx_nxsas,
                                 true_methods.get_init_latent_dist, test_traj)
      avg1, avg2, avg3 = np.mean(np_results, axis=0)
      std1, std2, std3 = np.std(np_results, axis=0)
      logging.info("Prediction of latent with true Tx")
      logging.info("%f,%f,%f,%f,%f,%f" % (avg1, std1, avg2, std2, avg3, std3))

      policy_errors = cal_latent_policy_error(NUM_AGENT, MDP_AGENT.num_states,
                                              MDP_AGENT.num_latents,
                                              traj_labeled_ver,
                                              true_methods.get_true_policy,
                                              var_inf_ul_conv.policy_nxs)
      logging.info(policy_errors)

    if magail:
      from aic_ml.baselines.latent_magail import lmagail_w_ppo
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

        list_pi_magail = lmagail_w_ppo(MDP_TASK, [init_state],
                                       [agent1.agent_model, agent2.agent_model],
                                       traj_labeled_ver[0:idx],
                                       num_processes=num_processes,
                                       demo_batch_size=gail_batch_size,
                                       ppo_batch_size=ppo_batch_size,
                                       num_iterations=num_iterations,
                                       do_pretrain=True,
                                       bc_pretrain_steps=pretrain_steps,
                                       only_pretrain=True,
                                       use_ce=use_ce,
                                       callback_loss=get_loss_each_round)

        def magail_policy_nxs(agent_idx, latent_idx, state_idx):
          return list_pi_magail[agent_idx][latent_idx, state_idx]

        np_results = get_result(magail_policy_nxs,
                                true_methods.get_true_Tx_nxsas,
                                true_methods.get_init_latent_dist, test_traj)
        avg1, avg2, avg3 = np.mean(np_results, axis=0)
        std1, std2, std3 = np.std(np_results, axis=0)

        policy_errors = cal_latent_policy_error(NUM_AGENT, MDP_AGENT.num_states,
                                                MDP_AGENT.num_latents,
                                                traj_labeled_ver,
                                                true_methods.get_true_policy,
                                                magail_policy_nxs)
        logging.info("%f,%f,%f,%f,%f,%f" % (avg1, std1, avg2, std2, avg3, std3))
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
        work_type = "Team" if is_team else "Indv"
        plt.savefig('latent_magail loss (dynamic %s %d).png' % (work_type, idx))


if __name__ == "__main__":
  file_prefix = "box_push_dynamic_results"
  sec, msec = divmod(time.time() * 1000, 1000)
  time_stamp = '-%s' % (time.strftime('%Y%m%d_%H%M%S', time.gmtime(sec)), )
  file_name = (file_prefix + time_stamp + '.log')

  logging.basicConfig(
      level=logging.INFO,
      format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
      handlers=[logging.FileHandler(file_name),
                logging.StreamHandler()],
      force=True)
  logging.info('box push dynamic results')
  main()
