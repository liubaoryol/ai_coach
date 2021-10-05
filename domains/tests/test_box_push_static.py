import glob
import os
import numpy as np
import ai_coach_core.model_inference.var_infer.var_infer_static_x as var_infer
from ai_coach_core.latent_inference.bayesian_inference import (
    bayesian_mind_inference)

from ai_coach_domain.box_push_static.mdp import StaticBoxPushMDP
from ai_coach_domain.box_push.maps import EXP1_MAP
from ai_coach_domain.box_push_static.policy import (get_static_policy,
                                                    get_static_action)
from ai_coach_domain.box_push.simulator import BoxPushSimulator_AloneOrTogether

BoxPushSimulator = BoxPushSimulator_AloneOrTogether

DATA_DIR = os.path.join(os.path.dirname(__file__), "data/")
GAME_MAP = EXP1_MAP

MDP_AGENT = StaticBoxPushMDP(**GAME_MAP)  # MDP for agent policy
TEMPERATURE = 1
LIST_POLICY = get_static_policy(MDP_AGENT, TEMPERATURE)


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


if __name__ == "__main__":
  # set simulator
  #############################################################################
  sim = BoxPushSimulator(0)
  sim.init_game(**GAME_MAP)
  sim.max_steps = 200

  def get_a1_action(**kwargs):
    return get_static_action(MDP_AGENT, BoxPushSimulator.AGENT1, TEMPERATURE,
                             **kwargs)

  def get_a2_action(**kwargs):
    return get_static_action(MDP_AGENT, BoxPushSimulator.AGENT2, TEMPERATURE,
                             **kwargs)

  def get_init_x(box_states, a1_pos, a2_pos):
    a1_latent = ("together", 0) if np.random.randint(2) == 0 else ("alone", 0)
    a2_latent = ("together", 0) if np.random.randint(2) == 0 else ("alone", 0)
    return a1_latent, a2_latent

  sim.set_autonomous_agent(cb_get_A1_action=get_a1_action,
                           cb_get_A2_action=get_a2_action,
                           cb_get_A1_mental_state=None,
                           cb_get_A2_mental_state=None,
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

  TRAIN_DIR = os.path.join(DATA_DIR, 'static_bp_train')
  TEST_DIR = os.path.join(DATA_DIR, 'static_bp_test')

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

    # load train set
    ##################################################
    trajectories = []
    latent_labels = []
    file_names = glob.glob(os.path.join(TRAIN_DIR, train_prefix + '*.txt'))
    for idx, file_nm in enumerate(file_names):
      trj = BoxPushSimulator.read_file(file_nm)
      traj = []
      for bstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat in trj:
        sidx = MDP_AGENT.conv_sim_states_to_mdp_sidx(a1pos, a2pos, bstt)
        aidx1 = MDP_AGENT.a1_a_space.action_to_idx[a1act]
        aidx2 = MDP_AGENT.a2_a_space.action_to_idx[a2act]

        traj.append((sidx, (aidx1, aidx2)))

        xidx1 = MDP_AGENT.latent_space.state_to_idx[a1lat]
        xidx2 = MDP_AGENT.latent_space.state_to_idx[a2lat]

      trajectories.append(traj)
      latent_labels.append((xidx1, xidx2))

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
      trj = BoxPushSimulator.read_file(file_nm)
      traj = []
      for bstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat in trj:
        sidx = MDP_AGENT.conv_sim_states_to_mdp_sidx(a1pos, a2pos, bstt)
        aidx1 = MDP_AGENT.a1_a_space.action_to_idx[a1act]
        aidx2 = MDP_AGENT.a2_a_space.action_to_idx[a2act]
        traj.append((sidx, (aidx1, aidx2)))

        xidx1 = MDP_AGENT.latent_space.state_to_idx[a1lat]
        xidx2 = MDP_AGENT.latent_space.state_to_idx[a2lat]

      test_traj.append(traj)
      test_labels.append((xidx1, xidx2))
    print(len(test_traj))

    BETA_PI = 1.5
    joint_num_action = (MDP_AGENT.a1_a_space.num_actions,
                        MDP_AGENT.a2_a_space.num_actions)

    idx_small = int(len(trajectories) / 5)
    print(idx_small)
    num_agents = sim.get_num_agents()

    if SHOW_TRUE:

      def policy_true(agent_idx, x_idx, state_idx, joint_action):
        return np.sum(LIST_POLICY[x_idx][state_idx].reshape(joint_num_action),
                      axis=(1 - agent_idx))[joint_action[agent_idx]]

      sup_conf_true, full_acc_true = get_bayesian_infer_result(
          num_agents, policy_true, MDP_AGENT.num_latents, test_traj,
          test_labels)

      print("Full - True")
      print_conf(sup_conf_true)
      print("4by4 Acc: " + str(full_acc_true))

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
      print("4by4 Acc: " + str(full_acc1))

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
      print("4by4 Acc: " + str(full_acc2))

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
      print("4by4 Acc: " + str(semi_full_acc))
