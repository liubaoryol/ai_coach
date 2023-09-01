import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
from aic_ml.BTIL.btil_static import BTILStatic
from aic_core.utils.static_inference import (bayesian_mental_state_inference)
from tests.examples.environment import RequestEnvironment
from tests.examples.tooldelivery_v3_env import ToolDeliveryEnv_V3


def read_sample(file_name):
  traj = []
  latents = []
  with open(file_name, newline='') as txtfile:
    lines = txtfile.readlines()
    latents = []
    for elem in lines[1].rstrip().split(", "):
      if elem.isdigit():
        latents.append(int(elem))
      elif elem == "None":
        latents.append(None)

    for i_r in range(3, len(lines)):
      line = lines[i_r]
      traj.append(tuple([int(elem) for elem in line.rstrip().split(", ")]))
  return traj, tuple(latents)


def get_tooldelivery_partial_traj(env: RequestEnvironment,
                                  full_trajectory,
                                  num_b4_request=None,
                                  num_af_request=None):
  '''
    num_b4_request: the number of steps before the tool request (>=0)
                    if None, take the sequence from the start
                    if 0, no sequence before the request will be included.
    num_af_request: the number of steps after the tool request (>=0)
                    if None, take the sequence to the end.
                    if 0, no sequence after the request will be included.
    if the number is larger than
        the actual length of the sequence before(after) the request,
        it will only take the available length.
    '''
  assert (num_b4_request is None) or num_b4_request >= 0
  assert (num_af_request is None) or num_af_request >= 0

  request_idx = 0
  for s_idx, a_idx in full_trajectory:
    if env.is_initiated_state(s_idx):
      break
    request_idx += 1

  len_trj = len(full_trajectory)
  start_idx = (max(request_idx -
                   num_b4_request, 0) if num_b4_request is not None else 0)
  end_idx = (min(request_idx + num_af_request, len_trj)
             if num_af_request is not None else len_trj)

  return full_trajectory[start_idx:end_idx], request_idx


def generate_multiple_sequences(env: RequestEnvironment, dir_path, num,
                                file_prefix):
  for dummy_i in range(num):
    np_init_p_state = env.get_initial_state_dist()
    state_choice = np.random.choice(np_init_p_state[:, 1],
                                    1,
                                    p=np_init_p_state[:, 0])
    init_state_idx = state_choice[0].astype(np.int32)

    sec, msec = divmod(time.time() * 1000, 1000)
    time_stamp = '%s.%03d' % (time.strftime('%Y-%m-%d_%H_%M_%S',
                                            time.gmtime(sec)), msec)
    file_name = (file_prefix + time_stamp + '.txt')
    file_path = os.path.join(dir_path, file_name)
    env.generate_sequence(init_state_idx,
                          timeout=1000,
                          save=True,
                          file_name=file_path)


def alignment_prediction_accuracy(conf):
  count_all = 0
  count_align_correct = 0

  ALIGNED = [(0, 0), (1, 1)]

  for key_true in conf:
    for key_inf in conf[key_true]:
      count_all += conf[key_true][key_inf]
      if key_true in ALIGNED:
        if key_inf in ALIGNED:
          count_align_correct += conf[key_true][key_inf]
      else:
        if key_inf not in ALIGNED:
          count_align_correct += conf[key_true][key_inf]

  return count_align_correct / count_all


def get_bayesian_infer_result(num_agent, list_np_policies, num_lstate,
                              test_full_trajectories, test_part_trajectories,
                              true_latent_labels):

  full_conf = {}
  full_conf[(0, 0)] = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
  full_conf[(0, 1)] = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
  full_conf[(1, 0)] = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
  full_conf[(1, 1)] = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
  part_conf = {}
  part_conf[(0, 0)] = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
  part_conf[(0, 1)] = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
  part_conf[(1, 0)] = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
  part_conf[(1, 1)] = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}

  tuple_num_lstate = tuple(num_lstate for dummy_i in range(num_agent))

  def n_xsa_policy(agent_idx, x_idx, state_idx, joint_action):
    action_idx = joint_action[agent_idx]
    return list_np_policies[agent_idx][x_idx][state_idx][action_idx]

  full_count_correct = 0
  for idx, trj in enumerate(test_full_trajectories):
    infer_lat = bayesian_mental_state_inference(trj, tuple_num_lstate,
                                                n_xsa_policy, num_agent)
    true_lat = true_latent_labels[idx]
    full_conf[true_lat][infer_lat] += 1
    if true_lat == infer_lat:
      full_count_correct += 1
  full_acc = full_count_correct / len(test_full_trajectories) * 100
  full_align_acc = alignment_prediction_accuracy(full_conf) * 100

  part_count_correct = 0
  for idx, trj in enumerate(test_part_trajectories):
    infer_lat = bayesian_mental_state_inference(trj, tuple_num_lstate,
                                                n_xsa_policy, num_agent)
    true_lat = true_latent_labels[idx]
    part_conf[true_lat][infer_lat] += 1
    if true_lat == infer_lat:
      part_count_correct += 1
  part_acc = part_count_correct / len(test_part_trajectories) * 100
  part_align_acc = alignment_prediction_accuracy(part_conf) * 100

  return (full_conf, part_conf, full_acc, part_acc, full_align_acc,
          part_align_acc)


if __name__ == "__main__":
  ##############################################
  # RESULT OPTIONS

  LOAD_TASK_MODEL = True
  SL_VAR = True
  SEMI_VAR = True
  GEN_TRAIN_SET = False
  GEN_TEST_SET = False

  ##############################################

  if LOAD_TASK_MODEL:
    data_dir = os.path.join(os.path.dirname(__file__),
                            "data/tooldelivery_v3_train_data/")
    file_prefix = 'td3_train_'
    tooldelivery_env = ToolDeliveryEnv_V3()
    num_agents = tooldelivery_env.num_brains
    num_ostates = tooldelivery_env.mmdp.num_states
    tuple_num_actions = (tooldelivery_env.mmdp.aCN_space.num_actions,
                         tooldelivery_env.mmdp.aSN_space.num_actions)
    possible_lstates = tooldelivery_env.policy.get_possible_latstate_indices()
    num_lstates = len(possible_lstates)

    if GEN_TRAIN_SET:
      file_names = glob.glob(os.path.join(data_dir, file_prefix + '*.txt'))
      for fmn in file_names:
        os.remove(fmn)

      # to generate sequences comment out below lines
      generate_multiple_sequences(tooldelivery_env,
                                  data_dir,
                                  1000,
                                  file_prefix=file_prefix)

    file_names = glob.glob(os.path.join(data_dir, '*.txt'))

    trajectories = []
    latent_labels = []
    count = 0
    num_labeled1 = 100
    SEMISUPER_HYPERPARAM = 1.5
    SUPER_HYPERPARAM = SEMISUPER_HYPERPARAM
    for file_nm in file_names:
      trj, true_lat = read_sample(file_nm)
      if true_lat[0] is None or true_lat[1] is None:
        continue

      trj_n = []
      for sidx, aidx in trj:
        if aidx < 0:
          break
        aCN, aSN, _ = tooldelivery_env.mmdp.np_idx_to_action[aidx]
        trj_n.append((sidx, (aCN, aSN)))

      partial_trj, request_idx = get_tooldelivery_partial_traj(
          tooldelivery_env, trj_n, num_b4_request=0, num_af_request=None)

      trajectories.append(partial_trj)
      latent_labels.append(true_lat)
      count += 1

    print(len(trajectories))

    ##############################################
    # test data
    test_dir = os.path.join(os.path.dirname(__file__),
                            "data/tooldelivery_v3_test_data/")
    test_file_prefix = 'td3_test_'

    if GEN_TEST_SET:
      file_names = glob.glob(os.path.join(test_dir, test_file_prefix + '*.txt'))
      for fmn in file_names:
        os.remove(fmn)

      generate_multiple_sequences(tooldelivery_env,
                                  test_dir,
                                  300,
                                  file_prefix=test_file_prefix)

    test_file_names = glob.glob(os.path.join(test_dir, '*.txt'))

    test_full_trajectories = []
    test_part_trajectories = []
    true_latent_labels = []
    for file_nm in test_file_names:
      trj, true_lat = read_sample(file_nm)
      if true_lat[0] is None or true_lat[1] is None:
        continue

      trj_n = []
      for sidx, aidx in trj:
        aCN, aSN, _ = tooldelivery_env.mmdp.np_idx_to_action[aidx]
        trj_n.append((sidx, (aCN, aSN)))

      full_trj, request_idx = get_tooldelivery_partial_traj(tooldelivery_env,
                                                            trj_n,
                                                            num_b4_request=0,
                                                            num_af_request=None)
      partial_trj, request_idx = get_tooldelivery_partial_traj(tooldelivery_env,
                                                               trj_n,
                                                               num_b4_request=0,
                                                               num_af_request=7)

      test_full_trajectories.append(full_trj)
      test_part_trajectories.append(partial_trj)
      true_latent_labels.append(true_lat)

    print(len(test_full_trajectories))

    full_acc_history = []
    part_acc_history = []

    def accuracy_history(num_agent, pi_hyper):
      list_np_policies = [None for dummy_i in range(num_agents)]
      # if np.isnan(pi_hyper[0].sum()):
      #   print("Nan acc_hist 1-1")
      for idx in range(num_agents):
        numerator = pi_hyper[idx] - 1
        action_sums = np.sum(numerator, axis=2)
        list_np_policies[idx] = numerator / action_sums[:, :, np.newaxis]
      (_, _, full_acc, part_acc, full_align_acc,
       part_align_acc) = get_bayesian_infer_result(num_agent, list_np_policies,
                                                   num_lstates,
                                                   test_full_trajectories,
                                                   test_part_trajectories,
                                                   true_latent_labels)
      # full_acc_history.append(full_acc)
      # part_acc_history.append(part_acc)
      full_acc_history.append(full_align_acc)
      part_acc_history.append(part_align_acc)

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

  ##############################################
  # supervised policy learning
  if SL_VAR:
    sup_infer1 = BTILStatic([], trajectories[0:num_labeled1],
                            latent_labels[0:num_labeled1], num_agents,
                            num_ostates, num_lstates, tuple_num_actions)
    sup_infer1.set_dirichlet_prior(SUPER_HYPERPARAM)
    sup_infer1.do_inference()
    sup_np_policy1 = sup_infer1.list_np_policy

    (sup_conf_full1, sup_conf_part1, full_acc1, part_acc1, full_align_acc1,
     part_align_acc1) = get_bayesian_infer_result(num_agents, sup_np_policy1,
                                                  num_lstates,
                                                  test_full_trajectories,
                                                  test_part_trajectories,
                                                  true_latent_labels)

    sup_infer2 = BTILStatic([], trajectories, latent_labels, num_agents,
                            num_ostates, num_lstates, tuple_num_actions)
    sup_infer2.set_dirichlet_prior(SUPER_HYPERPARAM)
    sup_infer2.do_inference()
    sup_np_policy2 = sup_infer2.list_np_policy
    (sup_conf_full2, sup_conf_part2, full_acc2, part_acc2, full_align_acc2,
     part_align_acc2) = get_bayesian_infer_result(num_agents, sup_np_policy2,
                                                  num_lstates,
                                                  test_full_trajectories,
                                                  test_part_trajectories,
                                                  true_latent_labels)
  # ##############################################
  # # semisupervised policy learning
  if SEMI_VAR:
    semisup_infer = BTILStatic(trajectories[num_labeled1:len(trajectories)],
                               trajectories[0:num_labeled1],
                               latent_labels[0:num_labeled1],
                               num_agents,
                               num_ostates,
                               num_lstates,
                               tuple_num_actions,
                               max_iteration=100,
                               epsilon=0.001)

    semisup_infer.set_dirichlet_prior(SEMISUPER_HYPERPARAM)

    start_time = time.time()
    semisup_infer.do_inference(callback=accuracy_history)
    elapsed_time = time.time() - start_time
    print(elapsed_time)

    semisup_np_policy = semisup_infer.list_np_policy
    (semi_conf_full, semi_conf_part, semi_full_acc, semi_part_acc,
     semi_full_align_acc, semi_part_align_acc) = get_bayesian_infer_result(
         num_agents, semisup_np_policy, num_lstates, test_full_trajectories,
         test_part_trajectories, true_latent_labels)

  ##############################################
  # results
  if SL_VAR:
    print("Full - super1")
    print_conf(sup_conf_full1)
    print("4by4 Acc: " + str(full_acc1))
    print("2by2 Acc: " + str(full_align_acc1))
    print("Part - super1")
    print_conf(sup_conf_part1)
    print("4by4 Acc: " + str(part_acc1))
    print("2by2 Acc: " + str(part_align_acc1))

    print("Full - super2")
    print_conf(sup_conf_full2)
    print("4by4 Acc: " + str(full_acc2))
    print("2by2 Acc: " + str(full_align_acc2))
    print("Part - super2")
    print_conf(sup_conf_part2)
    print("4by4 Acc: " + str(part_acc2))
    print("2by2 Acc: " + str(part_align_acc2))

  if SEMI_VAR:
    print("Full - semi")
    print_conf(semi_conf_full)
    print("4by4 Acc: " + str(semi_full_acc))
    print("2by2 Acc: " + str(semi_full_align_acc))
    print("Part - semi")
    print_conf(semi_conf_part)
    print("4by4 Acc: " + str(semi_part_acc))
    print("2by2 Acc: " + str(semi_part_align_acc))

    fig = plt.figure(figsize=(7.2, 3))
    # str_title = (
    #     "hyperparam: " + str(SEMISUPER_HYPERPARAM) +
    #     ", # labeled: " + str(len(trajectories)) +
    #     ", # unlabeled: " + str(len(unlabeled_traj)))
    str_title = ("hyperparameter u: " + str(SEMISUPER_HYPERPARAM))
    # fig.suptitle(str_title)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.grid(True)
    ax2.grid(True)
    ax1.plot(full_acc_history,
             '.-',
             label="SemiSL",
             clip_on=False,
             fillstyle='none')
    if SL_VAR:
      ax1.axhline(y=full_align_acc1, color='r', linestyle='-', label="SL-Small")
      ax1.axhline(y=full_align_acc2, color='g', linestyle='-', label="SL-Large")
    FONT_SIZE = 16
    TITLE_FONT_SIZE = 12
    LEGENT_FONT_SIZE = 12
    ax1.set_ylabel("Accuracy (%)", fontsize=FONT_SIZE)
    ax1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    # ax1.set_ylim([70, 100])
    # ax1.set_xlim([0, 16])
    ax1.set_title("Full Sequence", fontsize=TITLE_FONT_SIZE)

    ax2.plot(part_acc_history,
             '.-',
             label="SemiSL",
             clip_on=False,
             fillstyle='none')
    if SL_VAR:
      ax2.axhline(y=part_align_acc1, color='r', linestyle='-', label="SL-Small")
      ax2.axhline(y=part_align_acc2, color='g', linestyle='-', label="SL-Large")
    ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    # ax2.set_ylim([50, 80])
    # ax2.set_xlim([0, 16])
    ax2.set_title("Partial Sequence (5 Steps)", fontsize=TITLE_FONT_SIZE)
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles,
               labels,
               loc='center right',
               prop={'size': LEGENT_FONT_SIZE})
    fig.text(0.45, 0.04, 'Iteration', ha='center', fontsize=FONT_SIZE)
    fig.tight_layout(pad=2.0)
    fig.subplots_adjust(right=0.8, bottom=0.2)
    plt.show()
