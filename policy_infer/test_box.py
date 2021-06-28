import glob, os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from policy_infer.bayesian_policy_infer import (
    semisupervised_bayesian_policy_learning,
    supervised_bayesian_policy_learning)
from generate_policy.policy_utils import get_feature_state_indv_v2
from generate_policy.mdp_moving import MDPMovingLuggage_V2
from moving_luggage.hand_policies import (
    get_qlearn_numpy_policy_action, get_qlearn_numpy_policy_dist)
from moving_luggage.simulator import Simulator
from moving_luggage.constants import (
    AgentActions, KEY_AGENTS, KEY_BAGS, LATENT_HEAVY_BAGS, LATENT_LIGHT_BAGS,
    NUM_X_GRID, NUM_Y_GRID)
rc('font',**{'family':'serif','serif':['Palatino']})
rc('grid', linestyle=":", color='grey')

def generate_sequence(game, mdp, latents, goals, beta):
    env_id = 0
    game.finish_game(env_id)
    game.add_new_env(env_id, int(NUM_X_GRID * NUM_Y_GRID / 4))
    env = game.map_id_env[env_id]

    np_bags = env[KEY_BAGS]
    agent1 = env[KEY_AGENTS][0]
    agent2 = env[KEY_AGENTS][1]

    trajectory1 = []
    trajectory2 = []
    s_cur = (np_bags, agent1.coord, agent2.coord, agent1.hold, agent2.hold)
    count = 0
    while True:
        if game.is_finished(env_id):
            break
        count +=1

        a1_state = get_feature_state_indv_v2(s_cur, 0, game.goal_pos)
        a2_state = get_feature_state_indv_v2(s_cur, 1, game.goal_pos)
        a1_sidx = mdp.np_state_to_idx[a1_state]
        a2_sidx = mdp.np_state_to_idx[a2_state]

        action1 = get_qlearn_numpy_policy_action(
            env, 0, latents[0], goals, beta=beta, mdp_env=mdp)
        action2 = get_qlearn_numpy_policy_action(
            env, 1, latents[1], goals, beta=beta, mdp_env=mdp)
        a1_aidx = mdp.np_action_to_idx[action1.value]
        a2_aidx = mdp.np_action_to_idx[action2.value]
        trajectory1.append((a1_sidx, a1_aidx))
        trajectory2.append((a2_sidx, a2_aidx))

        game._take_simultaneous_step(
            env, AgentActions(action1), AgentActions(action2))

        np_bags_nxt = env[KEY_BAGS]
        agent1_nxt = env[KEY_AGENTS][0]
        agent2_nxt = env[KEY_AGENTS][1]
        s_nxt = (
            np_bags_nxt,
            agent1_nxt.coord, agent2_nxt.coord,
            agent1_nxt.hold, agent2_nxt.hold)

        s_cur = s_nxt

    return trajectory1, trajectory2

def save_sequence(file_name, traj1, traj2, latents):
    dir_path = os.path.dirname(file_name)
    if dir_path != '' and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(file_name, 'w', newline='') as txtfile:
        # header
        txtfile.write('# latent states\n')
        txtfile.write(', '.join(str(i) for i in latents))
        txtfile.write('\n')
        # sequence
        txtfile.write('# state action sequence\n')
        for idx in range(len(traj1)):
            sidx_a1, aidx_a1 = traj1[idx]
            sidx_a2, aidx_a2 = traj2[idx]
            txtfile.write('%d, %d, %d, %d' % 
            (sidx_a1, aidx_a1, sidx_a2, aidx_a2))
            txtfile.write('\n')

def read_sample(file_name):
    traj1 = []
    traj2 = []
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
            row_elem = [int(elem) for elem in line.rstrip().split(", ")]
            traj1.append((row_elem[0], row_elem[1]))
            traj2.append((row_elem[2], row_elem[3]))
    return traj1, traj2, tuple(latents)

def generate_multiple_samples(game, mdp, dir_path, num, file_prefix, beta):
    for dummy in range(num):
        a1_ls = random.choice([LATENT_HEAVY_BAGS, LATENT_LIGHT_BAGS])
        a2_ls = random.choice([LATENT_HEAVY_BAGS, LATENT_LIGHT_BAGS])
        latents = (a1_ls, a2_ls)
        traj1, traj2 = generate_sequence(
            game, mdp, latents, game.goal_pos, beta)

        sec, msec = divmod(time.time() * 1000, 1000)
        time_stamp = '%s.%03d' % (
            time.strftime('%Y-%m-%d_%H_%M_%S', time.gmtime(sec)), msec)
        file_name = (file_prefix + time_stamp + '.txt')
        file_path = os.path.join(dir_path, file_name)
        save_sequence(file_path, traj1, traj2, latents)
        print(dummy)


def infer_inidivial_latentstate(trajectory, policy, xstate_indices, i_brain):
    num_xstate = len(xstate_indices)
    np_px = np.ones((num_xstate,))
    np_prior = np.zeros((num_xstate,))
    for idx in range(num_xstate):
        np_prior[idx] = 1.0 / num_xstate

    np_log_prior = np.log(np_prior)
    np_log_px = np.zeros((num_xstate,))
    if len(trajectory) < 1:
        print("Empty trajectory")
        return None
    
    for idx in range(num_xstate):
        for state_idx, joint_action in trajectory:
            a_idx = joint_action
            p_a_sx = policy[i_brain][state_idx][idx][a_idx]
            np_px[idx] = np_px[idx] * p_a_sx
            np_log_px[idx] += np.log([p_a_sx])[0]
    
    np_px = np_px * np_prior
    np_log_px = np_log_px + np_log_prior

    index = np.argmax(np_log_px)
    return xstate_indices[index]

def bayesian_inference(trajectory, policy, xstate_indices, num_agents):
    list_latstates = []
    for i_b in range(num_agents):
        lat = infer_inidivial_latentstate(trajectory, policy, xstate_indices, i_b)
        list_latstates.append(lat)

    return tuple(list_latstates)


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

def alignment_prediction_recall(conf):
    count_p = 0
    count_tp = 0

    MISALIGNED = [(0, 1), (1, 0)]

    for key_true in conf:
        for key_inf in conf[key_true]:
            if key_true in MISALIGNED:
                count_p += conf[key_true][key_inf]
                if key_inf in MISALIGNED:
                    count_tp += conf[key_true][key_inf]
    
    return count_tp / count_p

def get_bayesian_infer_result_box(
    list_np_policies_1,
    list_np_policies_2,
    possible_lstates,
    test_part_trajectories_1,
    test_part_trajectories_2,
    true_latent_labels):

    part_conf = {}
    part_conf[(0, 0)] = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    part_conf[(0, 1)] = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    part_conf[(1, 0)] = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    part_conf[(1, 1)] = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}

    count_correct = 0
    for idx in range(len(test_part_trajectories_1)):
        infer_ls1 = bayesian_inference(
            test_part_trajectories_1[idx], list_np_policies_1, possible_lstates, 1)
        infer_ls2 = bayesian_inference(
            test_part_trajectories_2[idx], list_np_policies_2, possible_lstates, 1)

        true_lat = true_latent_labels[idx]
        infer_lat = (infer_ls1[0], infer_ls2[0])
        part_conf[true_lat][infer_lat] += 1
        if true_lat == infer_lat:
            count_correct += 1

    part_acc = count_correct / len(test_part_trajectories_1)

    return part_conf, part_acc


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
    

def read_training_set_folder(file_names, num_labeled, num_unlabeled):
    labeled_traj1 = []
    labeled_traj2 = []
    unlabeled_traj1 = []
    unlabeled_traj2 = []
    latent_labels1 = []
    latent_labels2 = []
    count = 0
    avg_seq_len = 0
    for file_nm in file_names:
        # print(file_nm)
        trj1, trj2, true_lat = read_sample(file_nm)
        if true_lat[0] is None or true_lat[1] is None:
            continue

        avg_seq_len += len(trj1)
        if count < num_labeled:
            labeled_traj1.append(trj1)
            labeled_traj2.append(trj2)
            latent_labels1.append(true_lat[0])
            latent_labels2.append(true_lat[1])
        elif count < num_labeled + num_unlabeled:
            unlabeled_traj1.append(trj1)
            unlabeled_traj2.append(trj2)
        else:
            break
        count += 1

    print(avg_seq_len / (len(labeled_traj1) + len(unlabeled_traj1)))
    return (
        labeled_traj1, labeled_traj2,
        unlabeled_traj1, unlabeled_traj2,
        latent_labels1, latent_labels2)


def read_test_set_folder(test_file_names, piece_len):
    test_part_trajectories_1 = []
    test_part_trajectories_2 = []
    true_part_latent_labels = [] 
    count = 0
    for file_nm in test_file_names:
        trj1, trj2, true_lat = read_sample(file_nm)
        if true_lat[0] is None or true_lat[1] is None:
            continue
        len_traj = len(trj1)
        count += len(trj1)

        for idx in range(0,len_traj, piece_len):
            if idx + piece_len > len_traj:
                break

            test_part_trajectories_1.append(trj1[idx:(idx + piece_len)])
            test_part_trajectories_2.append(trj2[idx:(idx + piece_len)])
            true_part_latent_labels.append(true_lat)

    return (
        test_part_trajectories_1,
        test_part_trajectories_2,
        true_part_latent_labels)


def plot_by_each_value(
    list_of_list_accuracy, list_of_list_recall, x_values, list_names):
    # colors = ['b', 'r', 'g', 'c', 'm', 'y']
    colors = ['b', 'r', 'c', 'm', 'y']
    markers =  ['s','v', '^']
    # markers =  ['s','v', 'o', '^']

    fig = plt.figure(figsize=(8,3))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.grid(True)
    ax2.grid(True)
    FONT_SIZE = 14
    TITLE_FONT_SIZE = 12
    LEGENT_FONT_SIZE = 12 
    for idx, list_accuracy in enumerate(list_of_list_accuracy):
        line_style = '-'
        for idx2 in range(idx):
            if list_accuracy == list_of_list_accuracy[idx2]:
                line_style = '--'

        ax1.plot(
            x_values, list_accuracy, label=list_names[idx],
            linestyle=line_style,clip_on=False, fillstyle='none', color=colors[idx], marker=markers[idx])

    ax1.set_ylabel("Accuracy (%)", fontsize=FONT_SIZE)
    ax1.set_xlabel("Sequence Length", fontsize=FONT_SIZE)
    ax1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax1.set_xlim([3, 11])
    xtick_values = [3, 5, 7, 9, 11]
    ax1.set_xticks(xtick_values)
    ax1.set_ylim([94, 100])
    # ytick_values = [65, 70, 80, 90, 100]
    # ax1.set_yticks(ytick_values)
    # ax1.set_title("Accuracy (%)")
    # ax1.legend(loc="lower right",ncol=2, frameon=False)

    for idx, list_recall in enumerate(list_of_list_recall):
        line_style = '-'
        for idx2 in range(idx):
            if list_recall == list_of_list_recall[idx2]:
                line_style = '--'

        ax2.plot(
            x_values, list_recall, label=list_names[idx],
            linestyle=line_style,clip_on=False, fillstyle='none', color=colors[idx], marker=markers[idx])

    # ax2.set_ylabel("%")
    ax2.set_ylabel("Recall (%)", fontsize=FONT_SIZE)
    ax2.set_xlabel("Sequence Length", fontsize=FONT_SIZE)
    ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax2.set_xlim([3, 11])
    ax2.set_xticks(xtick_values)
    ax2.set_ylim([99, 100])
    # ax2.set_yticks(ytick_values)
    # ax2.legend(loc="lower right",ncol=2,frameon=False)
    # ax2.set_title("Recall (%)")
    # fig.text(0.45, 0.04, 'Sequence Length', ha='center', fontsize=FONT_SIZE)
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', prop={'size': LEGENT_FONT_SIZE})
    fig.tight_layout(pad=2.0)
    # fig.subplots_adjust(right=0.8, bottom=0.2)   
    fig.subplots_adjust(right=0.8)
    plt.show() 


if __name__ == "__main__":

    ##############################################
    ##############################################
    # RESULT OPTIONS
    HYPERPARAM = 5
    num_labeled = 100
    num_unlabeled = 90
    piece_len = 11
    load_task_model = False
    do_orig_infer = False
    do_sup_infer = False
    do_semi_infer = False
    plot_final = True
    ##############################################
    if load_task_model:
        data_dir = './mvbag_trajectories/'
        file_prefix = 'mvbag_data_'
        game = Simulator()
        mdp = MDPMovingLuggage_V2()
        num_ostates = mdp.num_states
        num_actions = (mdp.num_actions,)
        possible_latents = [LATENT_HEAVY_BAGS, LATENT_LIGHT_BAGS]
        num_lstates = len(possible_latents)
        game.set_max_step(300)

        BETA = 3
        # # to generate sequences comment out below lines
        # file_names = glob.glob(os.path.join(data_dir, file_prefix + '*.txt'))
        # for fmn in file_names:
        #     os.remove(fmn)
        # generate_multiple_samples(game, mdp, data_dir, 50, file_prefix, BETA)

        file_names = glob.glob(os.path.join(data_dir, '*.txt'))


        (
            labeled_traj1, labeled_traj2,
            unlabeled_traj1, unlabeled_traj2,
            latent_labels1, latent_labels2
        ) = read_training_set_folder(file_names, num_labeled, num_unlabeled)

        print(len(labeled_traj1))
        print(len(unlabeled_traj1))

        ##############################################
        # test sets
        test_dir = './mvbag_trajectories_test/'
        test_file_prefix = 'mvbag_test_'
        # generate_multiple_samples(game, mdp, test_dir, 50, test_file_prefix, BETA)
        test_file_names = glob.glob(os.path.join(test_dir, '*.txt'))

        (
            test_part_trajectories_1, 
            test_part_trajectories_2,
            true_part_latent_labels
        ) = read_test_set_folder(test_file_names, piece_len)

        print("seq_len: " + str(piece_len))
        print(len(test_part_trajectories_1))

    ##############################################
    # true policy inference
    if do_orig_infer:
        list_np_policy = get_qlearn_numpy_policy_dist(BETA, mdp_env=mdp)

        orig_conf, orig_acc = get_bayesian_infer_result_box(
            list_np_policy[0:1],
            list_np_policy[1:2],
            possible_latents,
            test_part_trajectories_1,
            test_part_trajectories_2,
            true_part_latent_labels)
    ##############################################
    # supervised inference
    if do_sup_infer:
        sup_infer1 = supervised_bayesian_policy_learning(
            labeled_traj1, latent_labels1, 1, num_ostates,
            num_lstates, num_actions)

        sup_infer1.set_dirichlet_prior(HYPERPARAM)
        sup_infer1.do_inference()
        sup_np_policy1 = sup_infer1.np_policy

        sup_infer2 = supervised_bayesian_policy_learning(
            labeled_traj2, latent_labels2, 1, num_ostates,
            num_lstates, num_actions)

        sup_infer2.set_dirichlet_prior(HYPERPARAM)
        sup_infer2.do_inference()
        sup_np_policy2 = sup_infer2.np_policy

        sup_conf, sup_acc = get_bayesian_infer_result_box(
            sup_np_policy1,
            sup_np_policy2,
            possible_latents,
            test_part_trajectories_1,
            test_part_trajectories_2,
            true_part_latent_labels)
    ##############################################
    # semisupervised
    if do_semi_infer:
        semi_infer1 = semisupervised_bayesian_policy_learning(
            unlabeled_traj1, labeled_traj1, latent_labels1,
            1, num_ostates, num_lstates, num_actions,
            iteration=100, epsilon=0.001)

        semi_infer1.set_dirichlet_prior(HYPERPARAM)
        semi_infer1.do_inference()
        semi_np_policy1 = semi_infer1.np_policy

        semi_infer2 = semisupervised_bayesian_policy_learning(
            unlabeled_traj2, labeled_traj2, latent_labels2,
            1, num_ostates, num_lstates, num_actions,
            iteration=100, epsilon=0.001)

        semi_infer2.set_dirichlet_prior(HYPERPARAM)
        semi_infer2.do_inference()
        semi_np_policy2 = semi_infer2.np_policy

        semi_conf, semi_acc = get_bayesian_infer_result_box(
            semi_np_policy1,
            semi_np_policy2,
            possible_latents,
            test_part_trajectories_1,
            test_part_trajectories_2,
            true_part_latent_labels)

    ##############################################
    # results
    if do_sup_infer:
        print_conf(sup_conf)
        print(sup_acc)
        print(alignment_prediction_accuracy(sup_conf))
        print(alignment_prediction_recall(sup_conf))

    if do_semi_infer:
        print_conf(semi_conf)
        print(semi_acc)
        print(alignment_prediction_accuracy(semi_conf))
        print(alignment_prediction_recall(semi_conf))

    if do_orig_infer:
        print_conf(orig_conf)
        print(orig_acc)
        print(alignment_prediction_accuracy(orig_conf))
        print(alignment_prediction_recall(orig_conf))

    if plot_final:
        list_of_list_acc = [
            [97.40927208, 98.79328437, 99.51159951, 99.61832061, 99.96110463],
            [95.14369625, 97.53410283, 98.68131868, 99.20483461, 99.37767406],
            # [78.52947346, 81.11227702, 82.97924298, 83.65139949, 84.44185142],
            [95.14369625, 97.53410283, 98.68131868, 99.20483461, 99.37767406]
        ]

        # plot_by_each_value(
        #     list_of_list_acc,
        #     [3, 5, 7, 9, 11],
        #     ["True policy", "Superivsed-100", "Supervised-10", "Semisupervised"],
        #     "Accuracy")

        list_of_list_recall = [
            [99.78723404, 99.96453901, 100, 100, 100],
            [99.17021277, 99.60992908, 99.65363681, 99.74210187, 99.76359338],
            # [68.70212766, 69.5035461, 69.91588323, 70.66408769, 71.00078802],
            [99.17021277, 99.60992908, 99.65363681, 99.74210187, 99.76359338]
        ]

        plot_by_each_value(
            list_of_list_acc,
            list_of_list_recall,
            [3, 5, 7, 9, 11],
            ["TruePolicy", "SL-Large", "SemiSL"])
            # ["TruePolicy", "SL-Large", "SL-Small", "SemiSL"])
