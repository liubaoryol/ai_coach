import glob, os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
import random
import pickle
import time
from bayesian_team_eval.make_results import get_tooldelivery_partial_traj
from bayesian_team_eval.internals.bayesian_inference import read_sample
from bayesian_team_eval.tooldelivery_v3.tooldelivery_v3_env import (
    ToolDeliveryEnv_V3
)
import bayesian_policy_infer
from reduced_mdp_tool_delivery_v3 import Reduced_MDP_Tool_Delivery_v3, ToolLocNew
from test_irl import conv_traj_to_reduced_mdp
rc('font',**{'family':'serif','serif':['Palatino']})
rc('grid', linestyle=":", color='grey')

def infer_inidivial_latentstate(trajectory, policy, xstate_indices, i_brain):
    dict_xidx_idx = {}
    for idx, xidx in enumerate(xstate_indices):
        dict_xidx_idx[xidx] = idx

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
            lstate = xstate_indices[idx]
            a_idx = joint_action[i_brain]
            p_a_sx = policy[i_brain][state_idx][lstate][a_idx]
            np_px[idx] = np_px[idx] * p_a_sx
            np_log_px[idx] += np.log([p_a_sx])[0]
    
    # print("np_prior_indv")
    # print(np_prior)
    # print("np_px_indv")
    # print(np_px)
    # print("np_log_prior")
    # print(np_log_prior)
    # print("np_log_px")
    # print(np_log_px)
    np_px = np_px * np_prior
    np_log_px = np_log_px + np_log_prior
    # sum_p = np.sum(np_px)
    # if sum_p == 0.0:
    #     print("sum of probability is 0")
    #     return None

    # np_px = np_px / sum_p
    # index = np.argmax(np_px)
    list_same_idx = np.argwhere(np_log_px == np.max(np_log_px))
    # if len(list_same_idx) > 1:
    #     print("same")
    # index = np.argmax(np_log_px)
    return xstate_indices[random.choice(list_same_idx)[0]]

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


def get_bayesian_infer_result(
    num_agent, list_np_policies,
    possible_lstates,
    test_full_trajectories,
    test_part_trajectories,
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

    full_count_correct = 0
    for idx, trj in enumerate(test_full_trajectories):
        infer_lat = bayesian_inference(
            trj, list_np_policies, possible_lstates, num_agent)
        true_lat = true_latent_labels[idx]
        full_conf[true_lat][infer_lat] += 1
        if true_lat == infer_lat:
            full_count_correct += 1
    full_acc = full_count_correct / len(test_full_trajectories) * 100
    full_align_acc = alignment_prediction_accuracy(full_conf) * 100

    part_count_correct = 0
    for idx, trj in enumerate(test_part_trajectories):
        infer_lat = bayesian_inference(
            trj, list_np_policies, possible_lstates, num_agent)
        true_lat = true_latent_labels[idx]
        part_conf[true_lat][infer_lat] += 1
        if true_lat == infer_lat:
            part_count_correct += 1
    part_acc = part_count_correct / len(test_part_trajectories) * 100
    part_align_acc = alignment_prediction_accuracy(part_conf) *100

    return full_conf, part_conf, full_acc, part_acc, full_align_acc, part_align_acc



if __name__ == "__main__":
    ##############################################
    ##############################################
    # RESULT OPTIONS

    load_task_model = False
    do_sup_infer = False
    do_semi_infer = False
    do_irl_infer = False
    do_semi_plot = True

    ##############################################
    ##############################################

    if load_task_model:
        data_dir = './tooldelivery_v3_data/'
        file_prefix = 'td3_data_'
        tooldelivery_env = ToolDeliveryEnv_V3()
        num_agents = tooldelivery_env.num_brains
        num_ostates = tooldelivery_env.mmdp.num_states
        tuple_num_actions = (
            tooldelivery_env.mmdp.aCN_space.num_actions,
            tooldelivery_env.mmdp.aSN_space.num_actions)
        # num_actions = tooldelivery_env.mmdp.num_actions
        possible_lstates = tooldelivery_env.policy.get_possible_latstate_indices()
        num_lstates = len(possible_lstates)

        # file_names = glob.glob(os.path.join(data_dir, file_prefix + '*.txt'))
        # for fmn in file_names:
        #     os.remove(fmn)

        # to generate sequences comment out below lines
        # generate_multiple_sequences(
        #     tooldelivery_env, data_dir, 1000,
        #     file_prefix=file_prefix)

        file_names = glob.glob(os.path.join(data_dir, '*.txt'))

        trajectories = []
        latent_labels = []
        # unlabeled_traj = []
        count = 0
        num_labeled1 = 100
        SEMISUPER_HYPERPARAM = 1.5
        SUPER_HYPERPARAM = SEMISUPER_HYPERPARAM
        for file_nm in file_names:
            # print(file_nm)
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
                tooldelivery_env, trj_n,
                num_b4_request=0,
                num_af_request=None)

            # if count < num_labeled:
            trajectories.append(partial_trj)
            latent_labels.append(true_lat)
            # elif count < num_labeled + num_unlabeled:
            #     unlabeled_traj.append(partial_trj)
            # else:
            #     break
            count += 1

        print(len(trajectories))
        # print(len(unlabeled_traj))

        ##############################################
        # test data
        # test_dir = './tooldelivery_v3_test/'
        # test_file_prefix = 'td3_test_'
        test_dir = '../bayesian_team_eval/tooldelivery_trajectories3/'
        test_file_prefix = 'td3_sequence_'
        # generate_multiple_sequences(
        #     tooldelivery_env, test_dir, 300,
        #     file_prefix=test_file_prefix)

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

            full_trj, request_idx = get_tooldelivery_partial_traj(
                tooldelivery_env, trj_n,
                num_b4_request=0,
                num_af_request=None)
            partial_trj, request_idx = get_tooldelivery_partial_traj(
                tooldelivery_env, trj_n,
                num_b4_request=0,
                num_af_request=5)

            test_full_trajectories.append(full_trj)
            test_part_trajectories.append(partial_trj)
            true_latent_labels.append(true_lat)


        full_acc_history = []
        part_acc_history = []
        def accuracy_history(num_agent, pi_hyper):
            list_np_policies = [None for dummy_i in range(num_agents)]
            for idx in range(num_agents):
                numerator = pi_hyper[idx] - 1
                action_sums = np.sum(numerator, axis=2)
                list_np_policies[idx] = numerator / action_sums[:, :, np.newaxis]
            _, _, full_acc, part_acc, full_align_acc, part_align_acc = get_bayesian_infer_result(
                num_agent,
                list_np_policies,
                possible_lstates,
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
    if do_sup_infer:
        sup_infer1 = bayesian_policy_infer.supervised_bayesian_policy_learning(
            trajectories[0:num_labeled1], latent_labels[0:num_labeled1],
            num_agents, num_ostates, num_lstates, tuple_num_actions)
        sup_infer1.set_dirichlet_prior(SUPER_HYPERPARAM)
        sup_infer1.do_inference()
        sup_np_policy1 = sup_infer1.np_policy

        (
            sup_conf_full1, sup_conf_part1, full_acc1, part_acc1,
            full_align_acc1, part_align_acc1
            ) = get_bayesian_infer_result(
                num_agents,
                sup_np_policy1,
                possible_lstates,
                test_full_trajectories,
                test_part_trajectories,
                true_latent_labels)

        sup_infer2 = bayesian_policy_infer.supervised_bayesian_policy_learning(
            trajectories, latent_labels,
            num_agents, num_ostates, num_lstates, tuple_num_actions)
        sup_infer2.set_dirichlet_prior(SUPER_HYPERPARAM)
        sup_infer2.do_inference()
        sup_np_policy2 = sup_infer2.np_policy
        (
            sup_conf_full2, sup_conf_part2, full_acc2, part_acc2,
            full_align_acc2, part_align_acc2
            ) = get_bayesian_infer_result(
                num_agents,
                sup_np_policy2,
                possible_lstates,
                test_full_trajectories,
                test_part_trajectories,
                true_latent_labels)
    # ##############################################
    # # semisupervised policy learning
    if do_semi_infer:
        semisup_infer = bayesian_policy_infer.semisupervised_bayesian_policy_learning(
            trajectories[num_labeled1:len(trajectories)], trajectories[0:num_labeled1],
            latent_labels[0:num_labeled1], num_agents, num_ostates,
            num_lstates, tuple_num_actions, iteration=100, epsilon=0.001)
        
        semisup_infer.set_dirichlet_prior(SEMISUPER_HYPERPARAM)

        start_time = time.time()
        semisup_infer.do_inference(callback=accuracy_history)
        elapsed_time = time.time() - start_time
        print(elapsed_time)

        semisup_np_policy = semisup_infer.np_policy
        (
            semi_conf_full, semi_conf_part, semi_full_acc, semi_part_acc,
            semi_full_align_acc, semi_part_align_acc
            ) = get_bayesian_infer_result(
            num_agents,
            semisup_np_policy,
            possible_lstates,
            test_full_trajectories,
            test_part_trajectories,
            true_latent_labels)

    ##############################################
    # results
    if do_sup_infer:
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

    if do_semi_infer:
        print("Full - semi")
        print_conf(semi_conf_full)
        print("4by4 Acc: " + str(semi_full_acc))
        print("2by2 Acc: " + str(semi_full_align_acc))
        print("Part - semi")
        print_conf(semi_conf_part)
        print("4by4 Acc: " + str(semi_part_acc))
        print("2by2 Acc: " + str(semi_part_align_acc))

        fig = plt.figure(figsize=(7.2,3))
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
        ax1.plot(full_acc_history, '.-', label="SemiSL", clip_on=False, fillstyle='none')
        if do_sup_infer:
            ax1.axhline(y=full_align_acc1, color='r', linestyle='-', label="SL-Small")
            ax1.axhline(y=full_align_acc2, color='g', linestyle='-', label="SL-Large")
        FONT_SIZE = 16
        TITLE_FONT_SIZE = 12
        LEGENT_FONT_SIZE = 12
        # ax1.set_xlabel("Iteration", fontsize=FONT_SIZE)
        ax1.set_ylabel("Accuracy (%)", fontsize=FONT_SIZE)
        ax1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax1.set_ylim([70, 100])
        ax1.set_xlim([0, 16])
        ax1.set_title("Full Sequence", fontsize=TITLE_FONT_SIZE)
        # ax1.legend(frameon=False, loc='lower right', prop={'size': LEGENT_FONT_SIZE})

        ax2.plot(part_acc_history, '.-', label="SemiSL", clip_on=False, fillstyle='none')
        if do_sup_infer:
            ax2.axhline(y=part_align_acc1, color='r', linestyle='-', label="SL-Small")
            ax2.axhline(y=part_align_acc2, color='g', linestyle='-', label="SL-Large")
        # ax2.set_xlabel("Iteration", fontsize=FONT_SIZE)
        # ax2.set_ylabel("Accuracy (%)", fontsize=FONT_SIZE)
        ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax2.set_ylim([50, 80])
        ax2.set_xlim([0, 16])
        ax2.set_title("Partial Sequence (5 Steps)", fontsize=TITLE_FONT_SIZE)
        handles, labels = ax2.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', prop={'size': LEGENT_FONT_SIZE})
        fig.text(0.45, 0.04, 'Iteration', ha='center', fontsize=FONT_SIZE)
        # ax2.legend(frameon=False, loc='lower right', prop={'size': LEGENT_FONT_SIZE})
        fig.tight_layout(pad=2.0)
        fig.subplots_adjust(right=0.8, bottom=0.2)   
        plt.show() 

    if do_semi_plot:
        FONT_SIZE = 16
        TITLE_FONT_SIZE = 12
        LEGENT_FONT_SIZE = 12

        fig = plt.figure(figsize=(7, 3))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        hyper_values = [1.01, 1.1, 1.2, 1.5, 2, 3, 5, 10, 50, 100 ]
        tick_values = [1.01,1.5,  3, 5, 10, 50, 100 ]
        full_acc_values = [
            95.20295203, 97.41697417, 97.78597786, 97.78597786, 97.78597786,
            97.04797048, 92.6199262, 87.82287823, 69.37269373, 59.4095941]
        partial_acc_values = [
            71.58671587, 73.06273063, 73.80073801, 74.1697417, 73.06273063,
            73.06273063, 73.43173432, 73.80073801, 60.14760148, 55.35055351]

        ax1.plot(hyper_values, full_acc_values, '.-', clip_on=False, fillstyle='none')
        ax1.set_title("Full Sequence", fontsize=TITLE_FONT_SIZE)
        ax1.set_ylabel("Accuracy (%)", fontsize=FONT_SIZE)
        # ax1.set_xlabel("Hyperparameter u", fontsize=FONT_SIZE)
        ax1.set_xscale("log")
        ax1.set_xticks(tick_values)
        ax1.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        ax1.get_xaxis().set_minor_locator(matplotlib.ticker.FixedLocator([1.1, 1.2, 2]))
        ax1.get_xaxis().set_minor_formatter(matplotlib.ticker.FormatStrFormatter(""))
        # ax1.get_xaxis().set_tick_params(which='minor', size=0)
        # ax1.get_xaxis().set_tick_params(which='minor', width=0) 
        ax1.set_ylim([50, 100])
        ax2.set_ylim([50, 100])
        ax1.grid(True)
        ax2.grid(True)
        fig.text(0.5, 0.04, 'Hyperparameter $u$', ha='center', fontsize=FONT_SIZE)

        ax2.plot(hyper_values, partial_acc_values, '.-', clip_on=False, fillstyle='none')
        ax2.set_title("Partial Sequence (5 Steps)", fontsize=TITLE_FONT_SIZE)
        # ax2.set_ylabel("Accuracy (%)", fontsize=FONT_SIZE)
        # ax2.set_xlabel("Hyperparameter u", fontsize=FONT_SIZE)
        ax2.set_xscale("log")
        ax2.set_xticks(tick_values)
        ax2.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        ax2.get_xaxis().set_minor_locator(matplotlib.ticker.FixedLocator([1.1, 1.2, 2]))
        ax2.get_xaxis().set_minor_formatter(matplotlib.ticker.FormatStrFormatter(""))
        # ax2.get_xaxis().set_tick_params(which='minor', size=0)
        # ax2.get_xaxis().set_tick_params(which='minor', width=0) 

        fig.tight_layout(pad=2.0)
        fig.subplots_adjust(bottom=0.2)   
        plt.show() 

 
    # ##############################################
    # # IRL  
    if do_irl_infer:
        # list_weights = []
        # with open('irl_weights_0.pickle', 'rb') as f:
        #     list_weights.append(pickle.load(f))
        # with open('irl_weights_1.pickle', 'rb') as f:
        #     list_weights.append(pickle.load(f))
        list_policy = []
        with open('irl_policy_0.pickle', 'rb') as f:
            list_policy.append(pickle.load(f))
        with open('irl_policy_1.pickle', 'rb') as f:
            list_policy.append(pickle.load(f))

        mdp_reduced = Reduced_MDP_Tool_Delivery_v3()

        test_full_trajectories_reduced = []
        test_part_trajectories_reduced = []
        for idx in range(len(test_full_trajectories)):
            traj_full = test_full_trajectories[idx]
            traj_part = test_part_trajectories[idx]
            test_full_trajectories_reduced.append(
                conv_traj_to_reduced_mdp(tooldelivery_env.mmdp, mdp_reduced, traj_full))
            test_part_trajectories_reduced.append(
                conv_traj_to_reduced_mdp(tooldelivery_env.mmdp, mdp_reduced, traj_part))
        
        tuple_num_actions_re = (
            mdp_reduced.aCN_space.num_actions,
            mdp_reduced.aSN_space.num_actions)

        # joint policy to individual policy
        irl_np_policy = []
        for idx in range(num_agents):
            irl_np_policy.append(
                np.zeros((mdp_reduced.num_states, num_lstates, tuple_num_actions_re[idx])))
        
        for x_idx in range(num_lstates):
            for a_idx in range(mdp_reduced.num_actions):
                a_cn_i, a_sn_i = mdp_reduced.np_idx_to_action[a_idx]
                irl_np_policy[0][:, x_idx, a_cn_i] += list_policy[x_idx][:, a_idx]
                irl_np_policy[1][:, x_idx, a_sn_i] += list_policy[x_idx][:, a_idx]

        irl_conf_full, irl_conf_part, irl_acc_full, irl_acc_part, irl_align_acc_full, irl_align_acc_part = (
            get_bayesian_infer_result(
                num_agents,
                irl_np_policy,
                possible_lstates,
                test_full_trajectories_reduced,
                test_part_trajectories_reduced,
                true_latent_labels))

        print("Full - IRL")
        print_conf(irl_conf_full)
        print(irl_acc_full)
        print(irl_align_acc_full)
        print("Part - IRL")
        print_conf(irl_conf_part)
        print(irl_acc_part)
        print(irl_align_acc_part)



    # ordered_key = [(0, 0), (1, 1), (0, 1), (1, 0)]
    # count_all = 0
    # sum_corrent = 0
    # for key1 in ordered_key:
    #     print(key1)
    #     txt_pred = ""
    #     for key2 in ordered_key:
    #         txt_pred = txt_pred + str(key2) + ": " + str(conf[key1][key2]) + ", "
    #         count_all += conf[key1][key2]
    #         if key1 == key2:
    #             sum_corrent += conf[key1][key2]
    #     print(txt_pred)
    
    # print(sum_corrent / count_all)
    
    

    
