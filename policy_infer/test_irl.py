import glob, os
from os.path import join
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
from bayesian_team_eval.make_results import get_tooldelivery_partial_traj
from bayesian_team_eval.internals.bayesian_inference import read_sample
from bayesian_team_eval.tooldelivery_v3.tooldelivery_v3_env import (
    ToolDeliveryEnv_V3
)
from bayesian_team_eval.tooldelivery_v3.tooldelivery_v3_state_action import (
    ToolLoc, ToolNames, StateAsked, StatePatient
)
from reduced_mdp_tool_delivery_v3 import Reduced_MDP_Tool_Delivery_v3, ToolLocNew
from maxent_irl.maxent_irl import (
    CMaxEntIRL, compute_relative_freq, cal_policy_error)


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
    
    np_px = np_px * np_prior
    np_log_px = np_log_px + np_log_prior
    list_same_idx = np.argwhere(np_log_px == np.max(np_log_px))
    return xstate_indices[random.choice(list_same_idx)[0]]


def bayesian_inference(trajectory, policy, xstate_indices, num_agents):
    list_latstates = []
    for i_b in range(num_agents):
        lat = infer_inidivial_latentstate(trajectory, policy, xstate_indices, i_b)
        list_latstates.append(lat)

    return tuple(list_latstates)


def get_initial_dist(mdp):
    np_init_p_sScalpel_s = np.array(
        [[1., mdp.sScal_space.state_to_idx[ToolLocNew.STORAGE]]])
    np_init_p_sSuture_s = np.array(
        [[1., mdp.sSut_space.state_to_idx[ToolLocNew.CABINET]]])
        # [[1., mdp.dict_sTools_space[ToolNames.SUTURE_S].state_to_idx[ToolLoc.CABINET]]])
    # np_init_p_sCNPos = np.array([
    #     [0.9, mdp.sCNPos_space.state_to_idx[mdp.handover_loc]]]
    #     )
    list_p_cnpos = []
    for pos in mdp.sCNPos_space.statespace:
        if pos == (4, 1):
            list_p_cnpos.append((0.69, mdp.sCNPos_space.state_to_idx[pos]))
        elif pos == (4, 0):
            list_p_cnpos.append((0.1, mdp.sCNPos_space.state_to_idx[pos]))
        elif pos == (4, 2):
            list_p_cnpos.append((0.09, mdp.sCNPos_space.state_to_idx[pos]))
        elif pos == (3, 0):
            list_p_cnpos.append((0.01, mdp.sCNPos_space.state_to_idx[pos]))
        elif pos == (3, 1):
            list_p_cnpos.append((0.09, mdp.sCNPos_space.state_to_idx[pos]))
        elif pos == (3, 2):
            list_p_cnpos.append((0.01, mdp.sCNPos_space.state_to_idx[pos]))
        else:
            list_p_cnpos.append((0.01 / 19, mdp.sCNPos_space.state_to_idx[pos]))
    np_init_p_sCNPos = np.array(list_p_cnpos)
    
    init_prop = np.zeros((mdp.num_states))
    # dict_init_p_state_idx = {}
    for p_sSb, sSb in np_init_p_sScalpel_s:
        for p_sFb, sFb in np_init_p_sSuture_s:
            for p_sPos, sPos in np_init_p_sCNPos:
                init_p = (p_sSb * p_sFb * p_sPos)
                state_idx = mdp.np_state_to_idx[
                    sSb.astype(np.int32),
                    sFb.astype(np.int32),
                    sPos.astype(np.int32)]
                # dict_init_p_state_idx[state_idx] = init_p
                init_prop[state_idx] = init_p

    # np_init_p_state_idx = np.zeros((len(dict_init_p_state_idx), 2))
    # iter_idx = 0
    # for state_idx in dict_init_p_state_idx:
    #     np_next_p = dict_init_p_state_idx.get(state_idx)
    #     np_init_p_state_idx[iter_idx] = np_next_p, state_idx
    #     iter_idx += 1
    

    return init_prop


def conv_traj_to_reduced_mdp(mdp_orig, mdp_reduced, trajectory):
    traj_new = []
    for s_idx, joint_a_idx in trajectory:
        state_vector = mdp_orig.np_idx_to_state[s_idx]
        sScal_p, sSut_p, sScal_s, sSut_s, sPat, sCNPos, sAsk = state_vector 

        s_idx_r = mdp_reduced.np_state_to_idx[sScal_s, sSut_s, sCNPos]

        # action_vector = mdp_orig.np_idx_to_action[a_idx]
        # aCN, aSN = action_vector
        # a_idx_r = mdp_reduced.np_action_to_idx[aCN, aSN]
        traj_new.append((s_idx_r, joint_a_idx))

    return traj_new


def get_bayesian_infer_result(
    num_agent, list_np_policies,
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
    full_acc = full_count_correct / len(test_full_trajectories)

    part_count_correct = 0
    for idx, trj in enumerate(test_part_trajectories):
        infer_lat = bayesian_inference(
            trj, list_np_policies, possible_lstates, num_agent)
        true_lat = true_latent_labels[idx]
        part_conf[true_lat][infer_lat] += 1
        if true_lat == infer_lat:
            part_count_correct += 1
    part_acc = part_count_correct / len(test_part_trajectories)

    return full_conf, part_conf, full_acc, part_acc

if __name__ == "__main__":
    data_dir = './tooldelivery_v3_data/'
    file_prefix = 'td3_data_'
    tooldelivery_env = ToolDeliveryEnv_V3()
    num_agents = tooldelivery_env.num_brains
    num_ostates = tooldelivery_env.mmdp.num_states
    num_actions = tooldelivery_env.mmdp.num_actions
    possible_lstates = tooldelivery_env.policy.get_possible_latstate_indices()
    num_lstates = len(possible_lstates)

    # training trajectories
    file_names = glob.glob(os.path.join(data_dir, '*.txt'))

    trajectories = {}
    for lst in possible_lstates:
        trajectories[lst] = []

    mdp_reduced = Reduced_MDP_Tool_Delivery_v3()

    count = 0
    len_sum = 0
    for file_nm in file_names:
        # print(file_nm)
        trj, true_lat = read_sample(file_nm)
        if true_lat[0] is None or true_lat[1] is None:
            continue

        if true_lat[0] != true_lat[1]:
            continue

        partial_trj, request_idx = get_tooldelivery_partial_traj(
            tooldelivery_env, trj,
            num_b4_request=0,
            num_af_request=None)
        len_sum += len(partial_trj)
        count += 1
        traj_re = conv_traj_to_reduced_mdp(
            tooldelivery_env.mmdp, mdp_reduced, partial_trj)

        trajectories[true_lat[0]].append(traj_re)


    avg_len = int(len_sum / count) + 1
    print(avg_len) 

    print(len(trajectories[possible_lstates[0]]))
    print(len(trajectories[possible_lstates[1]]))

    # initial probability
    init_prop = get_initial_dist(mdp_reduced)

    def feature_ext(mdp, state, action):
        feat = np.zeros(mdp.num_states)
        feat[state] = 1
        return feat

    irl_list = []
    # rel_freq_list = []
    for lst in possible_lstates:
        irl = CMaxEntIRL(
            trajectories[lst],
            mdp_reduced,
            feature_extractor=feature_ext,
            initial_prop=init_prop,
            horizon=avg_len)
    
        irl_list.append(irl)
        # rel_freq_list.append(compute_relative_freq(tooldelivery_env.mmdp, trajectories[lst]))

    # reward_error = []
    # policy_error = []
    # def compute_errors(reward_fn, policy_fn):
    #     policy_error.append(cal_policy_error(rel_freq, toy_mdp, policy_fn, sto_pi))

    for idx, irl in enumerate(irl_list):
        irl.do_inverseRL(epsilon=0.001, n_max_run=300)

        with open('irl_weights_' + str(idx) + '.pickle', 'wb') as f:
            pickle.dump(irl.weights, f, pickle.HIGHEST_PROTOCOL) 
        with open('irl_policy_' + str(idx) + '.pickle', 'wb') as f:
            pickle.dump(irl.pi_est, f, pickle.HIGHEST_PROTOCOL) 

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

        test_full_trajectories.append(
            conv_traj_to_reduced_mdp(tooldelivery_env.mmdp, mdp_reduced, full_trj))
        test_part_trajectories.append(
            conv_traj_to_reduced_mdp(tooldelivery_env.mmdp, mdp_reduced, partial_trj))
        true_latent_labels.append(true_lat)
    
    # irl_conf_full, irl_conf_part, _, _ = get_bayesian_infer_result(num_agents, semisup_np_policy)
