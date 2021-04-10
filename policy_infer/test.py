import glob, os
import random
import time
import numpy as np
from policy_infer.bayesian_policy_infer import (
    supervised_bayesian_policy_learning)
from policy.policy_utils import get_feature_state_indv_v2
from policy.mdp_moving import MDPMovingLuggage_V2
from moving_luggage.hand_policies import get_qlearn_numpy_policy
from moving_luggage.simulator import Simulator
from moving_luggage.constants import (
    AgentActions, KEY_AGENTS, KEY_BAGS, LATENT_HEAVY_BAGS, LATENT_LIGHT_BAGS,
    NUM_X_GRID, NUM_Y_GRID)


def generate_sequence(game, mdp, latents, goals):
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

        action1 = get_qlearn_numpy_policy(env, 0, latents[0], goals, mdp)
        action2 = get_qlearn_numpy_policy(env, 1, latents[1], goals, mdp)
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

def generate_multiple_samples(game, mdp, dir_path, num, file_prefix):
    for dummy in range(num):
        a1_ls = random.choice([LATENT_HEAVY_BAGS, LATENT_LIGHT_BAGS])
        a2_ls = random.choice([LATENT_HEAVY_BAGS, LATENT_LIGHT_BAGS])
        latents = (a1_ls, a2_ls)
        traj1, traj2 = generate_sequence(
            game, mdp, latents, game.goal_pos)

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


if __name__ == "__main__":
    data_dir = './mvbag_trajectories/'
    file_prefix = 'mvbag_data_'
    game = Simulator()
    mdp = MDPMovingLuggage_V2()
    num_ostates = mdp.num_states
    num_actions = mdp.num_actions
    possible_latents = [LATENT_HEAVY_BAGS, LATENT_LIGHT_BAGS]
    num_lstates = len(possible_latents)


    # # to generate sequences comment out below lines
    # file_names = glob.glob(os.path.join(data_dir, file_prefix + '*.txt'))
    # for fmn in file_names:
    #     os.remove(fmn)
    # generate_multiple_samples(game, mdp, data_dir, 10, file_prefix)

    file_names = glob.glob(os.path.join(data_dir, '*.txt'))

    trajectories1 = []
    trajectories2 = []
    latent_labels1 = []
    latent_labels2 = []
    for file_nm in file_names:
        # print(file_nm)
        trj1, trj2, true_lat = read_sample(file_nm)
        if true_lat[0] is None or true_lat[1] is None:
            continue

        trajectories1.append(trj1)
        trajectories2.append(trj2)
        latent_labels1.append(true_lat[0])
        latent_labels2.append(true_lat[1])

    ##############################################
    # supervised inference
    sup_infer1 = supervised_bayesian_policy_learning(
        trajectories1, latent_labels1, 1, num_ostates,
        num_lstates, num_actions)

    sup_infer1.set_dirichlet_prior(10)
    sup_infer1.do_inference()
    sup_np_policy1 = sup_infer1.np_policy

    sup_infer2 = supervised_bayesian_policy_learning(
        trajectories2, latent_labels2, 1, num_ostates,
        num_lstates, num_actions)

    sup_infer2.set_dirichlet_prior(10)
    sup_infer2.do_inference()
    sup_np_policy2 = sup_infer2.np_policy

    ##############################################
    # test sets
    test_dir = './mvbag_trajectories_test/'
    test_file_prefix = 'mvbag_test_'
    # generate_multiple_samples(game, mdp, test_dir, 50, test_file_prefix)

    test_file_names = glob.glob(os.path.join(test_dir, '*.txt'))

    test_trajectories = []
    true_latent_labels = []
    infer_latent_labels = []
    conf = {}
    conf[(0, 0)] = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    conf[(0, 1)] = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    conf[(1, 0)] = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    conf[(1, 1)] = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    count = 0
    len_seq = 10
    for file_nm in test_file_names:
        # print(file_nm)
        trj1, trj2, true_lat = read_sample(file_nm)
        if true_lat[0] is None or true_lat[1] is None:
            continue
        len_traj = len(trj1)
        count += len(trj1)
        i_st = random.randint(0, len_traj - len_seq)

        infer_ls1 = bayesian_inference(
            trj1[i_st:i_st + len_seq], sup_np_policy1, possible_latents, 1)
        infer_ls2 = bayesian_inference(
            trj2[i_st:i_st + len_seq], sup_np_policy2, possible_latents, 1)

        # test_trajectories.append(partial_trj)
        true_latent_labels.append(true_lat)
        infer_ls = (infer_ls1[0], infer_ls2[0])
        infer_latent_labels.append(infer_ls)
        conf[true_lat][infer_ls] += 1

    # print(conf)
    # print(count / 50)
   
    ordered_key = [(0, 0), (1, 1), (0, 1), (1, 0)]
    for key1 in ordered_key:
        print(key1)
        txt_pred = ""
        for key2 in ordered_key:
            txt_pred = txt_pred + str(key2) + ": " + str(conf[key1][key2]) + ", "
        print(txt_pred) 

    
