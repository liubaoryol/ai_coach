# import pickle5 as pickle
import matplotlib.pyplot as plt
import glob, os
import argparse
from mdp import *
from maxent_irl import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generating results for Problem Set 2")
    parser.add_argument("--p231", dest="policy", action="store_true", help="show the optimal policy and value")
    parser.add_argument("--p232", dest="num_samples", type=int, help="generate trajectory samples")
    parser.add_argument("--p233", dest="irl_p233", action="store_true", help="train IRL with original features and plot errors")
    parser.add_argument("--p234", dest="irl_p234", action="store_true", help="train IRL with updated features and plot errors")
    args = parser.parse_args()

    trans = get_transition_p2()

    # print_trans(trans)
    mdp = CMDP_P2(trans, 0.9)
    value_fn = value_iteration(mdp, 0.001)
    # value_fn = soft_value_iteration(mdp, lambda s, a: mdp.reward(s, a, None), 0.001)
    pi = best_policy(mdp, value_fn)
    sto_pi = get_stochastic_policy(mdp, pi)

    if args.policy:
        for state in sorted(mdp.states()):
            val = value_fn[state]
            action = pi[state]
            str_action = "wait"
            if action == (0, 1):
                str_action = "uparrow"
            elif action == (0, -1):
                str_action = "downarrow"
            elif action == (1, 0):
                str_action = "rightarrow"
            elif action == (-1, 0):
                str_action = "leftarrow"
            elif action == (1, 1):
                str_action = "nearrow"
            elif action == (-1, 1):
                str_action = "nwarrow"
            elif action == (1, -1):
                str_action = "searrow"
            elif action == (-1, -1):
                str_action = "swarrow"
            elif action == PICK:
                str_action = "pick"

            print(str(state) + ": " + str_action + " / " + "%.2f" % (val,))

    if args.num_samples:
        if args.num_samples > 0:
            for dummy in range(args.num_samples):
                sample = gen_trajectory(mdp, sto_pi)
                file_path = os.path.join(DATA_DIR, str(dummy) + '.txt')
                save_trajectory(sample, file_path)
                # print(sample)
            print("data generated")

    if args.irl_p233 or args.irl_p234:
        trajectories = []
        len_sum = 0
        file_names = glob.glob(os.path.join(DATA_DIR, '*.txt'))
        for file_nm in file_names:
            traj = read_trajectory(file_nm)
            len_sum += len(traj)
            trajectories.append(traj)
        
        avg_len = int(len_sum / len(trajectories)) + 1
        # print(avg_len)

        init_prop = {s: 0 for s in mdp.states()}
        init_prop[(0, 0, 0)] = 1

        rel_freq = compute_relative_freq(mdp, trajectories)
        reward_error = []
        policy_error = []
        def compute_errors(reward_fn, policy_fn):
            reward_error.append(cal_reward_error(mdp, reward_fn))
            policy_error.append(cal_policy_error(rel_freq, mdp, policy_fn, sto_pi))

        if args.irl_p233:
            irl = CMaxEntIRL(trajectories, mdp, feature_extractor=feature_extract, initial_prop=init_prop, horizon=avg_len)
            irl.do_inverseRL(epsilon=0.001, n_max_run=1000, callback_reward_pi=compute_errors)

            # with open('reward_error.pickle', 'wb') as f:
            #     pickle.dump(reward_error, f, pickle.HIGHEST_PROTOCOL) 
            # with open('policy_error.pickle', 'wb') as f:
            #     pickle.dump(policy_error, f, pickle.HIGHEST_PROTOCOL) 
       
            f = plt.figure(figsize=(10,5))
            ax1 = f.add_subplot(121)
            ax2 = f.add_subplot(122)
            ax1.plot(reward_error)
            ax1.set_ylabel('reward_error')
            # plt.show()

            ax2.plot(policy_error)
            ax2.set_ylabel('policy_error')
            plt.show() 

        if args.irl_p234:
            irl = CMaxEntIRL(trajectories, mdp, feature_extractor=feature_extract_updated, initial_prop=init_prop, horizon=avg_len)
            irl.do_inverseRL(epsilon=0.001, n_max_run=1000, callback_reward_pi=compute_errors)

            # with open('reward_error_updated.pickle', 'wb') as f:
            #     pickle.dump(reward_error, f, pickle.HIGHEST_PROTOCOL) 
            # with open('policy_error_updated.pickle', 'wb') as f:
            #     pickle.dump(policy_error, f, pickle.HIGHEST_PROTOCOL) 

            f = plt.figure(figsize=(10,5))
            ax1 = f.add_subplot(121)
            ax2 = f.add_subplot(122)
            ax1.plot(reward_error)
            ax1.set_ylabel('reward_error')
            # plt.show()

            ax2.plot(policy_error)
            ax2.set_ylabel('policy_error')
            plt.show()
    
    # use_original_feature_set = False
    # if use_original_feature_set:
    #     with open('reward_error.pickle', 'rb') as f:
    #         reward_error = pickle.load(f)
    #     with open('policy_error.pickle', 'rb') as f:
    #         policy_error = pickle.load(f)
    # else:
    #     with open('reward_error_updated.pickle', 'rb') as f:
    #         reward_error = pickle.load(f)
    #     with open('policy_error_updated.pickle', 'rb') as f:
    #         policy_error = pickle.load(f)
    