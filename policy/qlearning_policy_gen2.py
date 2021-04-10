import numpy as np
import pickle

from moving_luggage.constants import (
    AgentActions, KEY_AGENTS, KEY_BAGS, NUM_X_GRID, NUM_Y_GRID)
from moving_luggage.simulator import Simulator
from policy.policy_utils import get_feature_state_indv_v2
from policy.qlearning_numpy import QLearningAgent_Numpy
from policy.mdp_moving import MDPMovingLuggage_V2


if __name__ == "__main__":
    game = Simulator()

    def reward_light(state_c, action, state_n, agent_idx):
        np_bag_c, a1_p_c, a2_p_c, a1_h_c, a2_h_c = state_c
        np_bag_n, a1_p_n, a2_p_n, a1_h_n, a2_h_n = state_n

        my_p_c, op_p_c, my_h_c, op_h_c = a1_p_c, a2_p_c, a1_h_c, a2_h_c
        my_p_n, op_p_n, my_h_n, op_h_n = a1_p_n, a2_p_n, a1_h_n, a2_h_n
        if agent_idx == 1:
            my_p_c, op_p_c, my_h_c, op_h_c = a2_p_c, a1_p_c, a2_h_c, a1_h_c
            my_p_n, op_p_n, my_h_n, op_h_n = a2_p_n, a1_p_n, a2_h_n, a1_h_n

        reward = -0.1
        if my_h_c and (my_p_n in game.goal_pos):
            if (my_p_c == op_p_c and my_h_c and op_h_c):
                reward += 5
            else:
                reward += 10
        
        return reward

    def reward_heavy(state_c, action, state_n, agent_idx):
        np_bag_c, a1_p_c, a2_p_c, a1_h_c, a2_h_c = state_c
        np_bag_n, a1_p_n, a2_p_n, a1_h_n, a2_h_n = state_n

        my_p_c, op_p_c, my_h_c, op_h_c = a1_p_c, a2_p_c, a1_h_c, a2_h_c
        my_p_n, op_p_n, my_h_n, op_h_n = a1_p_n, a2_p_n, a1_h_n, a2_h_n
        if agent_idx == 1:
            my_p_c, op_p_c, my_h_c, op_h_c = a2_p_c, a1_p_c, a2_h_c, a1_h_c
            my_p_n, op_p_n, my_h_n, op_h_n = a2_p_n, a1_p_n, a2_h_n, a1_h_n

        reward = -0.1
        # reward = 0
        if my_h_c and (my_p_n in game.goal_pos):
            reward += 10

        if (
            (my_p_c == op_p_c and my_h_c and op_h_c) and
            (my_p_n == op_p_n and my_h_n and op_h_n)):
            reward += 0.05

        if (
            not (my_p_c == op_p_c and my_h_c and op_h_c) and
            (my_h_c and my_h_n and my_p_c != my_p_n)):
            reward -= 10
        
        # if (
        #     (my_p_c == op_p_c and my_h_c and op_h_c) and
        #     (op_h_c and op_h_n and op_p_c != op_p_n) and
        #     (my_p_c == my_p_n)):
        #     reward -= 10
        
        return reward

    mdp_env = MDPMovingLuggage_V2()

    NUM_TRAIN = 100000
    qlearn_a1 = QLearningAgent_Numpy(
        mdp_env=mdp_env,
        num_training=NUM_TRAIN, epsilon=-1, beta=1, alpha=0.05, gamma=0.99)
    qlearn_a2 = QLearningAgent_Numpy( 
        mdp_env=mdp_env,
        num_training=NUM_TRAIN, epsilon=-1, beta=1, alpha=0.05, gamma=0.99)

    # with open('heavy_a1.pickle', 'rb') as f:
    #     w_heavy_a1 = pickle.load(f)
    #     qlearn_a1.weights = w_heavy_a1
    # with open('heavy_a2.pickle', 'rb') as f:
    #     w_heavy_a2 = pickle.load(f)
    #     qlearn_a2.weights = w_heavy_a2

    env_id = 0
    game.set_max_step(3000)
    for i in range(NUM_TRAIN):
        game.finish_game(env_id)
        game.add_new_env(env_id, int(NUM_X_GRID * NUM_Y_GRID / 4))
        env = game.map_id_env[env_id]

        np_bags = env[KEY_BAGS]
        agent1 = env[KEY_AGENTS][0]
        agent2 = env[KEY_AGENTS][1]

        s_cur = (np_bags, agent1.coord, agent2.coord, agent1.hold, agent2.hold)
        qlearn_a1.startEpisode()
        qlearn_a2.startEpisode()
        count = 0
        while True:
            if game.is_finished(env_id):
                # print("finished")
                break
            # print("episodes: %d, count: %d" % 
            # (qlearn_a1.get_episodes_sofar(), count))
            count +=1

            a1_state = get_feature_state_indv_v2(s_cur, 0, game.goal_pos)
            a2_state = get_feature_state_indv_v2(s_cur, 1, game.goal_pos)

            action1 = qlearn_a1.getAction(a1_state)
            action2 = qlearn_a2.getAction(a2_state)

            # joint_action = action2 * len(AgentActions) + action1

            game._take_simultaneous_step(
                env, AgentActions(action1), AgentActions(action2))

            np_bags_nxt = env[KEY_BAGS]
            agent1_nxt = env[KEY_AGENTS][0]
            agent2_nxt = env[KEY_AGENTS][1]
            s_nxt = (
                np_bags_nxt,
                agent1_nxt.coord, agent2_nxt.coord,
                agent1_nxt.hold, agent2_nxt.hold)

            a1_state_n = get_feature_state_indv_v2(s_nxt, 0, game.goal_pos)
            a2_state_n = get_feature_state_indv_v2(s_nxt, 1, game.goal_pos)

            # reward_a1 = reward_light(s_cur, action1, s_nxt, 0)
            # reward_a2 = reward_light(s_cur, action2, s_nxt, 1)
            reward_a1 = reward_heavy(s_cur, action1, s_nxt, 0)
            reward_a2 = reward_heavy(s_cur, action2, s_nxt, 1)
            qlearn_a1.observeTransition(
                a1_state, action1, a1_state_n, reward_a1)
            qlearn_a2.observeTransition(
                a2_state, action2, a2_state_n, reward_a2)
            s_cur = s_nxt
        qlearn_a1.stopEpisode()
        qlearn_a2.stopEpisode()
        print("episodes: %d, count: %d, a1_reward: %.2f, a2_reward: %.2f" %
            (
                qlearn_a1.get_episodes_sofar(), count,
                qlearn_a1.episodeRewards, qlearn_a2.episodeRewards))
        # if qlearn_a1.episodeRewards > 45 and qlearn_a2.episodeRewards > 45:
        #     break

    # with open('light_a1_q_table.pickle', 'wb') as f:
    #     pickle.dump(qlearn_a1.np_q_values, f, pickle.HIGHEST_PROTOCOL)
    # with open('light_a2_q_table.pickle', 'wb') as f:
    #     pickle.dump(qlearn_a2.np_q_values, f, pickle.HIGHEST_PROTOCOL)

    with open('heavy_a1_q_table2.pickle', 'wb') as f:
        pickle.dump(qlearn_a1.np_q_values, f, pickle.HIGHEST_PROTOCOL)
    with open('heavy_a2_q_table2.pickle', 'wb') as f:
        pickle.dump(qlearn_a2.np_q_values, f, pickle.HIGHEST_PROTOCOL)
