import numpy as np
import pickle
import random
from moving_luggage.constants import (
    KEY_BAGS, KEY_AGENTS, AgentActions, LATENT_LIGHT_BAGS, LATENT_HEAVY_BAGS)
from generate_policy.policy_utils import (
    get_feature_state_indv_v2)
from generate_policy.qlearning_numpy import QLearningAgent_Numpy
from generate_policy.mdp_moving import MDPMovingLuggage_V2


with open('moving_luggage/policies/light_a1_q_table2.pickle', 'rb') as f:
    np_q_values_light_a1 = pickle.load(f)
with open('moving_luggage/policies/light_a2_q_table2.pickle', 'rb') as f:
    np_q_values_light_a2 = pickle.load(f)

with open('moving_luggage/policies/heavy_a1_q_table.pickle', 'rb') as f:
    np_q_values_heavy_a1 = pickle.load(f)
with open('moving_luggage/policies/heavy_a2_q_table.pickle', 'rb') as f:
    np_q_values_heavy_a2 = pickle.load(f)


def get_qlearn_numpy_policy_action(
    env, agent_idx, mental_model, goals, beta=3, mdp_env=None):
    np_bags = env[KEY_BAGS]
    agent1 = env[KEY_AGENTS][0]
    agent2 = env[KEY_AGENTS][1]

    if mdp_env is None:
        mdp_env = MDPMovingLuggage_V2()

    NUM_TRAIN = 1000
    qlearn = QLearningAgent_Numpy(
        mdp_env=mdp_env,
        num_training=NUM_TRAIN, epsilon=-1, alpha=0.05, gamma=0.99)

    s_cur = (np_bags, agent1.coord, agent2.coord, agent1.hold, agent2.hold)

    feat_state = get_feature_state_indv_v2(s_cur, agent_idx, goals)
    if mental_model == LATENT_LIGHT_BAGS:
        qlearn.np_q_values = (
            np_q_values_light_a1 if agent_idx == 0 else np_q_values_light_a2)
    else:
        qlearn.np_q_values = (
            np_q_values_heavy_a1 if agent_idx == 0 else np_q_values_heavy_a2)
        # qlearn.weights = w_heavy_a1 if agent_idx == 0 else w_heavy_a2

    action = qlearn.getRandomActionFromSoftmaxQ(feat_state, scalar=beta)
    return AgentActions(action)


def get_qlearn_numpy_policy_dist(beta=3, mdp_env=None):

    if mdp_env is None:
        mdp_env = MDPMovingLuggage_V2()

    NUM_TRAIN = 1000
    qlearn = QLearningAgent_Numpy(
        mdp_env=mdp_env,
        num_training=NUM_TRAIN, epsilon=-1, alpha=0.05, gamma=0.99)

    NUM_AGENTS = 2
    NUM_LATENT_STATES = 2
    list_q_values_a1 = {}
    list_q_values_a1[LATENT_HEAVY_BAGS] = np_q_values_heavy_a1
    list_q_values_a1[LATENT_LIGHT_BAGS] = np_q_values_light_a1
    list_q_values_a2 = {}
    list_q_values_a2[LATENT_HEAVY_BAGS] = np_q_values_heavy_a2
    list_q_values_a2[LATENT_LIGHT_BAGS] = np_q_values_light_a2
    list_q_values = [list_q_values_a1, list_q_values_a2]
    
    list_np_policy = []
    num_states = mdp_env.num_states
    num_actions = mdp_env.num_actions
    for n_idx in range(NUM_AGENTS):
        list_np_policy.append(
            np.zeros((num_states, NUM_LATENT_STATES, num_actions)))
        for x_idx in [LATENT_HEAVY_BAGS, LATENT_LIGHT_BAGS]:
            qlearn.np_q_values = list_q_values[n_idx][x_idx]
            np_sa_policy = qlearn.getStochasticPolicy(beta)
            for s_idx in range(num_states):
                for a_idx in range(num_actions):
                    list_np_policy[n_idx][s_idx][x_idx][a_idx] = np_sa_policy[s_idx][a_idx]

    return list_np_policy
