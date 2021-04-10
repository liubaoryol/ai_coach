import numpy as np
import pickle
from moving_luggage.constants import (
    KEY_BAGS, KEY_AGENTS, AgentActions, LATENT_LIGHT_BAGS)
from policy.policy_utils import (
    get_feature_state_indv_v2)
from policy.qlearning import ApproximateQAgent
from policy.qlearning_numpy import QLearningAgent_Numpy
from policy.mdp_moving import MDPMovingLuggage_V2


with open('light_a1_q_table.pickle', 'rb') as f:
    np_q_values_light_a1 = pickle.load(f)
with open('light_a2_q_table.pickle', 'rb') as f:
    np_q_values_light_a2 = pickle.load(f)

with open('heavy_a1_q_table.pickle', 'rb') as f:
    np_q_values_heavy_a1 = pickle.load(f)
with open('heavy_a2_q_table.pickle', 'rb') as f:
    np_q_values_heavy_a2 = pickle.load(f)


def get_qlearn_numpy_policy(env, agent_idx, mental_model, goals, mdp_env=None):
    np_bags = env[KEY_BAGS]
    agent1 = env[KEY_AGENTS][0]
    agent2 = env[KEY_AGENTS][1]

    if mdp_env is None:
        mdp_env = MDPMovingLuggage_V2()

    NUM_TRAIN = 1000
    qlearn = QLearningAgent_Numpy(
        mdp_env=mdp_env,
        num_training=NUM_TRAIN, epsilon=0.1, beta=10, alpha=0.05, gamma=0.99)

    s_cur = (np_bags, agent1.coord, agent2.coord, agent1.hold, agent2.hold)

    feat_state = get_feature_state_indv_v2(s_cur, agent_idx, goals)
    if mental_model == LATENT_LIGHT_BAGS:
        qlearn.np_q_values = (
            np_q_values_light_a1 if agent_idx == 0 else np_q_values_light_a2)
    else:
        qlearn.np_q_values = (
            np_q_values_heavy_a1 if agent_idx == 0 else np_q_values_heavy_a2)
        # qlearn.weights = w_heavy_a1 if agent_idx == 0 else w_heavy_a2

    action = qlearn.getStochasticPolicy(feat_state, scalar=3)
    return AgentActions(action)