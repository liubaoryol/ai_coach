from os import stat
import random
import numpy as np
from generate_policy.qlearning import CReinforcement


class QLearningAgent_Numpy(CReinforcement):
    'assume all actions are legal'

    def __init__(self, mdp_env, beta=1, **args):
        "You can initialize Q-values here..."
        CReinforcement.__init__(self, **args)

        self.mdp_env = mdp_env
        self.beta = beta
        self.np_q_values = np.zeros((mdp_env.num_states, mdp_env.num_actions))

    def getQValue(self, state, action):
        """
        Returns Q(state,action)
        Should return 0.0 if we have never seen a state
        or the Q node value otherwise
        """
        state_idx = self.mdp_env.np_state_to_idx[state]
        action_idx = self.mdp_env.np_action_to_idx[action]

        return self.np_q_values[state_idx, action_idx]

    def computeValueFromQValues(self, state):
        """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
        """

        state_idx = self.mdp_env.np_state_to_idx[state]
        return np.max(self.np_q_values[state_idx, :])

    def computeActionFromQValues(self, state):
        """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
        """
        state_idx = self.mdp_env.np_state_to_idx[state]
        list_act_idx = np.argwhere(
            self.np_q_values[state_idx, :] == (
                self.computeValueFromQValues(state)))
        if len(list_act_idx) == 0:  # cannot be 0. inspect if NaN 
            return None

        return self.mdp_env.np_idx_to_action[random.choice(list_act_idx)]

    def getAction(self, state):
        action = None

        if self.epsilon >= 0:
            rand_val = random.random()
            if rand_val < self.epsilon:
                action = self.mdp_env.np_idx_to_action[
                    random.choice(range(self.mdp_env.num_actions))]
            else:
                action = self.computeActionFromQValues(state)
        else:
            action = self.getRandomActionFromSoftmaxQ(state, self.beta)

        self.doAction(state, action)
        return action

    def update(self, state, action, nextState, reward):
        """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here
        """
        state_idx = self.mdp_env.np_state_to_idx[state]
        action_idx = self.mdp_env.np_action_to_idx[action]

        self.np_q_values[state_idx, action_idx] = (
            (1 - self.alpha) * self.getQValue(state, action) +
            self.alpha * (
                reward +
                (self.discount * self.computeValueFromQValues(nextState))))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def getRandomActionFromSoftmaxQ(self, state, scalar=1):
        state_idx = self.mdp_env.np_state_to_idx[state]
        np_q = np.array(self.np_q_values[state_idx, :])
        np_q = np_q - np.min(np_q)
        np_q = np.exp(scalar * np_q)
        # sum_q = np.sum(np_q)
        # np_q = np_q / sum_q
        action_idx = random.choices(
            range(self.mdp_env.num_actions), weights=np_q.tolist())[0]
        return self.mdp_env.np_idx_to_action[action_idx]

    def getStochasticPolicy(self, scalar=1):
        # np_q = np.array(self.np_q_values[state_idx, :])
        np_q = self.np_q_values - np.min(self.np_q_values, axis=1)[:, np.newaxis]
        np_q = np.exp(scalar * np_q)
        sum_q = np.sum(np_q, axis=1)
        np_q = np_q / sum_q[:, np.newaxis]
         
        return np_q