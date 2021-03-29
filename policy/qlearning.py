import random
import numpy as np

class CValueEstimation():

    def __init__(self, num_training=100, epsilon=0.5, alpha=0.5, gamma=0.9):
        self.alpha = float(alpha)
        self.epsilon = float(epsilon) 
        self.discount = float(gamma)
        self.num_training = int(num_training)

    def getQValue(self, state, action):
        """
        Should return Q(state,action)
        """
        pass

    def getValue(self, state):
        """
        What is the value of this state under the best action?
        Concretely, this is given by

        V(s) = max_{a in actions} Q(s,a)
        """
        pass

    def getPolicy(self, state):
        """
        What is the best action to take in the state. Note that because
        we might want to explore, this might not coincide with getAction
        Concretely, this is given by

        policy(s) = arg_max_{a in actions} Q(s,a)

        If many actions achieve the maximal Q-value,
        it doesn't matter which is selected.
        """
        pass

    def getAction(self, state):
        """
        state: can call state.getLegalActions()
        Choose an action and return it.
        """
        pass


class CReinforcement(CValueEstimation):

    def __init__(self, actionFn=None, **args):
        """
        actionFn: Function which takes a state and
                    returns the list of legal actions

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes,
                      i.e. no learning after these many episodes
        """
        super().__init__(**args)

        if actionFn is None:
            def empty_states(state):
                return []

            actionFn = empty_states  # no actions available

        self.actionFn = actionFn
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0

        self.startEpisode()

    def get_episodes_sofar(self):
        return self.episodesSoFar

    def update(self, state, action, nextState, reward):
        """
        This class will call this function, which you write, after
        observing a transition and reward
        """
        pass

    def getLegalActions(self, state):
        """
        Get the actions available for a given
        state. This is what you should use to
        obtain legal actions for a state
        """
        return self.actionFn(state)

    def observeTransition(self, state, action, nextState, deltaReward):
        self.episodeRewards += deltaReward
        self.update(state, action, nextState, deltaReward)

    def startEpisode(self):
        """
        Called by environment when new episode is starting
        """
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

    def stopEpisode(self):
        """
        Called by environment when episode is done
        """
        if self.episodesSoFar < self.num_training:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.num_training:
            # Take off the training wheels
            self.epsilon = 0.0    # no exploration
            self.alpha = 0.0      # no learning

    def isInTraining(self):
        return self.episodesSoFar < self.num_training

    def isInTesting(self):
        return not self.isInTraining()

    def doAction(self, state, action):
        """
            Called by inherited class when
            an action is taken in a state
        """
        self.lastState = state
        self.lastAction = action


class QLearningAgent(CReinforcement):

    def __init__(self, beta=1, **args):
        "You can initialize Q-values here..."
        CReinforcement.__init__(self, **args)

        self.q_values = {}
        self.beta = beta

    def getQValue(self, state, action):
        """
        Returns Q(state,action)
        Should return 0.0 if we have never seen a state
        or the Q node value otherwise
        """
        key = (state, action)
        if key in self.q_values:
            return self.q_values[key]
        else:
            return 0.0

    def debug(self, state, action):
        pass

    def computeValueFromQValues(self, state):
        """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
        """
        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return 0.0

        max_q = float("-inf")
        for action in actions:
            cur_q = self.getQValue(state, action)
            if cur_q > max_q:
                max_q = cur_q
        return max_q

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return None

        max_q = self.computeValueFromQValues(state)

        max_actions = []
        for action in actions:
            cur_q = self.getQValue(state, action)
            if cur_q == max_q:
                max_actions.append(action)
        # if max_q == float("-inf") or max_q == float("inf"):
        #     print(str(state))
        #     print("max_q %f, num_actions %d, num_max_act %d" %
        #     (max_q, len(actions), len(max_actions)))
        #     for action in actions:
        #         self.debug(state, action)
        return random.choice(max_actions)

    def getAction(self, state):
        """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None

        if len(legalActions) == 0:
            return None

        if self.epsilon >= 0:
            rand_val = random.random()
            if rand_val < self.epsilon:
                action = random.choice(legalActions)
            else:
                action = self.computeActionFromQValues(state)
        else:
            action = self.getStochasticPolicy(state, self.beta)

        self.doAction(state, action)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
        """
        key = (state, action)
        self.q_values[key] = (
            (1 - self.alpha) * self.getQValue(state, action) +
            self.alpha * (reward +
                          (self.discount *
                           self.computeValueFromQValues(nextState))))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def getStochasticPolicy(self, state, scalar=1):
        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return None

        np_q = np.zeros(len(actions))
        for idx, action in enumerate(actions):
            np_q[idx] = self.getQValue(state, action)
        np_q = np.exp(scalar * np_q)
        # sum_q = np.sum(np_q)
        # np_q = np_q / sum_q

        return random.choices(actions, weights=np_q.tolist())[0]

class ApproximateQAgent(QLearningAgent):
    """
       ApproximateQLearningAgent
    """
    def __init__(self, feature_extractor=None, **args):
        self.feat_extractor = feature_extractor
        QLearningAgent.__init__(self, **args)
        self.weights = {}

    def getWeights(self):
        return self.weights

    def debug(self, state, action):
        features = self.feat_extractor(state, action)
        print(action)
        print(features)
        weights = self.getWeights()
        sum_values = 0
        for key in features:
            if key not in weights:
                weights[key] = 0
            sum_values += weights[key] * features[key]

        return sum_values

    def getQValue(self, state, action):
        """
        Should return Q(state,action) = w * featureVector
        where * is the dotProduct operator
        """
        features = self.feat_extractor(state, action)
        weights = self.getWeights()
        sum_values = 0
        for key in features:
            if key not in weights:
                weights[key] = 0
            sum_values += weights[key] * features[key]

        return sum_values

    def update(self, state, action, nextState, reward):
        """
        Should update your weights based on transition
        """
        weights = self.getWeights()
        features = self.feat_extractor(state, action)
        diff = (
            reward + self.discount * self.computeValueFromQValues(nextState) -
            self.getQValue(state, action))
        for key in features:
            weights[key] = weights[key] + self.alpha * diff * features[key]
