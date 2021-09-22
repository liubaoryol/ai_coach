import abc
import random
import numpy as np


class QLearningInterface():
  # Python 2 style but Python3 is also compatible with this.
  __metaclass__ = abc.ABCMeta

  def __init__(self,
               num_states: int,
               num_actions: int,
               num_training: int = 100,
               alpha: float = 0.1,
               gamma: float = 0.9):
    """
        alpha    - learning rate
        gamma    - discount factor
        numTraining - number of training episodes,
                      i.e. no learning after these many episodes
        """

    self.num_states = num_states
    self.num_actions = num_actions
    self.np_q_values = np.zeros((num_states, num_actions))

    self.alpha = alpha
    self.discount = gamma
    self.num_training = num_training

    self.episodes_sofar = 0
    self.accum_train_rewards = 0.0
    self.accum_test_rewards = 0.0

    self.start_episode()

  def get_episodes_sofar(self):
    return self.episodes_sofar

  def observe_transition(self, state_idx: int, action_idx: int, next_idx: int,
                         delta_reward: float):
    self.episode_rewards += delta_reward
    if self.is_in_training():
      self.update(state_idx, action_idx, next_idx, delta_reward)

  def start_episode(self):
    """
        Called by environment when new episode is starting
        """
    self.episode_rewards = 0.0

  def stop_episode(self):
    """
        Called by environment when episode is done
        """
    if self.episodes_sofar < self.num_training:
      self.accum_train_rewards += self.episode_rewards
    else:
      self.accum_test_rewards += self.episode_rewards
    self.episodes_sofar += 1

  def is_in_training(self):
    return self.episodes_sofar < self.num_training

  def get_q_value(self, state_idx: int, action_idx: int):
    """
        Returns Q(state,action)
        Should return 0.0 if we have never seen a state
        or the Q node value otherwise
        """
    return self.np_q_values[state_idx, action_idx]

  def get_v_value(self, state_idx: int):
    """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
        """

    return np.max(self.np_q_values[state_idx, :])

  def get_policy(self, state_idx: int):
    """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
        """
    list_act_idx = np.argwhere(
        self.np_q_values[state_idx, :] == self.get_v_value(state_idx))
    if len(list_act_idx) == 0:  # cannot be 0. inspect if NaN is inside
      return None

    return random.choice(list_act_idx)[0]

  @abc.abstractmethod
  def get_action(self, state_idx: int):
    raise NotImplementedError

  def update(self, state_idx: int, action_idx: int, next_idx: int,
             reward: float):
    self.np_q_values[state_idx, action_idx] = (
        (1 - self.alpha) * self.get_q_value(state_idx, action_idx) +
        self.alpha * (reward + (self.discount * self.get_v_value(next_idx))))

  def get_stochastic_policy_table(self, beta: float = 1):
    np_q = self.np_q_values - np.max(self.np_q_values, axis=1)[:, np.newaxis]
    np_q = np.exp(beta * np_q)
    sum_q = np.sum(np_q, axis=1)
    np_q = np_q / sum_q[:, np.newaxis]

    return np_q


class QLearningGreedy(QLearningInterface):
  def __init__(self, epsilon: float = 0.1, *args, **kwargs):
    self.epsilon = epsilon
    super().__init__(*args, **kwargs)

  def get_action(self, state_idx: int):
    if not self.is_in_training():
      return self.get_policy(state_idx)

    rand_val = random.random()
    if rand_val < self.epsilon:
      action_idx = random.choice(range(self.num_actions))
    else:
      action_idx = self.get_policy(state_idx)

    return action_idx


class QLearningSoftmax(QLearningInterface):
  def __init__(self, beta: float = 1., *args, **kwargs):
    self.beta = beta
    super().__init__(*args, **kwargs)

  def set_beta(self, beta: float):
    self.beta = beta

  def get_action(self, state_idx: int):
    np_q = self.np_q_values[state_idx, :]
    np_q_norm = np_q - np.max(np_q)
    np_q_norm = np.exp(self.beta * np_q_norm)
    # sum_q = np.sum(np_q)
    # np_q = np_q / sum_q
    action_idx = random.choices(range(self.num_actions),
                                weights=np_q_norm.tolist())[0]
    return action_idx
