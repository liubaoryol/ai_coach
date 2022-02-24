from typing import Callable, Optional, Tuple, Sequence
import random
import numpy as np
import sparse
from tqdm import tqdm
from ai_coach_core.RL.planning import soft_value_iteration
from ai_coach_core.models.mdp import MDP, softmax_policy_from_q_value

T_StateActionSeqence = Sequence[Tuple[int, int]]


class MaxEntIRL():
  def __init__(self,
               trajectories: Sequence[T_StateActionSeqence],
               mdp: MDP,
               feature_extractor: Callable[[MDP, int, int], np.ndarray],
               gamma: float = 0.9,
               initial_prop: Optional[np.ndarray] = None,
               learning_rate: float = 0.01,
               decay: float = 0.001,
               max_value_iter: int = 100,
               epsilon: float = 0.001):
    self.feature_extractor = feature_extractor
    self.mdp = mdp
    self.weights = None
    self.gamma = gamma
    self.alpha = learning_rate
    self.decay = decay
    self.eps = epsilon
    self.iteration = 0
    self.trajectories = trajectories
    self.max_value_iter = max_value_iter
    self.pi_est = None
    self.empirical_feature_cnt = None

    self.initial_prop = np.zeros((mdp.num_states))
    if initial_prop is not None:
      self.initial_prop = initial_prop
    else:
      n_states = self.mdp.num_states
      for state in range(n_states):
        self.initial_prop[state] = 1.0 / n_states

  def init_weights(self):
    feature = self.feature_extractor(self.mdp, 0, 0)  # F by 1 array
    self.weights = np.zeros(feature.shape)
    for idx in range(len(feature)):
      self.weights[idx] = random.uniform(0, 1)

  def get_weights(self):
    return self.weights

  def update_weights(self):
    gradient = self.get_gradient()
    self.weights = (self.weights + self.alpha /
                    (1 + self.decay * self.iteration) * gradient)

  def calc_empirical_feature_cnt(self):
    num_traj = len(self.trajectories)
    feature_bar = np.zeros(self.weights.shape)
    for traj in self.trajectories:
      feat_cnt = self.get_avg_feature_counts(traj)
      feature_bar = feature_bar + feat_cnt

    feature_bar = feature_bar / num_traj

    return feature_bar

  def get_gradient(self):
    if self.empirical_feature_cnt is None:
      self.empirical_feature_cnt = self.calc_empirical_feature_cnt()
    optimal_feature_cnt = self.calc_optimal_feature_cnt(self.eps)

    gradient = self.empirical_feature_cnt - optimal_feature_cnt

    return gradient

  def get_avg_feature_counts(self, trajectory: T_StateActionSeqence):
    feat_cnt = np.zeros(self.weights.shape)
    counts = len(trajectory)
    for state, action in trajectory:
      feature = self.feature_extractor(self.mdp, state, action)
      feat_cnt = feat_cnt + feature
    feat_cnt = feat_cnt / counts

    return feat_cnt

  def reward(self, state: int, action: int):
    feat = self.feature_extractor(self.mdp, state, action)
    return np.dot(self.weights, feat)

  def get_np_reward(self):
    np_reward = np.full((self.mdp.num_states, self.mdp.num_actions), -np.inf)
    for state in range(self.mdp.num_states):
      if self.mdp.is_terminal(state):
        # there is no valid action at the terminal state
        # but set one of them to have 0 reward
        # in order to make planning algorithms work
        np_reward[state, 0] = 0
      else:
        for action in self.mdp.legal_actions(state):
          np_reward[state, action] = self.reward(state, action)
    return np_reward

  def state_frequency(self):
    # slightly differnet from the original paper
    # i.e. utilize a discount factor to ensure convergence

    d_s_sum = np.array(self.initial_prop)

    iteration_idx = 0
    delta = self.eps + 1
    while (iteration_idx < self.max_value_iter) and (delta > self.eps):
      if isinstance(self.mdp.np_transition_model, sparse.COO):
        d_s_sum_new = (
            np.array(self.initial_prop) +
            self.gamma * sparse.tensordot(d_s_sum[:, np.newaxis] * self.pi_est,
                                          self.mdp.np_transition_model,
                                          axes=2))
      else:
        d_s_sum_new = (
            np.array(self.initial_prop) +
            self.gamma * np.tensordot(d_s_sum[:, np.newaxis] * self.pi_est,
                                      self.mdp.np_transition_model,
                                      axes=2))

      delta = np.linalg.norm(d_s_sum_new - d_s_sum)
      d_s_sum = d_s_sum_new
      iteration_idx += 1
    # print(iteration_idx)

    d_s_sum = d_s_sum / np.sum(d_s_sum)

    return d_s_sum

  def compute_stochastic_pi(self, epsilon: float):
    np_reward = self.get_np_reward()
    _, q_val = soft_value_iteration(self.mdp.np_transition_model,
                                    np_reward,
                                    discount_factor=self.gamma,
                                    max_iteration=self.max_value_iter,
                                    epsilon=epsilon,
                                    temperature=1.,
                                    show_progress_bar=False)
    self.pi_est = softmax_policy_from_q_value(q_val, temperature=1.)

  def policy(self, state, action):
    return self.pi_est[state, action]

  def calc_optimal_feature_cnt(self, epsilon_val_iter: float):
    self.compute_stochastic_pi(epsilon_val_iter)

    d_s = self.state_frequency()
    feat_cnt = np.zeros(self.weights.shape)
    for state in range(self.mdp.num_states):
      d_s_cur = d_s[state]
      for action in self.mdp.legal_actions(state):
        feat = self.feature_extractor(self.mdp, state, action)
        feat_cnt = feat_cnt + feat * d_s_cur * self.policy(state, action)

    return feat_cnt

  def do_inverseRL(self,
                   epsilon: float = 0.001,
                   n_max_run: int = 100,
                   callback_reward_pi: Optional[Callable[[Callable, Callable],
                                                         None]] = None):
    self.init_weights()
    delta = np.inf
    self.iteration = 0
    progress_bar = tqdm(total=n_max_run)
    while delta > epsilon and self.iteration < n_max_run:
      weights_old = self.weights.copy()
      self.update_weights()
      if callback_reward_pi:
        callback_reward_pi(self.reward, self.policy)
      diff = np.max(np.abs(weights_old - self.weights))

      delta = diff
      self.iteration += 1
      # print("Delta-" + str(delta) + ", cnt-" + str(self.iteration))
      # print(self.weights)
      progress_bar.set_postfix({'delta': delta})
      progress_bar.update()
    progress_bar.close()


def compute_relative_freq(num_states: int,
                          trajectories: Sequence[T_StateActionSeqence]):
  rel_freq = np.zeros(num_states)
  count = 0
  for traj in trajectories:
    for s, _ in traj:
      rel_freq[s] += 1
      count += 1

  rel_freq = rel_freq / count

  return rel_freq


def cal_reward_error(mdp: MDP, fn_reward_irl: Callable[[int, int], float]):
  sum_diff = 0
  for s_idx in range(mdp.num_states):
    for a_idx in mdp.legal_actions(s_idx):
      r_irl = fn_reward_irl(s_idx, a_idx)
      r_mdp = mdp.reward(s_idx, a_idx, None)
      sum_diff += (r_irl - r_mdp)**2

  return np.sqrt(sum_diff)


def cal_policy_error(rel_freq: np.ndarray, mdp: MDP,
                     pi_irl: Callable[[int, int], float], pi_true: np.ndarray):
  def compute_kl(s_idx):
    sum_val = 0
    for a_idx in mdp.legal_actions(s_idx):
      p_irl = pi_irl(s_idx, a_idx)
      p_h = pi_true[s_idx, a_idx]
      if p_irl != 0 and p_h != 0:
        sum_val += p_irl * (np.log(p_irl) - np.log(p_h))
    return sum_val

  sum_kl = 0
  for s_idx in range(mdp.num_states):
    sum_kl += rel_freq[s_idx] * compute_kl(s_idx)

  return sum_kl
