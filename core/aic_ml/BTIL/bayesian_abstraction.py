from typing import Tuple, Sequence
import numpy as np
from tqdm import tqdm
from scipy.special import digamma, softmax


class Bayes_Abstraction:

  def __init__(self,
               trajectories: Sequence[Sequence[Tuple[int, Tuple[int, int]]]],
               num_states: int,
               tuple_num_actions: Tuple[int, ...],
               max_iteration: int = 1000,
               epsilon: float = 0.1,
               lr: float = 0.1,
               decay: float = 0.01,
               num_abstates: int = 30) -> None:
    '''
      trajectories: list of list of (state, joint action)-tuples
      tuple_num_latents: truncated stick-breaking number + 1
    '''

    HYPER_PI = 3
    HYPER_ABS = 3

    self.trajectories = trajectories

    self.list_hyper_pi = [HYPER_PI / n_a for n_a in tuple_num_actions]
    self.hyper_abs = HYPER_ABS / num_abstates

    self.num_agents = len(tuple_num_actions)
    self.num_ostates = num_states
    self.tuple_num_actions = tuple_num_actions
    self.num_abstates = num_abstates  # num abstract states

    self.list_np_policy = [None for dummy_i in range(self.num_agents)
                           ]  # type: list[np.ndarray]
    self.np_prob_abstate = None

    self.max_iteration = max_iteration
    self.epsilon = epsilon

    self.list_param_pi = None  # type: list[np.ndarray]
    self.param_abs = None  # type: np.ndarray
    self.lr = lr
    self.decay = decay

  def set_prior(self, pi_prior: float, abs_prior: float):
    self.list_hyper_pi = [pi_prior / n_a for n_a in self.tuple_num_actions]
    self.hyper_abs = abs_prior / self.num_abstates

  def compute_q_z(self, trajectory, np_abs_tilde, list_np_pi_tilde):
    with np.errstate(divide='ignore'):
      log_q_z = np.log(np.zeros((len(trajectory), self.num_abstates)))

    for t in range(len(trajectory)):
      stt, joint_a = trajectory[t]
      log_q_z[t] = np.log(np_abs_tilde[stt])
      for idx_a in range(self.num_agents):
        log_q_z[t] += (np.log((list_np_pi_tilde[idx_a][:, joint_a[idx_a]])))

    q_z = softmax(log_q_z, axis=1)

    return q_z

  def compute_local_variables(self, samples):
    list_q_z = []

    np_abs_tilde = self.get_prob_tilda_from_lambda(self.param_abs)
    list_np_pi_tilde = [
        self.get_prob_tilda_from_lambda(self.list_param_pi[idx_a])
        for idx_a in range(self.num_agents)
    ]

    for m_th in range(len(samples)):
      trajectory = samples[m_th]

      np_q_z = self.compute_q_z(trajectory, np_abs_tilde, list_np_pi_tilde)

      list_q_z.append(np_q_z)

    return list_q_z

  def update_global_variables(self, samples, lr,
                              list_q_z: Sequence[np.ndarray]):

    batch_ratio = len(self.trajectories) / len(samples)
    param_abs_hat = np.zeros_like(self.param_abs)
    list_param_pi_hat = [
        np.zeros_like(self.list_param_pi[idx_a])
        for idx_a in range(self.num_agents)
    ]

    # -- compute sufficient statistics
    for m_th in range(len(samples)):
      traj = samples[m_th]
      q_z = list_q_z[m_th]

      for t in range(len(traj)):
        state, joint_a = traj[t]
        param_abs_hat[state] += q_z[t]
        for idx_a in range(self.num_agents):
          list_param_pi_hat[idx_a][:, joint_a[idx_a]] += q_z[t]

    # - update agent-level global variables
    for idx_a in range(self.num_agents):
      list_param_pi_hat[idx_a] = (batch_ratio * list_param_pi_hat[idx_a] +
                                  self.list_hyper_pi[idx_a])

      self.list_param_pi[idx_a] = ((1 - lr) * self.list_param_pi[idx_a] +
                                   lr * list_param_pi_hat[idx_a])

    # - update task-level global variables
    param_abs_hat = batch_ratio * param_abs_hat + self.hyper_abs
    self.param_abs = ((1 - lr) * self.param_abs + lr * param_abs_hat)

  def get_prob_tilda_from_lambda(self, np_lambda):

    sum_lambda_pi = np.sum(np_lambda, axis=-1)
    ln_pi = digamma(np_lambda) - digamma(sum_lambda_pi)[..., None]
    return np.exp(ln_pi)

  def initialize_param(self):
    INIT_RANGE = (1, 1.1)

    self.list_param_pi = []
    # init abstraction parameters
    self.param_abs = np.random.uniform(low=INIT_RANGE[0],
                                       high=INIT_RANGE[1],
                                       size=(self.num_ostates,
                                             self.num_abstates))
    for idx_a in range(self.num_agents):
      self.list_param_pi.append(
          np.random.uniform(low=INIT_RANGE[0],
                            high=INIT_RANGE[1],
                            size=(self.num_abstates,
                                  self.tuple_num_actions[idx_a])))

  def do_inference(self, batch_size):
    num_traj = len(self.trajectories)
    batch_iter = int(num_traj / batch_size)
    count = 0
    progress_bar = tqdm(total=self.max_iteration)
    while count < self.max_iteration:
      if batch_size >= num_traj:
        samples = self.trajectories
      else:
        batch_idx = count % batch_iter
        if batch_idx == 0:
          perm_index = np.random.permutation(num_traj)

        samples = [
            self.trajectories[idx]
            for idx in perm_index[batch_idx * batch_size:(batch_idx + 1) *
                                  batch_size]
        ]

      count += 1

      prev_param_abs = np.copy(self.param_abs)
      prev_list_param_pi = [
          np.copy(self.list_param_pi[idx_a]) for idx_a in range(self.num_agents)
      ]

      list_q_z = self.compute_local_variables(samples)  # TODO: use batch

      # lr = (count + 1)**(-self.forgetting_rate)
      lr = self.lr / (count * self.decay + 1)
      self.update_global_variables(samples, lr, list_q_z)

      # compute delta
      delta_team = 0
      for idx_a in range(self.num_agents):
        delta_team = max(
            delta_team,
            np.max(np.abs(self.list_param_pi[idx_a] -
                          prev_list_param_pi[idx_a])))
        delta_team = max(delta_team,
                         np.max(np.abs(self.param_abs - prev_param_abs)))

      if delta_team < self.epsilon:
        break

      progress_bar.update()
      progress_bar.set_postfix({'delta': delta_team})
    progress_bar.close()

    self.convert_params_to_prob()

  def convert_params_to_prob(self):
    for i_a in range(self.num_agents):
      numerator = self.list_param_pi[i_a]
      action_sums = np.sum(numerator, axis=-1)
      self.list_np_policy[i_a] = numerator / action_sums[..., np.newaxis]

    numerator = self.param_abs
    abstract_sums = np.sum(numerator, axis=-1)
    self.np_prob_abstate = self.param_abs / abstract_sums[..., np.newaxis]
