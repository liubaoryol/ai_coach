from typing import Tuple, Sequence
import numpy as np
from tqdm import tqdm
from scipy.special import digamma, logsumexp, softmax
from ai_coach_core.model_learning.BTIL.transition_x import TransitionX
import time

T_SAXSeqence = Sequence[Tuple[int, Tuple[int, int], Tuple[int, int]]]


class BTIL_Abstraction:

  def __init__(
      self,
      trajectories: Sequence[T_SAXSeqence],
      num_states: int,
      tuple_num_latents: Tuple[int, ...],
      tuple_num_actions: Tuple[int, ...],
      trans_x_dependency=(True, True, True, True, False),  # s, a1, a2, a3, sn
      max_iteration: int = 1000,
      epsilon_g: float = 0.1,
      epsilon_l: float = 0.05,
      lr: float = 0.1,
      decay: float = 0.01,
      num_abstates: int = 30,
      num_mc_4_qz: int = 30,
      num_mc_4_qx: int = 10,
      save_file_prefix: str = None) -> None:
    '''
      trajectories: list of list of (state, joint action)-tuples
      tuple_num_latents: truncated stick-breaking number + 1
    '''

    assert len(tuple_num_actions) + 2 == len(trans_x_dependency)

    HYPER_GEM = 3
    HYPER_TX = 3
    HYPER_PI = 3
    HYPER_ABS = 3

    self.trajectories = trajectories
    self.save_prefix = save_file_prefix

    self.hyper_gem = HYPER_GEM
    self.hyper_tx = HYPER_TX
    self.hyper_pi = HYPER_PI
    self.hyper_abs = HYPER_ABS

    self.num_agents = len(tuple_num_actions)
    self.num_ostates = num_states
    self.tuple_num_latents = tuple_num_latents
    self.tuple_num_actions = tuple_num_actions
    self.num_abstates = num_abstates  # num abstract states
    self.num_mc_4_qz = num_mc_4_qz
    self.num_mc_4_qx = num_mc_4_qx

    self.list_np_policy = [None for dummy_i in range(self.num_agents)
                           ]  # type: list[np.ndarray]
    self.tx_dependency = trans_x_dependency
    self.list_Tx = []  # type: list[TransitionX]
    self.list_bx = [None for dummy_i in range(self.num_agents)
                    ]  # type: list[np.ndarray]
    self.np_prob_abstate = None

    self.max_iteration = max_iteration
    self.epsilon_g = epsilon_g
    self.epsilon_l = epsilon_l

    self.list_param_beta = None  # type: list[np.ndarray]
    self.list_param_bx = None  # type: list[np.ndarray]
    self.list_param_pi = None  # type: list[np.ndarray]
    self.param_abs = None  # type: np.ndarray
    self.lr = lr
    self.decay = decay

  def set_prior(self, gem_prior: float, tx_prior: float, pi_prior: float,
                abs_prior: float):
    self.hyper_gem = gem_prior
    self.hyper_tx = tx_prior
    self.hyper_pi = pi_prior
    self.hyper_abs = abs_prior

  def forward_backward_messaging(self, idx_a, trajectory, np_q_z, np_pi_tilde,
                                 np_tx_tilde, np_bx_tilde):
    n_lat = self.tuple_num_latents[idx_a]
    # Forward messaging
    with np.errstate(divide='ignore'):
      np_log_forward = np.log(np.zeros((len(trajectory), n_lat)))

    t = 0
    _, joint_a_p, _ = trajectory[t]

    with np.errstate(divide='ignore'):
      np_log_forward[t] = 0.0
      np_log_forward[t] += np.log(
          (np_q_z[t] @ np_bx_tilde) *
          (np_pi_tilde[:, :, joint_a_p[idx_a]] @ np_q_z[t]))

    # t = 1:N-1
    for t in range(1, len(trajectory)):
      t_p = t - 1
      _, joint_a, _ = trajectory[t]

      with np.errstate(divide='ignore'):
        np_log_prob = np_log_forward[t_p][:, None]

        tx_index = (slice(None), *joint_a_p, slice(None), slice(None))
        np_log_prob = np_log_prob + np.log(
            np.tensordot(np_tx_tilde[tx_index], np_q_z[t], axes=(1, 0)))

        np_log_prob = np_log_prob + np.log(
            np_pi_tilde[:, :, joint_a[idx_a]] @ np_q_z[t])[None, :]

      np_log_forward[t] = logsumexp(np_log_prob, axis=0)

      joint_a_p = joint_a

    # Backward messaging
    with np.errstate(divide='ignore'):
      np_log_backward = np.log(np.zeros((len(trajectory), n_lat)))
    # t = N-1
    t = len(trajectory) - 1

    _, joint_a_n, _ = trajectory[t]

    np_log_backward[t] = 0.0

    # t = 0:N-2
    for t in reversed(range(0, len(trajectory) - 1)):
      t_n = t + 1
      _, joint_a, _ = trajectory[t]

      with np.errstate(divide='ignore'):
        np_log_prob = np_log_backward[t_n][None, :]

        tx_index = (slice(None), *joint_a, slice(None), slice(None))
        np_log_prob = np_log_prob + np.log(
            np.tensordot(np_tx_tilde[tx_index], np_q_z[t_n], axes=(1, 0)))

        np_log_prob = np_log_prob + np.log(
            np_pi_tilde[:, :, joint_a_n[idx_a]] @ np_q_z[t_n])[None, :]

      np_log_backward[t] = logsumexp(np_log_prob, axis=1)  # noqa: E501

      joint_a_n = joint_a

    # compute q_x, q_x_xp
    log_q_x = np_log_forward + np_log_backward

    q_x = softmax(log_q_x, axis=1)

    with np.errstate(divide='ignore'):
      log_q_x_xn = np.log(np.zeros((len(trajectory) - 1, n_lat, n_lat)))

    for t in range(len(trajectory) - 1):
      _, joint_a, _ = trajectory[t]
      _, joint_a_n, _ = trajectory[t + 1]

      with np.errstate(divide='ignore'):
        log_q_x_xn[t] = (np_log_forward[t][:, None] +
                         np_log_backward[t + 1][None, :])
        tx_index = (slice(None), *joint_a, slice(None), slice(None))
        log_q_x_xn[t] += np.log(
            np.tensordot(np_tx_tilde[tx_index], np_q_z[t + 1], axes=(1, 0)))

        log_q_x_xn[t] += np.log(
            np_pi_tilde[:, :, joint_a_n[idx_a]] @ np_q_z[t + 1])[None, :]

    q_x_xn = softmax(log_q_x_xn, axis=(1, 2))

    return q_x, q_x_xn

  def compute_q_z(self, idx_a, trajectory, np_abs_tilde, np_pi_tilde,
                  np_tx_tilde, np_bx_tilde, np_q_x, np_q_xx):
    q_z = np.zeros((len(trajectory), self.num_abstates))

    t = 0
    stt_p, joint_a_p, _ = trajectory[t]
    q_z[t] = (np_abs_tilde[stt_p] * (np_bx_tilde @ np_q_x[t]) *
              (np_q_x[t] @ np_pi_tilde[:, :, joint_a_p[idx_a]]))

    for t in range(1, len(trajectory)):
      stt, joint_a, _ = trajectory[t]

      tx_index = (slice(None), *joint_a_p, slice(None), slice(None))
      q_z[t] = (np_abs_tilde[stt] * np.tensordot(
          np_tx_tilde[tx_index], np_q_xx[t - 1], axes=((0, 2), (0, 1))) *
                (np_q_x[t] @ np_pi_tilde[:, :, joint_a[idx_a]]))

      joint_a_p = joint_a

    q_z = q_z / np.sum(q_z, axis=1)[..., None]

    return q_z

  def compute_local_variables(self, idx_a, samples):
    list_q_z = []
    list_q_x = []
    list_q_x_xn = []

    np_abs_tilde = self.get_prob_tilda_from_lambda(self.param_abs)
    np_pi_tilde = self.get_prob_tilda_from_lambda(self.list_param_pi[idx_a])
    np_bx_tilde = self.get_prob_tilda_from_lambda(self.list_param_bx[idx_a])
    np_tx_tilde = self.get_prob_tilda_from_lambda(
        self.list_Tx[idx_a].np_lambda_Tx)

    np_p_sz_norm = np_abs_tilde / np.sum(np_abs_tilde, axis=1)[..., None]

    for m_th in range(len(samples)):
      trajectory = samples[m_th]
      len_traj = len(trajectory)

      # initialize q_z
      np_q_z = np.zeros((len_traj, self.num_abstates))
      for t, (stt, _, _) in enumerate(trajectory):
        np_q_z[t] = np_p_sz_norm[stt]

      # run until convergence
      for dummy_i in range(10):
        prev_np_q_z = np.copy(np_q_z)

        # compute q_x, q_xx
        np_q_x, np_q_xx = self.forward_backward_messaging(
            idx_a, trajectory, np_q_z, np_pi_tilde, np_tx_tilde, np_bx_tilde)

        # compute q_z
        np_q_z = self.compute_q_z(idx_a, trajectory, np_abs_tilde, np_pi_tilde,
                                  np_tx_tilde, np_bx_tilde, np_q_x, np_q_xx)

        # check convergence
        delta = np.max(abs(np_q_z - prev_np_q_z))
        if delta < self.epsilon_l:
          break

      list_q_x.append(np_q_x)
      list_q_x_xn.append(np_q_xx)
      list_q_z.append(np_q_z)

    return list_q_x, list_q_x_xn, list_q_z

  def update_global_variables(self, samples, lr,
                              list_list_q_x: Sequence[Sequence[np.ndarray]],
                              list_list_q_x_xn: Sequence[Sequence[np.ndarray]],
                              list_list_q_z: Sequence[Sequence[np.ndarray]]):

    batch_ratio = len(self.trajectories) / len(samples)
    param_abs_hat = np.zeros_like(self.param_abs)
    for idx_a in range(self.num_agents):
      param_pi_hat = np.zeros_like(self.list_param_pi[idx_a])
      param_bx_hat = np.zeros_like(self.list_param_bx[idx_a])
      param_tx_hat = np.zeros_like(self.list_Tx[idx_a].np_lambda_Tx)

      # -- compute sufficient statistics
      for m_th in range(len(samples)):
        traj = samples[m_th]
        q_x = list_list_q_x[idx_a][m_th]
        q_xx = list_list_q_x_xn[idx_a][m_th]
        q_z = list_list_q_z[idx_a][m_th]

        t = 0
        state, joint_a, _ = traj[t]
        param_abs_hat[state] += q_z[t]
        q_zx_tmp = q_z[t, :, None] * q_x[t, None, :]
        param_bx_hat += q_zx_tmp
        param_pi_hat[:, :, joint_a[idx_a]] += q_zx_tmp.transpose()

        for t in range(1, len(traj)):
          tp = t - 1
          state_p, joint_a_p, _ = traj[tp]
          state, joint_a, _ = traj[t]

          param_abs_hat[state] += q_z[t]
          tx_index = (slice(None), *joint_a_p, slice(None), slice(None))
          param_tx_hat[tx_index] += q_xx[tp][:, None, :] * q_z[t][None, :, None]
          param_pi_hat[:, :,
                       joint_a[idx_a]] += q_x[t, :, None] * q_z[t, None, :]

      # - update agent-level global variables
      x_prior = self.hyper_tx * self.list_param_beta[idx_a]
      param_pi_hat = batch_ratio * param_pi_hat + self.hyper_pi
      param_bx_hat = batch_ratio * param_bx_hat + x_prior
      param_tx_hat = batch_ratio * param_tx_hat + x_prior

      self.list_param_pi[idx_a] = ((1 - lr) * self.list_param_pi[idx_a] +
                                   lr * param_pi_hat)
      self.list_param_bx[idx_a] = ((1 - lr) * self.list_param_bx[idx_a] +
                                   lr * param_bx_hat)
      self.list_Tx[idx_a].np_lambda_Tx = (
          (1 - lr) * self.list_Tx[idx_a].np_lambda_Tx + lr * param_tx_hat)

      # -- update beta.
      # ref: https://people.eecs.berkeley.edu/~jordan/papers/liang-jordan-klein-haba.pdf
      # ref: http://proceedings.mlr.press/v32/johnson14.pdf
      num_K = len(self.list_param_beta[idx_a]) - 1
      grad_ln_p_beta = (np.ones(num_K) * (1 - self.hyper_gem) /
                        self.list_param_beta[idx_a][-1])
      for k in range(num_K):
        for i in range(k + 1, num_K):
          sum_beta = np.sum(self.list_param_beta[idx_a][:i])
          grad_ln_p_beta[k] += 1 / (1 - sum_beta)

      const_tmp = -digamma(x_prior) + digamma(sum(x_prior))
      param_tx = self.list_Tx[idx_a].np_lambda_Tx
      param_bx = self.list_param_bx[idx_a]
      sum_param_tx = np.sum(param_tx, axis=-1)
      sum_param_bx = np.sum(param_bx, axis=-1)
      grad_ln_E_p_tx = np.sum(digamma(param_tx) -
                              digamma(sum_param_tx)[..., None],
                              axis=tuple(range(param_tx.ndim - 1)))
      grad_ln_E_p_tx += np.sum(digamma(param_bx) -
                               digamma(sum_param_bx)[..., None],
                               axis=tuple(range(param_bx.ndim - 1)))
      num_x_dists = np.prod(param_tx.shape[:-1]) + np.prod(param_bx.shape[:-1])
      grad_ln_E_p_tx = (self.hyper_tx * grad_ln_E_p_tx[:-1] +
                        num_x_dists * const_tmp[:-1])

      grad_beta = grad_ln_p_beta + grad_ln_E_p_tx
      grad_beta_norm = np.linalg.norm(grad_beta)
      grad_beta /= grad_beta_norm

      # line search
      reach = np.zeros(num_K + 1)
      # distance to each canonical hyperplane
      reach[:-1] = -self.list_param_beta[idx_a][:-1] / grad_beta
      # signed distance to all-one hyperplane
      reach[-1] = self.list_param_beta[idx_a][-1] / np.sum(grad_beta)
      max_reach = min(reach[reach > 0])
      search_reach = min(max_reach, grad_beta_norm)

      self.list_param_beta[idx_a][:-1] = (self.list_param_beta[idx_a][:-1] +
                                          lr * search_reach * grad_beta)
      self.list_param_beta[idx_a][-1] = (
          1 - np.sum(self.list_param_beta[idx_a][:-1]))

    # - update task-level global variables
    param_abs_hat = batch_ratio * param_abs_hat + self.hyper_abs
    self.param_abs = ((1 - lr) * self.param_abs + lr * param_abs_hat)

  def get_prob_tilda_from_lambda(self, np_lambda):

    sum_lambda_pi = np.sum(np_lambda, axis=-1)
    ln_pi = digamma(np_lambda) - digamma(sum_lambda_pi)[..., None]
    return np.exp(ln_pi)

  def initialize_param(self):
    INIT_RANGE = (1, 1.1)

    self.list_param_beta = []
    self.list_param_bx = []
    self.list_param_pi = []
    # init abstraction parameters
    self.param_abs = np.random.uniform(low=INIT_RANGE[0],
                                       high=INIT_RANGE[1],
                                       size=(self.num_ostates,
                                             self.num_abstates))

    for idx_a in range(self.num_agents):
      num_K = self.tuple_num_latents[idx_a] - 1
      # init beta
      tmp_np_v = np.random.beta(1, self.hyper_gem, num_K)
      tmp_np_beta = np.zeros(num_K + 1)
      for idx in range(num_K):
        tmp_np_beta[idx] = tmp_np_v[idx]
        for pidx in range(idx):
          tmp_np_beta[idx] *= 1 - tmp_np_v[pidx]
      tmp_np_beta[-1] = 1 - np.sum(tmp_np_beta[:-1])
      self.list_param_beta.append(tmp_np_beta)
      # init bx param
      self.list_param_bx.append(
          np.random.uniform(low=INIT_RANGE[0],
                            high=INIT_RANGE[1],
                            size=(self.num_abstates, num_K + 1)))
      self.list_param_pi.append(
          np.random.uniform(low=INIT_RANGE[0],
                            high=INIT_RANGE[1],
                            size=(num_K + 1, self.num_abstates,
                                  self.tuple_num_actions[idx_a])))
      # init tx param
      num_s = self.num_abstates if self.tx_dependency[0] else None
      list_num_a = []
      for i_a in range(self.num_agents):
        if self.tx_dependency[i_a + 1]:
          list_num_a.append(self.tuple_num_actions[i_a])
        else:
          list_num_a.append(None)

      num_sn = self.num_abstates if self.tx_dependency[-1] else None

      var_param_tx = TransitionX(num_K + 1, num_s, tuple(list_num_a), num_sn,
                                 num_K + 1)
      var_param_tx.init_lambda_Tx(*INIT_RANGE)
      self.list_Tx.append(var_param_tx)

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
          perm_index = np.random.permutation(len(self.trajectories))

        samples = [
            self.trajectories[idx]
            for idx in perm_index[batch_idx * batch_size:(batch_idx + 1) *
                                  batch_size]
        ]

      count += 1

      delta_team = 0
      prev_param_abs = np.copy(self.param_abs)
      prev_list_param_bx = []
      prev_list_param_pi = []
      prev_list_param_tx = []

      list_list_q_x = []
      list_list_q_xx = []
      list_list_q_z = []
      for idx_a in range(self.num_agents):
        prev_list_param_bx.append(np.copy(self.list_param_bx[idx_a]))
        prev_list_param_pi.append(np.copy(self.list_param_pi[idx_a]))
        prev_list_param_tx.append(np.copy(self.list_Tx[idx_a].np_lambda_Tx))

        list_q_x, list_q_x_xn, list_q_z = self.compute_local_variables(
            idx_a, samples)  # TODO: use batch
        list_list_q_x.append(list_q_x)
        list_list_q_xx.append(list_q_x_xn)
        list_list_q_z.append(list_q_z)

      # lr = (count + 1)**(-self.forgetting_rate)
      lr = self.lr / (count * self.decay + 1)
      self.update_global_variables(samples, lr, list_list_q_x, list_list_q_xx,
                                   list_list_q_z)

      # compute delta
      for idx_a in range(self.num_agents):
        delta_team = max(
            delta_team,
            np.max(np.abs(self.list_param_bx[idx_a] -
                          prev_list_param_bx[idx_a])))
        delta_team = max(
            delta_team,
            np.max(np.abs(self.list_param_pi[idx_a] -
                          prev_list_param_pi[idx_a])))
        delta_team = max(
            delta_team,
            np.max(
                np.abs(self.list_Tx[idx_a].np_lambda_Tx -
                       prev_list_param_tx[idx_a])))
        delta_team = max(delta_team,
                         np.max(np.abs(self.param_abs - prev_param_abs)))

      if delta_team < self.epsilon_g:
        break

      if count % 10 == 0:
        print("Save parameters...")
        self.save_params()
        print("Finished saving")

      progress_bar.update()
      progress_bar.set_postfix({'delta': delta_team})
    progress_bar.close()

    self.convert_params_to_prob()

  def convert_params_to_prob(self):
    for i_a in range(self.num_agents):
      numerator = self.list_param_pi[i_a]
      action_sums = np.sum(numerator, axis=-1)
      self.list_np_policy[i_a] = numerator / action_sums[..., np.newaxis]

      self.list_Tx[i_a].conv_to_Tx()

      numerator = self.list_param_bx[i_a]
      latent_sums = np.sum(numerator, axis=-1)
      self.list_bx[i_a] = numerator / latent_sums[..., np.newaxis]

    numerator = self.param_abs
    abstract_sums = np.sum(numerator, axis=-1)
    self.np_prob_abstate = self.param_abs / abstract_sums[..., np.newaxis]

  def save_params(self):

    for idx in range(self.num_agents):
      np.save(self.save_prefix + "_param_pi" + f"_a{idx + 1}",
              self.list_param_pi[idx])
      np.save(self.save_prefix + "_param_tx" + f"_a{idx + 1}",
              self.list_Tx[idx].np_lambda_Tx)
      np.save(self.save_prefix + "_param_bx" + f"_a{idx + 1}",
              self.list_param_bx[idx])
      np.save(self.save_prefix + "_param_beta" + f"_a{idx + 1}", self.list_param_beta[idx])

    np.save(self.save_prefix + "_param_abs", self.param_abs)
  
  def load_params(self):
    self.initialize_param()
    for idx in range(self.num_agents):
      self.list_param_pi[idx] = np.load(self.save_prefix + "_param_pi" + f"_a{idx + 1}.npy")
      self.list_Tx[idx].np_lambda_Tx = np.load(self.save_prefix + "_param_tx" + f"_a{idx + 1}.npy")
      self.list_param_bx[idx] = np.load(self.save_prefix + "_param_bx" + f"_a{idx + 1}.npy")
      self.list_param_beta[idx] = np.load(self.save_prefix + "_param_beta" + f"_a{idx + 1}.npy")

    self.param_abs = np.load(self.save_prefix + "_param_abs.npy")