from typing import Tuple, Sequence
import numpy as np
from tqdm import tqdm
from scipy.special import digamma, logsumexp, softmax
from aic_ml.BTIL.transition_x import TransitionX

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
      lr_beta: float = 0.001,
      decay: float = 0.01,
      num_abstates: int = 30,
      save_file_prefix: str = None,
      no_gem: bool = False) -> None:
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
    self.list_hyper_pi = [HYPER_PI / n_a for n_a in tuple_num_actions]
    self.hyper_abs = HYPER_ABS / num_abstates
    self.no_gem = no_gem

    self.num_agents = len(tuple_num_actions)
    self.num_ostates = num_states
    self.tuple_num_latents = tuple_num_latents
    self.tuple_num_actions = tuple_num_actions
    self.num_abstates = num_abstates  # num abstract states

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
    self.lr_beta = lr_beta
    self.decay = decay
    self.dict_qz = {}

  def set_prior(self, gem_prior: float, tx_prior: float, pi_prior: float,
                abs_prior: float):
    self.hyper_gem = gem_prior
    self.hyper_tx = tx_prior
    self.list_hyper_pi = [pi_prior / n_a for n_a in self.tuple_num_actions]
    self.hyper_abs = abs_prior / self.num_abstates

  def forward_backward_messaging(self, idx_a, trajectory, np_q_z, np_pi_tilde,
                                 np_tx_tilde, np_bx_tilde):
    n_lat = self.tuple_num_latents[idx_a]
    # Forward messaging
    with np.errstate(divide='ignore'):
      np_log_forward = np.log(np.zeros((len(trajectory), n_lat)))

    t = 0
    _, joint_a_p, joint_x_p = trajectory[t]

    idx_xp = joint_x_p[idx_a]
    len_xp = 1
    if joint_x_p[idx_a] is None:
      idx_xp = slice(None)
      len_xp = n_lat

    with np.errstate(divide='ignore'):
      np_log_forward[t][idx_xp] = 0.0
      np_log_forward[t][idx_xp] += np.log(
          (np_q_z[t] @ np_bx_tilde)[idx_xp] *
          (np_pi_tilde[idx_xp, :, joint_a_p[idx_a]] @ np_q_z[t]))

    # t = 1:N-1
    for t in range(1, len(trajectory)):
      t_p = t - 1
      _, joint_a, joint_x = trajectory[t]

      idx_x = joint_x[idx_a]
      len_x = 1
      if joint_x[idx_a] is None:
        idx_x = slice(None)
        len_x = n_lat

      with np.errstate(divide='ignore'):
        np_log_prob = np_log_forward[t_p][idx_xp].reshape(len_xp, 1)

        tx_index = (slice(None), *joint_a_p, slice(None), slice(None))
        np_log_prob = np_log_prob + np.log(
            np.tensordot(np_tx_tilde[tx_index][idx_xp, :, idx_x].reshape(
                len_xp, -1, len_x),
                         np_q_z[t],
                         axes=(1, 0)))

        np_log_prob = np_log_prob + np.log(
            np_pi_tilde[idx_x, :, joint_a[idx_a]] @ np_q_z[t]).reshape(
                1, len_x)

      np_log_forward[t] = logsumexp(np_log_prob, axis=0)

      joint_a_p = joint_a
      idx_xp = idx_x
      len_xp = len_x

    # Backward messaging
    with np.errstate(divide='ignore'):
      np_log_backward = np.log(np.zeros((len(trajectory), n_lat)))
    # t = N-1
    t = len(trajectory) - 1

    _, joint_a_n, joint_x_n = trajectory[t]

    idx_xn = joint_x_n[idx_a]
    len_xn = 1
    if joint_x_n[idx_a] is None:
      idx_xn = slice(None)
      len_xn = n_lat

    np_log_backward[t][idx_xn] = 0.0

    # t = 0:N-2
    for t in reversed(range(0, len(trajectory) - 1)):
      t_n = t + 1
      _, joint_a, joint_x = trajectory[t]

      idx_x = joint_x[idx_a]
      len_x = 1
      if joint_x[idx_a] is None:
        idx_x = slice(None)
        len_x = n_lat

      with np.errstate(divide='ignore'):
        np_log_prob = np_log_backward[t_n][idx_xn].reshape(1, len_xn)

        tx_index = (slice(None), *joint_a, slice(None), slice(None))
        np_log_prob = np_log_prob + np.log(
            np.tensordot(np_tx_tilde[tx_index][idx_x, :, idx_xn].reshape(
                len_x, -1, len_xn),
                         np_q_z[t_n],
                         axes=(1, 0)))

        np_log_prob = np_log_prob + np.log(
            np_pi_tilde[idx_xn, :, joint_a_n[idx_a]] @ np_q_z[t_n]).reshape(
                1, len_xn)

      np_log_backward[t] = logsumexp(np_log_prob, axis=1)  # noqa: E501

      joint_a_n = joint_a
      idx_xn = idx_x
      len_xn = len_x

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

  def compute_q_z(self, trajectory, np_abs_tilde, list_np_pi_tilde,
                  list_np_tx_tilde, list_np_bx_tilde, list_np_q_x,
                  list_np_q_xx):
    with np.errstate(divide='ignore'):
      log_q_z = np.log(np.zeros((len(trajectory), self.num_abstates)))

    t = 0
    stt_p, joint_a_p, _ = trajectory[t]
    log_q_z[t] = np.log(np_abs_tilde[stt_p])
    for idx_a in range(self.num_agents):
      log_q_z[t] += (np.log(
          (list_np_bx_tilde[idx_a][t] @ list_np_q_x[idx_a][t])) +
                     np.log(list_np_q_x[idx_a][t]
                            @ list_np_pi_tilde[idx_a][:, :, joint_a_p[idx_a]]))

    for t in range(1, len(trajectory)):
      stt, joint_a, _ = trajectory[t]

      tx_index = (slice(None), *joint_a_p, slice(None), slice(None))
      log_q_z[t] = np.log(np_abs_tilde[stt])
      for idx_a in range(self.num_agents):
        log_q_z[t] += (np.log(
            np.tensordot(list_np_tx_tilde[idx_a][tx_index],
                         list_np_q_xx[idx_a][t - 1],
                         axes=((0, 2), (0, 1)))) + np.log(
                             (list_np_q_x[idx_a][t]
                              @ list_np_pi_tilde[idx_a][:, :, joint_a[idx_a]])))

      joint_a_p = joint_a

    q_z = softmax(log_q_z, axis=1)

    return q_z

  def compute_local_variables(self, samples):
    list_q_z = []
    list_list_q_x = []
    list_list_q_xx = []

    np_abs_tilde = self.get_prob_tilda_from_lambda(self.param_abs)
    list_np_pi_tilde = [
        self.get_prob_tilda_from_lambda(self.list_param_pi[idx_a])
        for idx_a in range(self.num_agents)
    ]
    list_np_bx_tilde = [
        self.get_prob_tilda_from_lambda(self.list_param_bx[idx_a])
        for idx_a in range(self.num_agents)
    ]

    list_np_tx_tilde = [
        self.get_prob_tilda_from_lambda(self.list_Tx[idx_a].np_lambda_Tx)
        for idx_a in range(self.num_agents)
    ]

    np_p_sz_norm = np_abs_tilde / np.sum(np_abs_tilde, axis=1)[..., None]

    for m_th in range(len(samples)):
      sam_idx = samples[m_th][0]
      trajectory = samples[m_th][1]
      len_traj = len(trajectory)

      # initialize q_z
      if sam_idx not in self.dict_qz:
        np_q_z = np.zeros((len_traj, self.num_abstates))
        for t, (stt, _, _) in enumerate(trajectory):
          np_q_z[t] = np_p_sz_norm[stt]
        self.dict_qz[sam_idx] = np_q_z
      else:
        np_q_z = self.dict_qz[sam_idx]

      list_np_q_x = [None for _ in range(self.num_agents)]
      list_np_q_xx = [None for _ in range(self.num_agents)]
      # run until convergence
      for dummy_i in range(10):
        prev_np_q_z = np.copy(np_q_z)

        for idx_a in range(self.num_agents):
          # compute q_x, q_xx
          np_q_x, np_q_xx = self.forward_backward_messaging(
              idx_a, trajectory, np_q_z, list_np_pi_tilde[idx_a],
              list_np_tx_tilde[idx_a], list_np_bx_tilde[idx_a])
          list_np_q_x[idx_a] = np_q_x
          list_np_q_xx[idx_a] = np_q_xx

        # compute q_z
        np_q_z = self.compute_q_z(trajectory, np_abs_tilde, list_np_pi_tilde,
                                  list_np_tx_tilde, list_np_bx_tilde,
                                  list_np_q_x, list_np_q_xx)

        # check convergence
        delta = np.max(abs(np_q_z - prev_np_q_z))
        if delta < self.epsilon_l:
          break

      self.dict_qz[sam_idx] = np_q_z

      list_list_q_x.append(list_np_q_x)
      list_list_q_xx.append(list_np_q_xx)
      list_q_z.append(np_q_z)

    return list_list_q_x, list_list_q_xx, list_q_z

  def update_global_variables(self, samples, lr, lr_beta,
                              list_list_q_x: Sequence[Sequence[np.ndarray]],
                              list_list_q_xx: Sequence[Sequence[np.ndarray]],
                              list_q_z: Sequence[np.ndarray]):

    batch_ratio = len(self.trajectories) / len(samples)
    param_abs_hat = np.zeros_like(self.param_abs)
    list_param_pi_hat = [
        np.zeros_like(self.list_param_pi[idx_a])
        for idx_a in range(self.num_agents)
    ]
    list_param_bx_hat = [
        np.zeros_like(self.list_param_bx[idx_a])
        for idx_a in range(self.num_agents)
    ]
    list_param_tx_hat = [
        np.zeros_like(self.list_Tx[idx_a].np_lambda_Tx)
        for idx_a in range(self.num_agents)
    ]

    # -- compute sufficient statistics
    for m_th in range(len(samples)):
      traj = samples[m_th][1]
      q_z = list_q_z[m_th]
      list_q_x = list_list_q_x[m_th]
      list_q_xx = list_list_q_xx[m_th]

      t = 0
      state, joint_a, _ = traj[t]
      param_abs_hat[state] += q_z[t]
      for idx_a in range(self.num_agents):
        q_zx_tmp = q_z[t, :, None] * list_q_x[idx_a][t, None, :]
        list_param_bx_hat[idx_a] += q_zx_tmp
        list_param_pi_hat[idx_a][:, :, joint_a[idx_a]] += q_zx_tmp.transpose()

      for t in range(1, len(traj)):
        tp = t - 1
        state_p, joint_a_p, _ = traj[tp]
        state, joint_a, _ = traj[t]
        param_abs_hat[state] += q_z[t]
        tx_index = (slice(None), *joint_a_p, slice(None), slice(None))
        for idx_a in range(self.num_agents):
          q_xx = list_q_xx[idx_a]
          q_x = list_q_x[idx_a]
          list_param_tx_hat[idx_a][tx_index] += (q_xx[tp][:, None, :] *
                                                 q_z[t][None, :, None])
          list_param_pi_hat[idx_a][:, :, joint_a[idx_a]] += (q_x[t, :, None] *
                                                             q_z[t, None, :])

    # - update agent-level global variables
    for idx_a in range(self.num_agents):
      x_prior = self.hyper_tx * self.list_param_beta[idx_a]
      list_param_pi_hat[idx_a] = (batch_ratio * list_param_pi_hat[idx_a] +
                                  self.list_hyper_pi[idx_a])
      list_param_bx_hat[idx_a] = (batch_ratio * list_param_bx_hat[idx_a] +
                                  x_prior)
      list_param_tx_hat[idx_a] = (batch_ratio * list_param_tx_hat[idx_a] +
                                  x_prior)

      self.list_param_pi[idx_a] = ((1 - lr) * self.list_param_pi[idx_a] +
                                   lr * list_param_pi_hat[idx_a])
      self.list_param_bx[idx_a] = ((1 - lr) * self.list_param_bx[idx_a] +
                                   lr * list_param_bx_hat[idx_a])
      self.list_Tx[idx_a].np_lambda_Tx = (
          (1 - lr) * self.list_Tx[idx_a].np_lambda_Tx +
          lr * list_param_tx_hat[idx_a])

      if not self.no_gem:
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
        num_x_dists = np.prod(param_tx.shape[:-1]) + np.prod(
            param_bx.shape[:-1])
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
                                            lr_beta * search_reach * grad_beta)
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
      num_x = self.tuple_num_latents[idx_a]
      # init beta
      if self.no_gem:
        self.list_param_beta.append(np.ones(num_x) / num_x)
      else:
        tmp_np_v = np.random.beta(1, self.hyper_gem, num_x - 1)
        tmp_np_beta = np.zeros(num_x)
        for idx in range(num_x - 1):
          tmp_np_beta[idx] = tmp_np_v[idx]
          for pidx in range(idx):
            tmp_np_beta[idx] *= 1 - tmp_np_v[pidx]
        tmp_np_beta[-1] = 1 - np.sum(tmp_np_beta[:-1])
        self.list_param_beta.append(tmp_np_beta)
      # init bx param
      self.list_param_bx.append(
          np.random.uniform(low=INIT_RANGE[0],
                            high=INIT_RANGE[1],
                            size=(self.num_abstates, num_x)))
      self.list_param_pi.append(
          np.random.uniform(low=INIT_RANGE[0],
                            high=INIT_RANGE[1],
                            size=(num_x, self.num_abstates,
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

      var_param_tx = TransitionX(num_x, num_s, tuple(list_num_a), num_sn, num_x)
      var_param_tx.init_lambda_Tx(*INIT_RANGE)
      self.list_Tx.append(var_param_tx)

  def do_inference(self, batch_size):
    num_traj = len(self.trajectories)
    traj_w_index = [(idx, traj) for idx, traj in enumerate(self.trajectories)]
    batch_iter = int(num_traj / batch_size)
    count = 0
    progress_bar = tqdm(total=self.max_iteration)
    while count < self.max_iteration:
      if batch_size >= num_traj:
        samples = traj_w_index
      else:
        batch_idx = count % batch_iter
        if batch_idx == 0:
          perm_index = np.random.permutation(len(traj_w_index))

        samples = [
            traj_w_index[idx]
            for idx in perm_index[batch_idx * batch_size:(batch_idx + 1) *
                                  batch_size]
        ]

      count += 1

      delta_team = 0
      prev_param_abs = np.copy(self.param_abs)
      prev_list_param_bx = [
          np.copy(self.list_param_bx[idx_a]) for idx_a in range(self.num_agents)
      ]
      prev_list_param_pi = [
          np.copy(self.list_param_pi[idx_a]) for idx_a in range(self.num_agents)
      ]
      prev_list_param_tx = [
          np.copy(self.list_Tx[idx_a].np_lambda_Tx)
          for idx_a in range(self.num_agents)
      ]

      list_list_q_x, list_list_q_xx, list_q_z = self.compute_local_variables(
          samples)  # TODO: use batch

      # lr = (count + 1)**(-self.forgetting_rate)
      lr = self.lr / (count * self.decay + 1)
      lr_beta = self.lr_beta / (count * self.decay + 1)
      self.update_global_variables(samples, lr, lr_beta, list_list_q_x,
                                   list_list_q_xx, list_q_z)

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

      if count % 50 == 0:
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
      np.save(self.save_prefix + "_param_beta" + f"_a{idx + 1}",
              self.list_param_beta[idx])

    np.save(self.save_prefix + "_param_abs", self.param_abs)

  def load_params(self):
    self.initialize_param()
    for idx in range(self.num_agents):
      self.list_param_pi[idx] = np.load(self.save_prefix + "_param_pi" +
                                        f"_a{idx + 1}.npy")
      self.list_Tx[idx].np_lambda_Tx = np.load(self.save_prefix + "_param_tx" +
                                               f"_a{idx + 1}.npy")
      self.list_param_bx[idx] = np.load(self.save_prefix + "_param_bx" +
                                        f"_a{idx + 1}.npy")
      self.list_param_beta[idx] = np.load(self.save_prefix + "_param_beta" +
                                          f"_a{idx + 1}.npy")

    self.param_abs = np.load(self.save_prefix + "_param_abs.npy")
