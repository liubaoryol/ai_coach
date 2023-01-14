from typing import Optional, Tuple, Callable, Sequence
import numpy as np
import copy
from tqdm import tqdm
from scipy.special import digamma, logsumexp, softmax
from ai_coach_core.model_learning.BTIL.transition_x import TransitionX

T_SAXSeqence = Sequence[Tuple[int, Tuple[int, int], Tuple[int, int]]]


class BTIL_SVI:

  def __init__(
      self,
      trajectories: Sequence[T_SAXSeqence],
      num_states: int,
      tuple_num_latents: Tuple[int, ...],
      tuple_num_actions: Tuple[int, ...],
      trans_x_dependency=(True, True, True, True, False),  # s, a1, a2, a3, sn
      max_iteration: int = 1000,
      epsilon: float = 0.001,
      forgetting_rate: float = 0.99) -> None:
    '''
      trajectories: list of list of (state, joint action)-tuples
      tuple_num_latents: truncated stick-breaking number + 1
    '''

    assert len(tuple_num_actions) + 2 == len(trans_x_dependency)

    PARAM_GEM_GAMMA = 3
    PARAM_TX_ALPHA = 3
    PARAM_PI_RHO = 3

    self.trajectories = trajectories

    self.param_gem_gamma = PARAM_GEM_GAMMA
    self.param_tx_alpha = PARAM_TX_ALPHA
    self.param_pi_rho = PARAM_PI_RHO
    self.num_agents = len(tuple_num_actions)
    self.num_ostates = num_states
    self.tuple_num_latents = tuple_num_latents
    self.tuple_num_actions = tuple_num_actions

    self.list_np_policy = [None for dummy_i in range(self.num_agents)
                           ]  # type: list[np.ndarray]
    self.tx_dependency = trans_x_dependency
    self.list_Tx = []  # type: list[TransitionX]
    self.list_bx = [None for dummy_i in range(self.num_agents)
                    ]  # type: list[np.ndarray]
    self.cb_Tx = None
    self.cb_bx = None

    self.max_iteration = max_iteration
    self.epsilon = epsilon

    self.var_param_beta = None  # type: list[np.ndarray]
    self.var_param_bx = None  # type: list[np.ndarray]
    self.var_param_pi = None  # type: list[np.ndarray]
    self.forgetting_rate = forgetting_rate

  def set_bx_and_Tx(self, cb_bx, cb_Tx=None):
    self.cb_bx = cb_bx
    self.cb_Tx = cb_Tx

  def get_Tx(self, agent_idx, sidx, tup_aidx, sidx_n):
    if self.cb_Tx is not None:
      return self.cb_Tx(agent_idx, sidx, tup_aidx, sidx_n)
    else:
      return self.list_Tx[agent_idx].get_Tx_prop(sidx, tup_aidx, sidx_n)

  def get_bx(self, agent_idx, sidx):
    if self.cb_bx is not None:
      return self.cb_bx(agent_idx, sidx)
    else:
      return self.list_bx[agent_idx][sidx]

  def set_prior(self, gem_gamma: float, tx_alpha: float, pi_rho: float):
    self.param_gem_gamma = gem_gamma
    self.param_tx_alpha = tx_alpha
    self.param_pi_rho = pi_rho

  def compute_local_variables(self, idx_a, samples):
    list_q_x = []
    list_q_x_xn = []

    for m_th in range(len(samples)):
      trajectory = samples[m_th]

      n_lat = self.tuple_num_latents[idx_a]
      # Forward messaging
      with np.errstate(divide='ignore'):
        np_log_forward = np.log(np.zeros((len(trajectory), n_lat)))

      t = 0
      stt_p, joint_a_p, joint_x_p = trajectory[t]

      idx_xp = joint_x_p[idx_a]
      len_xp = 1
      if joint_x_p[idx_a] is None:
        idx_xp = slice(None)
        len_xp = n_lat

      with np.errstate(divide='ignore'):
        np_log_forward[t][idx_xp] = 0.0

        np_log_forward[t][idx_xp] += np.log(
            self.get_bx(idx_a, stt_p)[idx_xp] *
            self.list_np_policy[idx_a][idx_xp, stt_p, joint_a_p[idx_a]])

      # t = 1:N-1
      for t in range(1, len(trajectory)):
        t_p = t - 1
        stt, joint_a, joint_x = trajectory[t]

        idx_x = joint_x[idx_a]
        len_x = 1
        if joint_x[idx_a] is None:
          idx_x = slice(None)
          len_x = n_lat

        # yapf: disable
        with np.errstate(divide='ignore'):
          np_log_prob = np_log_forward[t_p][idx_xp].reshape(len_xp, 1)

          np_log_prob = np_log_prob + np.log(
            self.get_Tx(idx_a, stt_p, joint_a_p, stt)[idx_xp, idx_x].reshape(len_xp, len_x)  # noqa: E501
          )

          np_log_prob = np_log_prob + np.log(
            self.list_np_policy[idx_a][idx_x, stt, joint_a[idx_a]].reshape(1, len_x)  # noqa: E501
          )

        np_log_forward[t][idx_x] = logsumexp(np_log_prob, axis=0)
        # yapf: enable

        stt_p = stt
        joint_a_p = joint_a
        idx_xp = idx_x
        len_xp = len_x

      # Backward messaging
      with np.errstate(divide='ignore'):
        np_log_backward = np.log(np.zeros((len(trajectory), n_lat)))
      # t = N-1
      t = len(trajectory) - 1

      stt_n, joint_a_n, joint_x_n = trajectory[t]

      idx_xn = joint_x_n[idx_a]
      len_xn = 1
      if joint_x_n[idx_a] is None:
        idx_xn = slice(None)
        len_xn = n_lat

      np_log_backward[t][idx_xn] = 0.0

      # t = 0:N-2
      for t in reversed(range(0, len(trajectory) - 1)):
        t_n = t + 1
        stt, joint_a, joint_x = trajectory[t]

        idx_x = joint_x[idx_a]
        len_x = 1
        if joint_x[idx_a] is None:
          idx_x = slice(None)
          len_x = n_lat

        # yapf: disable
        with np.errstate(divide='ignore'):
          np_log_prob = np_log_backward[t_n][idx_xn].reshape(1, len_xn)  # noqa: E501

          np_log_prob = np_log_prob + np.log(
            self.get_Tx(idx_a, stt, joint_a, stt_n)[idx_x, idx_xn].reshape(len_x, len_xn)  # noqa: E501
          )

          np_log_prob = np_log_prob + np.log(
            self.list_np_policy[idx_a][idx_xn, stt_n, joint_a_n[idx_a]].reshape(1, len_xn)  # noqa: E501
          )

        np_log_backward[t][idx_x] = logsumexp(np_log_prob, axis=1)  # noqa: E501
        # yapf: enable

        stt_n = stt
        joint_a_n = joint_a
        idx_xn = idx_x
        len_xn = len_x

      # compute q_x, q_x_xp
      log_q_x = np_log_forward + np_log_backward

      q_x = softmax(log_q_x, axis=1)

      if self.cb_Tx is None:
        # n_x = self.num_lstates
        with np.errstate(divide='ignore'):
          log_q_x_xn = np.log(np.zeros((len(trajectory) - 1, n_lat, n_lat)))

        for t in range(len(trajectory) - 1):
          stt, joint_a, joint_x = trajectory[t]
          sttn, joint_a_n, joint_x_n = trajectory[t + 1]

          # yapf: disable
          with np.errstate(divide='ignore'):
            log_q_x_xn[t] = (
              np_log_forward[t].reshape(n_lat, 1) +
              np_log_backward[t + 1].reshape(1, n_lat)
            )

            log_q_x_xn[t] += np.log(
              self.list_Tx[idx_a].get_q_xxn(stt, joint_a, sttn)
            )

            log_q_x_xn[t] += np.log(
              self.list_np_policy[idx_a][:, sttn, joint_a_n[idx_a]].reshape(1, n_lat)
            )
          # yapf: enable

        q_x_xn = softmax(log_q_x_xn, axis=(1, 2))

      list_q_x.append(q_x)
      if self.cb_Tx is None:
        list_q_x_xn.append(q_x_xn)

    return list_q_x, list_q_x_xn

  def update_global_variables(self, idx_a, samples, lr,
                              list_q_x: Sequence[Sequence[np.ndarray]],
                              list_q_x_xn: Sequence[Sequence[np.ndarray]]):

    map_suf_stat_pi = {}
    map_suf_stat_bx = {}
    map_suf_stat_tx = {}

    for m_th in range(len(samples)):
      traj = samples[m_th]
      q_x = list_q_x[m_th]

      for t, (state, joint_a, joint_x) in enumerate(traj):
        pi_sa_key = (state, joint_a[idx_a])
        if pi_sa_key not in map_suf_stat_pi:
          map_suf_stat_pi[pi_sa_key] = np.zeros(self.tuple_num_latents[idx_a])
        map_suf_stat_pi[pi_sa_key] += q_x[t, :]

        if t == 0 and self.cb_bx is None:
          bx_s_key = state
          if bx_s_key not in map_suf_stat_bx:
            map_suf_stat_bx[bx_s_key] = np.zeros(self.tuple_num_latents[idx_a])
          map_suf_stat_bx[bx_s_key] += q_x[t, :]

        if t > 0 and self.cb_Tx is None:
          state_p, joint_a_p, _ = traj[t - 1]
          tx_sas_key = (state_p, joint_a_p, state)
          if tx_sas_key not in map_suf_stat_tx:
            map_suf_stat_tx[tx_sas_key] = np.zeros(
                (self.tuple_num_latents[idx_a], self.tuple_num_latents[idx_a]))
          map_suf_stat_tx[tx_sas_key] += list_q_x_xn[m_th][t - 1, :, :]

    # -- update
    batch_ratio = len(self.trajectories) / len(samples)
    x_prior_param = self.param_tx_alpha * self.var_param_beta[idx_a]

    for pi_key in map_suf_stat_pi:
      s, a = pi_key
      param_pi_hat = self.param_pi_rho + batch_ratio * map_suf_stat_pi[pi_key]
      self.var_param_pi[idx_a][:, s, a] = (
          (1 - lr) * self.var_param_pi[idx_a][:, s, a] + lr * param_pi_hat)

    for bx_key in map_suf_stat_bx:
      s = bx_key
      param_bx_hat = x_prior_param + batch_ratio * map_suf_stat_bx[bx_key]
      self.var_param_bx[idx_a][s, :] = (
          (1 - lr) * self.var_param_bx[idx_a][s, :] + lr * param_bx_hat)

    # prepare for beta update
    const_tmp = -digamma(x_prior_param) + digamma(sum(x_prior_param))
    grad_ln_E_p_tx = len(map_suf_stat_tx) * const_tmp[:-1]

    # update tx
    for tx_key in map_suf_stat_tx:
      s, joint_a, sn = tx_key
      param_tx_hat = (x_prior_param[None, :] +
                      batch_ratio * map_suf_stat_tx[tx_key])
      self.list_Tx[idx_a].update_lambda_Tx(s, joint_a, sn, param_tx_hat, lr)

      # for beta update
      param_tx = self.list_Tx[idx_a].get_lambda_Tx(s, joint_a, sn)
      sum_param_tx = np.sum(param_tx, axis=-1)
      ln_tx_avg = self.param_tx_alpha * (digamma(param_tx) -
                                         digamma(sum_param_tx)[..., None])

      grad_ln_E_p_tx += np.sum(ln_tx_avg, axis=0)[:-1]

    # TODO: check if cb_Tx and cb_bx is not None

    # ref: derivation of beta
    # https://people.eecs.berkeley.edu/~jordan/papers/liang-jordan-klein-haba.pdf
    num_K = len(self.var_param_beta[idx_a]) - 1
    grad_ln_p_beta = (np.ones(num_K) * (1 - self.param_gem_gamma) /
                      self.var_param_beta[idx_a][-1])
    for k in range(num_K):
      for i in range(k + 1, num_K):
        sum_beta = np.sum(self.var_param_beta[idx_a][:i])
        grad_ln_p_beta[k] += 1 / (1 - sum_beta)

    grad_beta = grad_ln_p_beta + grad_ln_E_p_tx
    grad_beta_norm = np.linalg.norm(grad_beta)
    grad_beta /= grad_beta_norm

    reach = np.zeros(num_K + 1)
    # distance to each canonical hyperplane
    reach[:-1] = -self.var_param_beta[idx_a][:-1] / grad_beta
    # signed distance to hyperplane
    reach[-1] = self.var_param_beta[idx_a][-1] / np.sum(grad_beta)
    max_reach = min(reach[reach > 0])
    search_reach = min(max_reach, grad_beta_norm)

    self.var_param_beta[idx_a][:-1] = (self.var_param_beta[idx_a][:-1] +
                                       lr * search_reach * grad_beta)
    self.var_param_beta[idx_a][-1] = 1 - np.sum(self.var_param_beta[idx_a][:-1])

  def get_prob_tilda_from_lambda(self, np_lambda):

    sum_lambda_pi = np.sum(np_lambda, axis=-1)
    ln_pi = digamma(np_lambda) - digamma(sum_lambda_pi)[..., None]
    return np.exp(ln_pi)

  def do_inference(self):
    # initialize
    self.var_param_beta = []
    self.var_param_bx = []
    self.var_param_pi = []
    for idx_a in range(self.num_agents):
      num_K = self.tuple_num_latents[idx_a] - 1
      tmp_np_v = np.random.beta(1, self.param_gem_gamma, num_K)
      tmp_np_beta = np.zeros(num_K + 1)
      for idx in range(num_K):
        tmp_np_beta[idx] = tmp_np_v[idx]
        for pidx in range(idx):
          tmp_np_beta[idx] *= 1 - tmp_np_v[pidx]
      tmp_np_beta[-1] = 1 - np.sum(tmp_np_beta[:-1])
      self.var_param_beta.append(tmp_np_beta)

      # mid_val = self.param_tx_alpha * tmp_np_beta[0]
      # self.var_param_bx.append(
      #     np.random.uniform(low=max(0.1, mid_val - 0.5),
      #                       high=mid_val + 0.5,
      #                       size=(self.num_ostates, num_K + 1)))
      self.var_param_bx.append(
          np.ones((self.num_ostates, num_K + 1)) *
          (self.param_tx_alpha * tmp_np_beta))
      self.var_param_pi.append(
          np.random.uniform(low=max(0.1, self.param_pi_rho - 0.1),
                            high=self.param_pi_rho + 0.1,
                            size=(num_K + 1, self.num_ostates,
                                  self.tuple_num_actions[idx_a])))
      # tx param
      num_s = self.num_ostates if self.tx_dependency[0] else None
      list_num_a = []
      for i_a in range(self.num_agents):
        if self.tx_dependency[i_a + 1]:
          list_num_a.append(self.tuple_num_actions[i_a])
        else:
          list_num_a.append(None)

      num_sn = self.num_ostates if self.tx_dependency[-1] else None

      self.list_Tx.append(
          TransitionX(num_K + 1, num_s, tuple(list_num_a), num_sn, num_K + 1))
      self.list_Tx[-1].set_lambda_Tx_prior_param(self.param_tx_alpha *
                                                 tmp_np_beta)

    count = 0
    progress_bar = tqdm(total=self.max_iteration)
    while count < self.max_iteration:
      count += 1

      delta_team = 0
      # Don't know which is better to do first between mstep and estep.
      for idx_a in range(self.num_agents):
        prev_lambda_pi = copy.deepcopy(self.var_param_pi[idx_a])
        prev_lambda_tx = copy.deepcopy(self.list_Tx[idx_a].np_lambda_Tx)
        prev_lambda_bx = copy.deepcopy(self.var_param_bx[idx_a])

        self.list_np_policy[idx_a] = self.get_prob_tilda_from_lambda(
            self.var_param_pi[idx_a])
        self.list_Tx[idx_a].conv_to_Tx_tilda()
        self.list_bx[idx_a] = self.get_prob_tilda_from_lambda(
            self.var_param_bx[idx_a])

        list_q_x, list_q_x_xn = self.compute_local_variables(
            idx_a, self.trajectories)  # TODO: use batch

        lr = (count + 1)**(-self.forgetting_rate)
        self.update_global_variables(idx_a, self.trajectories, lr, list_q_x,
                                     list_q_x_xn)

        # compute delta
        delta = np.max(np.abs(self.var_param_pi[idx_a] - prev_lambda_pi))
        delta = max(delta,
                    np.max(np.abs(self.var_param_bx[idx_a] - prev_lambda_bx)))
        delta = max(
            delta,
            np.max(np.abs(self.list_Tx[idx_a].np_lambda_Tx - prev_lambda_tx)))
        delta_team = max(delta, delta_team)

      if delta_team < self.epsilon:
        break
      progress_bar.update()
      progress_bar.set_postfix({'delta': delta_team})
    progress_bar.close()

    for idx in range(self.num_agents):
      numerator = self.var_param_pi[idx]
      action_sums = np.sum(numerator, axis=-1)
      self.list_np_policy[idx] = numerator / action_sums[..., np.newaxis]

    if self.cb_Tx is None:
      for i_a in range(self.num_agents):
        self.list_Tx[i_a].conv_to_Tx()

    if self.cb_bx is None:
      for i_a in range(self.num_agents):
        numerator = self.var_param_bx[i_a]
        latent_sums = np.sum(numerator, axis=-1)
        self.list_bx[i_a] = numerator / latent_sums[..., np.newaxis]
