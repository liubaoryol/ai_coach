from typing import Optional, Tuple, Callable, Sequence
import numpy as np
from tqdm import tqdm
from scipy.special import digamma

T_SAXSeqence = Sequence[Tuple[int, Tuple[int, int], Tuple[int, int]]]

# input: trajectories, mental models (optional), number of agents,
# output: policy table

# components: dirichlet distribution
# pi: |X| x |S| x |A|

A1 = 0
A2 = 1


class TransitionX:
  # TODO: need to find a better way ...
  def __init__(self, num_x, num_s, num_a1, num_a2, num_sn, num_xn):
    assert (num_x > 0)
    assert (num_xn > 0)
    print("Gen TransX")

    self.num_x = num_x
    self.num_s = num_s
    self.num_a1 = num_a1
    self.num_a2 = num_a2
    self.num_sn = num_sn
    self.num_xn = num_xn

    shape = []

    shape.append(self.num_x)
    if self.num_s:
      shape.append(self.num_s)
    if self.num_a1:
      shape.append(self.num_a1)
    if self.num_a2:
      shape.append(self.num_a2)
    if self.num_sn:
      shape.append(self.num_sn)
    shape.append(self.num_xn)

    self.shape = shape
    self.np_lambda_Tx = None
    self.np_Tx = None  # share between Tx_tilda and final Tx

  def get_q_xxn(self, s, a1, a2, sn):
    index = [slice(None)]
    if self.num_s:
      index.append(s)
    if self.num_a1:
      index.append(a1)
    if self.num_a2:
      index.append(a2)
    if self.num_sn:
      index.append(sn)
    index.append(slice(None))

    return self.np_Tx[tuple(index)]

  def init_lambda_Tx(self, beta):
    self.np_lambda_Tx = np.full(self.shape, beta)

  def add_to_lambda_Tx(self, s, a1, a2, sn, q_xxn):
    index = [slice(None)]
    if self.num_s:
      index.append(s)
    if self.num_a1:
      index.append(a1)
    if self.num_a2:
      index.append(a2)
    if self.num_sn:
      index.append(sn)
    index.append(slice(None))
    self.np_lambda_Tx[tuple(index)] += q_xxn

  def conv_to_Tx_tilda(self):
    sum_lambda_Tx = np.sum(self.np_lambda_Tx, axis=-1)
    ln_Txi = digamma(self.np_lambda_Tx) - digamma(sum_lambda_Tx)[..., None]
    self.np_Tx = np.exp(ln_Txi)

  def get_Tx_prop(self, x, s, a1, a2, sn, xn):
    index = [x]
    if self.num_s:
      index.append(s)
    if self.num_a1:
      index.append(a1)
    if self.num_a2:
      index.append(a2)
    if self.num_sn:
      index.append(sn)
    index.append(xn)
    return self.np_Tx[tuple(index)]

  def conv_to_Tx(self):
    numerator = self.np_lambda_Tx - 1
    next_latent_sums = np.sum(numerator, axis=-1)
    self.np_Tx = numerator / next_latent_sums[..., np.newaxis]


class VarInferDuo:
  def __init__(
      self,
      trajectories: Sequence[T_SAXSeqence],
      num_states: int,
      num_latent_states: int,
      tuple_num_actions: Tuple[int, ...],
      cb_transition_s,
      trans_x_dependency=(True, True, True, False),  # s, a1, a2, sn
      max_iteration: int = 100,
      epsilon: float = 0.001) -> None:
    '''
      trajectories: list of list of (state, joint action)-tuples
    '''

    DIRICHLET_PARAM_PI = 3
    self.trajectories = trajectories
    self.beta_pi = DIRICHLET_PARAM_PI
    self.beta_T1 = DIRICHLET_PARAM_PI
    self.beta_T2 = DIRICHLET_PARAM_PI
    self.num_agents = 2
    self.num_ostates = num_states
    self.num_lstates = num_latent_states
    self.tuple_num_actions = tuple_num_actions
    self.cb_transition_s = cb_transition_s
    # num_agent x |X| x |S| x |A|
    self.list_np_policy = [None for dummy_i in range(self.num_agents)
                           ]  # type: list[np.ndarray]
    self.tx_dependency = trans_x_dependency
    self.list_Tx = None  # type: list[TransitionX]
    self.cb_Tx = None
    self.cb_bx = None

    self.max_iteration = max_iteration
    self.epsilon = epsilon
    self.file_name = None

  def set_bx_and_Tx(self, cb_bx, cb_Tx=None):
    self.cb_bx = cb_bx
    self.cb_Tx = cb_Tx

  def set_load_save_file_name(self, file_name):
    self.file_name = file_name

  def get_Tx_pr(self, agent_idx, xidx, sidx, aidx1, aidx2, sidx_n, xidx_n):
    if self.cb_Tx is not None:
      return self.cb_Tx(agent_idx, xidx, sidx, aidx1, aidx2, sidx_n, xidx_n)
    else:
      return self.list_Tx[agent_idx].get_Tx_prop(xidx, sidx, aidx1, aidx2,
                                                 sidx_n, xidx_n)

  def set_dirichlet_prior(self, beta_pi: float, beta_T1: float, beta_T2: float):
    # beta
    self.beta_pi = beta_pi
    self.beta_T1 = beta_T1
    self.beta_T2 = beta_T2

  def estep_local_variables(self, list_policy):

    list_q_x = []  # Ntraj x Nstep x X^2
    list_q_x_xn = []  # Ntraj x Nstep x X^4
    for m_th in range(len(self.trajectories)):
      trajectory = self.trajectories[m_th]

      # Forward messaging
      seq_forward = np.zeros(
          (len(trajectory), self.num_lstates, self.num_lstates))
      # t = 0
      t = 0
      stt_p, joint_a_p, joint_x_p = trajectory[t]
      possible_x1_p = (range(self.num_lstates)
                       if joint_x_p[A1] is None else [joint_x_p[A1]])
      possible_x2_p = (range(self.num_lstates)
                       if joint_x_p[A2] is None else [joint_x_p[A2]])

      for xidx1 in possible_x1_p:
        for xidx2 in possible_x2_p:
          seq_forward[t][xidx1, xidx2] = (
              self.cb_bx(A1, stt_p, xidx1) * self.cb_bx(A2, stt_p, xidx2) *
              # list_bx[A1][xidx1] * list_bx[A2][xidx2] *
              list_policy[A1][xidx1, stt_p, joint_a_p[A1]] *
              list_policy[A2][xidx2, stt_p, joint_a_p[A2]])

      # t = 1:N-1
      for t in range(1, len(trajectory)):
        t_p = t - 1
        stt, joint_a, joint_x = trajectory[t]
        a1_p = joint_a_p[A1]
        a2_p = joint_a_p[A2]
        possible_x1 = (range(self.num_lstates)
                       if joint_x[A1] is None else [joint_x[A1]])
        possible_x2 = (range(self.num_lstates)
                       if joint_x[A2] is None else [joint_x[A2]])
        for xidx1 in possible_x1:
          for xidx2 in possible_x2:
            # prev
            for xidx1_p in possible_x1_p:
              for xidx2_p in possible_x2_p:
                seq_forward[t][xidx1, xidx2] += (
                    seq_forward[t_p][xidx1_p, xidx2_p] *
                    self.get_Tx_pr(A1, xidx1_p, stt_p, a1_p, a2_p, stt, xidx1) *
                    self.get_Tx_pr(A2, xidx2_p, stt_p, a1_p, a2_p, stt, xidx2) *
                    # list_Tx[A1][xidx1_p, stt_p, a1_p, a2_p, xidx1] *
                    # list_Tx[A2][xidx2_p, stt_p, a1_p, a2_p, xidx2] *
                    self.cb_transition_s(stt_p, a1_p, a2_p, stt) *
                    list_policy[A1][xidx1, stt, joint_a[A1]] *
                    list_policy[A2][xidx2, stt, joint_a[A2]])
        stt_p = stt
        joint_a_p = joint_a
        # joint_x_p = joint_x
        possible_x1_p = possible_x1
        possible_x2_p = possible_x2

      # Backward messaging
      seq_backward = np.zeros(
          (len(trajectory), self.num_lstates, self.num_lstates))
      # t = N-1
      t = len(trajectory) - 1

      stt_n, joint_a_n, joint_x_n = trajectory[t]
      possible_x1_n = (range(self.num_lstates)
                       if joint_x_n[A1] is None else [joint_x_n[A1]])
      possible_x2_n = (range(self.num_lstates)
                       if joint_x_n[A2] is None else [joint_x_n[A2]])

      for xidx1 in possible_x1_n:
        for xidx2 in possible_x2_n:
          seq_backward[t][xidx1, xidx2] = 1

      # t = 0:N-2
      for t in reversed(range(0, len(trajectory) - 1)):
        t_n = t + 1
        stt, joint_a, joint_x = trajectory[t]
        # a1_n = joint_a_n[A1]
        # a2_n = joint_a_n[A2]
        a1 = joint_a[A1]
        a2 = joint_a[A2]
        possible_x1 = (range(self.num_lstates)
                       if joint_x[A1] is None else [joint_x[A1]])
        possible_x2 = (range(self.num_lstates)
                       if joint_x[A2] is None else [joint_x[A2]])
        for xidx1 in possible_x1:
          for xidx2 in possible_x2:
            # prev
            for xidx1_n in possible_x1_n:
              for xidx2_n in possible_x2_n:
                seq_backward[t][xidx1, xidx2] += (
                    seq_backward[t_n][xidx1_n, xidx2_n] *
                    self.get_Tx_pr(A1, xidx1, stt, a1, a2, stt_n, xidx1_n) *
                    self.get_Tx_pr(A2, xidx2, stt, a1, a2, stt_n, xidx2_n) *
                    # list_Tx[A1][xidx1, stt, a1, a2, xidx1_n] *
                    # list_Tx[A2][xidx2, stt, a1, a2, xidx2_n] *
                    self.cb_transition_s(stt, a1, a2, stt_n) *
                    list_policy[A1][xidx1_n, stt_n, joint_a_n[A1]] *
                    list_policy[A2][xidx2_n, stt_n, joint_a_n[A2]])
        stt_n = stt
        joint_a_n = joint_a
        # joint_x_n = joint_x
        possible_x1_n = possible_x1
        possible_x2_n = possible_x2

      # compute q_x, q_x_xp
      # z_partian = np.sum(seq_forward, axis=(1, 2))
      q_joint_x = seq_forward * seq_backward
      q_x1 = np.sum(q_joint_x, axis=2)
      q_x2 = np.sum(q_joint_x, axis=1)
      q_x1 = q_x1 / np.sum(q_x1, axis=1)[:, None]
      q_x2 = q_x2 / np.sum(q_x2, axis=1)[:, None]
      list_q_x.append([q_x1, q_x2])

      if self.cb_Tx is None:
        q_xx_xnxn = np.zeros(
            (len(trajectory) - 1, self.num_lstates, self.num_lstates,
             self.num_lstates, self.num_lstates))
        for t in range(len(trajectory) - 1):
          stt, joint_a, joint_x = trajectory[t]
          sttn, joint_a_n, joint_x_n = trajectory[t + 1]
          a1 = joint_a[A1]
          a2 = joint_a[A2]
          q_xx_xnxn[t] = (
              seq_forward[t, :, :, None, None] *
              seq_backward[t + 1, None, None, :, :] *
              self.list_Tx[A1].get_q_xxn(stt, a1, a2, sttn)[:, None, :, None] *
              self.list_Tx[A2].get_q_xxn(stt, a1, a2, sttn)[None, :, None, :] *
              # list_Tx[A1][:, stt, a1, a2, :][:, None, :, None] *
              # list_Tx[A2][:, stt, a1, a2, :][None, :, None, :] *
              self.cb_transition_s(stt, a1, a2, sttn) *
              list_policy[A1][:, sttn, joint_a_n[A1]][None, None, :, None] *
              list_policy[A2][:, sttn, joint_a_n[A2]][None, None, None, :])

        q_x_xn1 = np.sum(q_xx_xnxn, axis=(2, 4))
        q_x_xn1 = q_x_xn1 / np.sum(q_x_xn1, axis=(1, 2))[:, None, None]
        q_x_xn2 = np.sum(q_xx_xnxn, axis=(1, 3))
        q_x_xn2 = q_x_xn2 / np.sum(q_x_xn2, axis=(1, 2))[:, None, None]
        list_q_x_xn.append([q_x_xn1, q_x_xn2])

    return list_q_x, list_q_x_xn

  def mstep_global_variables(self, list_q_x: Sequence[Sequence[np.ndarray]],
                             list_q_x_xn: Sequence[Sequence[np.ndarray]]):

    list_lambda_pi = []
    # policy
    for idx in range(self.num_agents):
      lambda_pi = np.full(
          (self.num_lstates, self.num_ostates, self.tuple_num_actions[idx]),
          self.beta_pi)
      list_lambda_pi.append(lambda_pi)

    for m_th in range(len(self.trajectories)):
      q_x1, q_x2 = list_q_x[m_th]
      traj = self.trajectories[m_th]
      for t, (state, joint_a, joint_x) in enumerate(traj):
        list_lambda_pi[A1][:, state, joint_a[A1]] += q_x1[t, :]
        list_lambda_pi[A2][:, state, joint_a[A2]] += q_x2[t, :]

    # transition_x
    if self.cb_Tx is None and len(list_q_x_xn) > 0:
      self.list_Tx[A1].init_lambda_Tx(self.beta_T1)
      self.list_Tx[A2].init_lambda_Tx(self.beta_T2)
      # list_lambda_Tx = []
      # lambda_Ti_1 = np.full(
      #     (self.num_lstates, self.num_ostates, self.tuple_num_actions[A1],
      #      self.tuple_num_actions[A2], self.num_lstates), self.beta_T1)
      # lambda_Ti_2 = np.full(
      #     (self.num_lstates, self.num_ostates, self.tuple_num_actions[A1],
      #      self.tuple_num_actions[A2], self.num_lstates), self.beta_T2)
      # list_lambda_Tx.append(lambda_Ti_1)
      # list_lambda_Tx.append(lambda_Ti_2)

      for m_th in range(len(self.trajectories)):
        q_x_xn1, q_x_xn2 = list_q_x_xn[m_th]
        traj = self.trajectories[m_th]
        # for t, state, joint_a, joint_x in enumerate(traj):
        for t in range(len(traj) - 1):
          state, joint_a, _ = traj[t]
          state_n, _, _ = traj[t + 1]

          a1 = joint_a[A1]
          a2 = joint_a[A2]
          self.list_Tx[A1].add_to_lambda_Tx(state, a1, a2, state_n,
                                            q_x_xn1[t, :, :])
          self.list_Tx[A2].add_to_lambda_Tx(state, a1, a2, state_n,
                                            q_x_xn2[t, :, :])
          # list_lambda_Tx[A1][:, state, a1, a2, :] += q_x_xn1[t, :, :]
          # list_lambda_Tx[A2][:, state, a1, a2, :] += q_x_xn2[t, :, :]

    return list_lambda_pi

  def get_pi_tilda_from_lambda(self, list_lambda_pi):
    list_pi_tilda = []

    for idx in range(self.num_agents):
      sum_lambda_pi = np.sum(list_lambda_pi[idx], axis=-1)
      ln_pi = digamma(list_lambda_pi[idx]) - digamma(sum_lambda_pi)[:, :, None]
      list_pi_tilda.append(np.exp(ln_pi))

    return list_pi_tilda

  def do_inference(self,
                   callback: Optional[Callable[[int, Sequence[np.ndarray]],
                                               None]] = None):
    list_q_x = []
    for m_th in range(len(self.trajectories)):
      traj = self.trajectories[m_th]
      np_q_x1 = np.zeros((len(traj), self.num_lstates))
      np_q_x2 = np.zeros((len(traj), self.num_lstates))
      for t in range(len(traj)):
        _, _, joint_x = traj[t]

        if joint_x[A1] is not None:
          np_q_x1[t, joint_x[A1]] = 1
        else:
          np_q_x1[t, :] = 1 / self.num_lstates

        if joint_x[A2] is not None:
          np_q_x2[t, joint_x[A2]] = 1
        else:
          np_q_x2[t, :] = 1 / self.num_lstates

      list_q_x.append([np_q_x1, np_q_x2])

    list_q_x_xn = []
    if self.cb_Tx is None:
      for m_th in range(len(self.trajectories)):
        traj = self.trajectories[m_th]
        np_q_x_xn1 = np.zeros((len(traj), self.num_lstates, self.num_lstates))
        np_q_x_xn2 = np.zeros((len(traj), self.num_lstates, self.num_lstates))
        for t in range(len(traj) - 1):
          _, _, joint_x = traj[t]
          _, _, joint_x_n = traj[t + 1]

          if joint_x[A1] is not None:
            if joint_x_n[A1] is not None:
              np_q_x_xn1[t, joint_x[A1], joint_x_n[A1]] = 1
            else:
              np_q_x_xn1[t, joint_x[A1], :] = 1 / self.num_lstates
          else:
            if joint_x_n[A1] is not None:
              np_q_x_xn1[t, :, joint_x_n[A1]] = 1 / self.num_lstates
            else:
              np_q_x_xn1[t, :, :] = 1 / (self.num_lstates * self.num_lstates)

          if joint_x[A2] is not None:
            if joint_x_n[A2] is not None:
              np_q_x_xn2[t, joint_x[A2], joint_x_n[A2]] = 1
            else:
              np_q_x_xn2[t, joint_x[A2], :] = 1 / self.num_lstates
          else:
            if joint_x_n[A2] is not None:
              np_q_x_xn2[t, :, joint_x_n[A2]] = 1 / self.num_lstates
            else:
              np_q_x_xn2[t, :, :] = 1 / (self.num_lstates * self.num_lstates)

        list_q_x_xn.append([np_q_x_xn1, np_q_x_xn2])

    if self.cb_Tx is None:
      num_s = self.num_ostates if self.tx_dependency[0] else None
      num_a1 = self.tuple_num_actions[A1] if self.tx_dependency[1] else None
      num_a2 = self.tuple_num_actions[A2] if self.tx_dependency[2] else None
      num_sn = self.num_ostates if self.tx_dependency[3] else None
      self.list_Tx = []
      self.list_Tx.append(
          TransitionX(self.num_lstates, num_s, num_a1, num_a2, num_sn,
                      self.num_lstates))
      self.list_Tx.append(
          TransitionX(self.num_lstates, num_s, num_a1, num_a2, num_sn,
                      self.num_lstates))
      self.list_Tx[A1].init_lambda_Tx(self.beta_T1)
      self.list_Tx[A2].init_lambda_Tx(self.beta_T2)

    list_lambda_pi = []
    if self.file_name is not None:
      try:
        with np.load(self.file_name) as data:
          list_lambda_pi.append(data['arr_0'])
          list_lambda_pi.append(data['arr_1'])
          print("lambda_pi loaded from disk")
      except IOError:
        pass

    if len(list_lambda_pi) == 0:
      print("initialize lambda_pi")
      list_lambda_pi = [
          np.full(
              (self.num_lstates, self.num_ostates, self.tuple_num_actions[i_a]),
              self.beta_pi) for i_a in range(self.num_agents)
      ]

    list_lambda_pi_prev = None

    count = 0
    progress_bar = tqdm(total=self.max_iteration)
    while count < self.max_iteration:
      count += 1
      list_lambda_pi_prev = list_lambda_pi

      list_pi_tilda = self.get_pi_tilda_from_lambda(list_lambda_pi)
      if self.list_Tx is not None:
        self.list_Tx[A1].conv_to_Tx_tilda()
        self.list_Tx[A2].conv_to_Tx_tilda()

      # Don't know which is better to do between mstep and estp.
      list_q_x, list_q_x_xn = self.estep_local_variables(list_pi_tilda)
      list_lambda_pi = self.mstep_global_variables(list_q_x, list_q_x_xn)

      if self.file_name is not None and count % 1 == 0:
        np.savez(self.file_name, list_lambda_pi[A1], list_lambda_pi[A2])

      # if callback:
      #   callback(self.num_agents, list_q_pi_hyper)

      delta_team = 0
      for i_a in range(self.num_agents):
        delta = np.max(np.abs(list_lambda_pi[i_a] - list_lambda_pi_prev[i_a]))
        delta_team = max(delta_team, delta)

      if delta_team < self.epsilon:
        break
      progress_bar.update()
      progress_bar.set_postfix({'delta': delta_team})
    progress_bar.close()

    for idx in range(self.num_agents):
      numerator = list_lambda_pi[idx] - 1
      action_sums = np.sum(numerator, axis=-1)
      self.list_np_policy[idx] = numerator / action_sums[:, :, np.newaxis]

    if self.list_Tx is not None:
      self.list_Tx[A1].conv_to_Tx()
      self.list_Tx[A2].conv_to_Tx()


if __name__ == "__main__":
  pass
