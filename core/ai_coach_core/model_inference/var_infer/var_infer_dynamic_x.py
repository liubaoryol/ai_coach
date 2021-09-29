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


class VarInferDuo:
  def __init__(self,
               trajectories: Sequence[T_SAXSeqence],
               num_states: int,
               num_latent_states: int,
               tuple_num_actions: Tuple[int, ...],
               cb_transition_s,
               max_iteration: int = 1000,
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

    self.max_iteration = max_iteration
    self.epsilon = epsilon

    # num_agent x |X| x |S| x |A|
    self.list_base_q_pi_hyper = []  # type: list[np.ndarray]

    # num_agent x |X| x |S| x |A| x |A| x |X|   or
    # num_agent x |X| x |A| x |A| x |X|
    self.list_base_q_Ti_hyper = []  # type: list[np.ndarray]

    self.list_mental_models = [
        None for dummy_i in range(len(self.trajectories))
    ]  # type: list[tuple[int, ...]]

  def set_dirichlet_prior(self, beta_pi: float, beta_T1: float, beta_T2: float):
    # beta
    self.beta_pi = beta_pi
    self.beta_T1 = beta_T1
    self.beta_T2 = beta_T2

  def estep_local_variables(self, list_policy, list_Ti, list_bx0):

    list_q_x = []  # Ntraj x Nstep x X^2
    list_q_x_xn = []  # Ntraj x Nstep x X^4
    for m_th in range(len(self.trajectories)):
      trajectory = self.trajectories[m_th]

      # Forward messaging
      seq_forward = np.zeros(
          (len(trajectory), self.num_lstates, self.num_lstates))
      # t = 0
      t = 0
      state_p, joint_a_p, joint_x_p = trajectory[t]
      possible_x1_p = (range(self.num_lstates)
                       if joint_x_p[A1] is None else [joint_x_p[A1]])
      possible_x2_p = (range(self.num_lstates)
                       if joint_x_p[A2] is None else [joint_x_p[A2]])

      for xidx1 in possible_x1_p:
        for xidx2 in possible_x2_p:
          seq_forward[t][xidx1, xidx2] = (
              list_bx0[A1][xidx1] * list_bx0[A2][xidx2] *
              list_policy[A1][xidx1, state_p, joint_a_p[A1]] *
              list_policy[A2][xidx2, state_p, joint_a_p[A2]])

      # t = 1:N-1
      for t in range(1, len(trajectory)):
        t_p = t - 1
        state, joint_a, joint_x = trajectory[t]
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
                    list_Ti[A1][xidx1_p, state_p, a1_p, a2_p, xidx1] *
                    list_Ti[A2][xidx2_p, state_p, a1_p, a2_p, xidx2] *
                    self.cb_transition_s(state_p, a1_p, a2_p, state) *
                    list_policy[A1][xidx1, state, joint_a[A1]] *
                    list_policy[A2][xidx2, state, joint_a[A2]])
        state_p = state
        joint_a_p = joint_a
        joint_x_p = joint_x
        possible_x1_p = possible_x1
        possible_x2_p = possible_x2

      # Backward messaging
      seq_backward = np.zeros(
          (len(trajectory), self.num_lstates, self.num_lstates))
      # t = N-1
      t = len(trajectory) - 1

      state_n, joint_a_n, joint_x_n = trajectory[t]
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
        state, joint_a, joint_x = trajectory[t]
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
                    list_Ti[A1][xidx1, state, a1, a2, xidx1_n] *
                    list_Ti[A2][xidx2, state, a1, a2, xidx2_n] *
                    self.cb_transition_s(state, a1, a2, state_n) *
                    list_policy[A1][xidx1_n, state_n, joint_a_n[A1]] *
                    list_policy[A2][xidx2_n, state_n, joint_a_n[A2]])
        state_n = state
        joint_a_n = joint_a
        joint_x_n = joint_x
        possible_x1_n = possible_x1
        possible_x2_n = possible_x2

      # compute q_x, q_x_xp
      # z_partian = np.sum(seq_forward, axis=(1, 2))
      q_joint_x = seq_forward * seq_backward
      q_x1 = np.sum(q_joint_x, axis=1)
      q_x2 = np.sum(q_joint_x, axis=2)
      q_x1 = q_x1 / np.sum(q_x1, axis=1)[:, None]
      q_x2 = q_x2 / np.sum(q_x2, axis=1)[:, None]
      list_q_x.append([q_x1, q_x2])

      q_xx_xnxn = np.zeros(
          (len(trajectory) - 1, self.num_lstates, self.num_lstates,
           self.num_lstates, self.num_lstates))
      for t in range(len(trajectory) - 1):
        state, joint_a, joint_x = trajectory[t]
        state_n, joint_a_n, joint_x_n = trajectory[t + 1]
        a1 = joint_a[A1]
        a2 = joint_a[A2]
        q_xx_xnxn[t] = (
            seq_forward[t, :, :, None, None] *
            seq_backward[t + 1, None, None, :, :] *
            list_Ti[A1][:, state, a1, a2, :][:, None, :, None] *
            list_Ti[A2][:, state, a1, a2, :][None, :, None, :] *
            self.cb_transition_s(state, a1, a2, state_n) *
            list_policy[A1][:, state_n, joint_a_n[A1]][None, None, :, None] *
            list_policy[A2][:, state_n, joint_a_n[A2]][None, None, None, :])

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

        # if joint_x[A1] is None:
        #   list_u_pi[A1][:, state, joint_a[A1]] += q_x1[t, :]
        # else:
        #   list_u_pi[A1][joint_x[A1], state, joint_a[A1]] += 1

        # if joint_x[A2] is None:
        #   list_u_pi[A2][:, state, joint_a[A2]] += q_x2[t, :]
        # else:
        #   list_u_pi[A2][joint_x[A2], state, joint_a[A2]] += 1

    # transition_x
    list_lambda_Txi = []
    lambda_Ti_1 = np.full(
        (self.num_lstates, self.num_ostates, self.tuple_num_actions[A1],
         self.tuple_num_actions[A2], self.num_lstates), self.beta_T1)
    lambda_Ti_2 = np.full(
        (self.num_lstates, self.num_ostates, self.tuple_num_actions[A1],
         self.tuple_num_actions[A2], self.num_lstates), self.beta_T2)
    list_lambda_Txi.append(lambda_Ti_1)
    list_lambda_Txi.append(lambda_Ti_2)

    for m_th in range(len(self.trajectories)):
      q_x_xn1, q_x_xn2 = list_q_x_xn[m_th]
      traj = self.trajectories[m_th]
      # for t, state, joint_a, joint_x in enumerate(traj):
      for t in range(len(traj) - 1):
        state, joint_a, joint_x = traj[t]

        a1 = joint_a[A1]
        a2 = joint_a[A2]
        list_lambda_Txi[A1][:, state, a1, a2, :] += q_x_xn1[t, :, :]
        list_lambda_Txi[A2][:, state, a1, a2, :] += q_x_xn2[t, :, :]

    return list_lambda_pi, list_lambda_Txi

  def get_probability_from_lambda(self, list_lambda_pi, list_lambda_Txi):
    list_policy = []
    list_Ti = []
    for idx in range(self.num_agents):
      sum_lambda_pi = np.sum(list_lambda_pi[idx], axis=2)
      ln_pi = digamma(list_lambda_pi[idx]) - digamma(sum_lambda_pi)[:, :, None]
      list_policy.append(np.exp(ln_pi))

      sum_lambda_Txi = np.sum(list_lambda_Txi[idx], axis=2)
      ln_Txi = digamma(list_lambda_Txi[idx]) - digamma(sum_lambda_Txi)[:, :,
                                                                       None]
      list_Ti.append(np.exp(ln_Txi))

    return list_policy, list_Ti

  def do_inference(self,
                   callback: Optional[Callable[[int, Sequence[np.ndarray]],
                                               None]] = None):
    list_q_x = []
    for m_th in range(len(self.trajectories)):
      traj = self.trajectories[m_th]
      np_q_x1 = np.full((len(traj), self.num_lstates), 1 / self.num_lstates)
      np_q_x2 = np.full((len(traj), self.num_lstates), 1 / self.num_lstates)

      list_q_x.append([np_q_x1, np_q_x2])

    list_q_x_xn = []
    for m_th in range(len(self.trajectories)):
      traj = self.trajectories[m_th]
      np_q_x_xn1 = np.full((len(traj), self.num_lstates, self.num_lstates),
                           1 / (self.num_lstates * self.num_lstates))
      np_q_x_xn2 = np.full((len(traj), self.num_lstates, self.num_lstates),
                           1 / (self.num_lstates * self.num_lstates))

      list_q_x_xn.append([np_q_x_xn1, np_q_x_xn2])

    list_lambda_pi = [
        np.full(
            (self.num_lstates, self.num_ostates, self.tuple_num_actions[i_a]),
            self.beta_pi) for i_a in range(self.num_agents)
    ]
    list_lambda_pi_prev = None

    list_b0 = [
        np.full((self.num_lstates, ), 1 / self.num_lstates)
        for i_a in range(self.num_agents)
    ]

    count = 0
    progress_bar = tqdm(total=self.max_iteration)
    while count < self.max_iteration:
      count += 1
      list_lambda_pi_prev = list_lambda_pi
      # Don't know which is better to do between mstep and estp.
      list_lambda_pi, list_lambda_Txi = self.mstep_global_variables(
          list_q_x, list_q_x_xn)

      list_policy, list_Ti = self.get_probability_from_lambda(
          list_lambda_pi, list_lambda_Txi)

      list_q_x, list_q_x_xn = self.estep_local_variables(
          list_policy, list_Ti, list_b0)

      # if callback:
      #   callback(self.num_agents, list_q_pi_hyper)

      delta_team = 0
      for i_a in range(self.num_agents):
        delta = np.max(np.abs(list_lambda_pi[i_a] - list_lambda_pi_prev[i_a]))
        delta_team = max(delta_team, delta)
      if delta_team < self.epsilon:
        break

      progress_bar.update()
    progress_bar.close()
    self.list_np_policy = list_policy


if __name__ == "__main__":
  pass
