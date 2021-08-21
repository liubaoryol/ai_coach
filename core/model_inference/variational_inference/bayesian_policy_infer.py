from typing import Optional, Tuple, Callable, Sequence
import copy
import numpy as np
from tqdm import tqdm
from scipy.special import digamma

T_StateJointActionSeqence = Sequence[Tuple[int, Tuple[int, ...]]]

# input: trajectories, mental models (optional), number of agents,
# output: policy table

# components: dirichlet distribution
# pi: |S| x |X| x |A|


class bayesian_policy_learning:
  def __init__(self,
               unlabeled_data: Sequence[T_StateJointActionSeqence],
               labeled_data: Sequence[T_StateJointActionSeqence],
               labels: Sequence[Tuple[int, ...]],
               num_agents: int,
               num_states: int,
               num_latent_states: int,
               tuple_num_actions: Tuple[int, ...],
               max_iteration: int = 10000,
               epsilon: float = 0.001) -> None:
    '''
      trajectories: list of list of (state, joint action)-tuples
    '''
    assert len(labeled_data) == len(labels)

    DIRICHLET_PARAM_PI = 10
    self.labeled_data = labeled_data
    self.unlabeled_data = unlabeled_data
    self.labels = labels
    self.prior_hyperparam = DIRICHLET_PARAM_PI
    self.num_agents = num_agents
    self.num_ostates = num_states
    self.num_lstates = num_latent_states
    self.tuple_num_actions = tuple_num_actions
    self.np_policy = [None for dummy_i in range(num_agents)
                      ]  # type: list[np.ndarray]

    self.max_iteration = max_iteration
    self.epsilon = epsilon

  def set_dirichlet_prior(self, hyperparam: float):
    # added 1 just to make sure modes could be found
    # self.prior_hyperparam = (
    #     hyperparam / (self.num_actions) + 1)
    self.prior_hyperparam = hyperparam

  def do_inference(self, callback: Optional[Callable] = None):
    pass


class supervised_bayesian_policy_learning(bayesian_policy_learning):
  def __init__(
      self,
      trajectories: Sequence[T_StateJointActionSeqence],
      latent_labels: Sequence[Tuple[int, ...]],
      num_agents: int,
      num_states: int,
      num_latent_states: int,
      tuple_num_actions: Tuple[int, ...],
  ) -> None:
    '''
      trajectories: list of list of (state, joint action)-tuples
    '''
    super().__init__(None, trajectories, latent_labels, num_agents, num_states,
                     num_latent_states, tuple_num_actions)

  def do_inference(self, callback=None):
    for idx in range(self.num_agents):
      self.np_policy[idx] = np.full(
          (self.num_ostates, self.num_lstates, self.tuple_num_actions[idx]),
          self.prior_hyperparam - 1)

    for i_data in range(len(self.labeled_data)):
      joint_lstate = self.labels[i_data]
      for state, joint_action in self.labeled_data[i_data]:
        if self.num_agents == 1:
          ind_ls = joint_lstate
          ind_act = joint_action
          self.np_policy[0][state][ind_ls][ind_act] += 1
        else:
          for i_a in range(self.num_agents):
            ind_ls = joint_lstate[i_a]
            ind_act = joint_action[i_a]
            self.np_policy[i_a][state][ind_ls][ind_act] += 1

    for idx in range(self.num_agents):
      action_sums = np.sum(self.np_policy[idx], axis=2)
      self.np_policy[idx] = (self.np_policy[idx] /
                             action_sums[:, :, np.newaxis])


class semisupervised_bayesian_policy_learning(bayesian_policy_learning):
  def __init__(self,
               unlabeled_data: Sequence[T_StateJointActionSeqence],
               labeled_data: Sequence[T_StateJointActionSeqence],
               labels: Sequence[Tuple[int, ...]],
               num_agents: int,
               num_states: int,
               num_latent_states: int,
               tuple_num_actions: Tuple[int, ...],
               iteration: int = 10000,
               epsilon: float = 0.001) -> None:
    '''
        trajectories: list of list of (state, joint action)-tuples
        '''
    super().__init__(unlabeled_data, labeled_data, labels, num_agents,
                     num_states, num_latent_states, tuple_num_actions,
                     iteration, epsilon)
    # self.cal_base_pi_hyper()

    # for debug purpose
    self.list_mental_models = [
        None for dummy_i in range(len(self.unlabeled_data))
    ]  # type: list[tuple[int, ...]]

  def estep_local_variables(self, list_q_pi_hyper: Sequence[np.ndarray]):
    if len(self.unlabeled_data) == 0:
      return []

    avg_ln_pi = []  # type: list[np.ndarray]
    for idx in range(self.num_agents):
      sum_hyper = np.sum(list_q_pi_hyper[idx], axis=2)
      tmp_avg_ln_pi = (digamma(list_q_pi_hyper[idx]) -
                       digamma(sum_hyper)[:, :, np.newaxis])
      avg_ln_pi.append(tmp_avg_ln_pi)

    list_q_x = []  # type: list[np.ndarray]
    for traj in self.unlabeled_data:
      tmp_np_ln_q_x = np.zeros((self.num_agents, self.num_lstates))
      for state, joint_action in traj:
        for i_x in range(self.num_lstates):
          if self.num_agents == 1:
            tmp_np_ln_q_x[0][i_x] += (avg_ln_pi[0][state][i_x][joint_action])
          else:
            for i_a in range(self.num_agents):
              tmp_np_ln_q_x[i_a][i_x] += (
                  avg_ln_pi[i_a][state][i_x][joint_action[i_a]])
      tmp_np_ln_q_x = tmp_np_ln_q_x - np.max(tmp_np_ln_q_x, axis=1)[:, None]
      tmp_np_q_x = np.exp(tmp_np_ln_q_x)
      sum_np_q_x = np.sum(tmp_np_q_x, axis=1)
      np_q_x = tmp_np_q_x / sum_np_q_x[:, np.newaxis]
      list_q_x.append(np_q_x)

    return list_q_x

  def set_dirichlet_prior(self, hyperparam: float):
    self.prior_hyperparam = hyperparam
    self.cal_base_pi_hyper()

  def cal_base_pi_hyper(self):
    self.list_base_q_pi_hyper = []  # type: list[np.ndarray]
    for idx in range(self.num_agents):
      q_pi_hyper = np.full(
          (self.num_ostates, self.num_lstates, self.tuple_num_actions[idx]),
          self.prior_hyperparam,
          dtype=np.float64)
      self.list_base_q_pi_hyper.append(q_pi_hyper)

    # labeled data
    for i_data in range(len(self.labeled_data)):
      joint_lstate = self.labels[i_data]
      for state, joint_action in self.labeled_data[i_data]:
        if self.num_agents == 1:
          # ind_ls = joint_lstate
          # ind_act = joint_action
          self.list_base_q_pi_hyper[0][state][joint_lstate][joint_action] += 1
        else:
          for i_a in range(self.num_agents):
            ind_ls = joint_lstate[i_a]
            ind_act = joint_action[i_a]
            self.list_base_q_pi_hyper[i_a][state][ind_ls][ind_act] += 1

  def mstep_global_variables(self, list_q_x: Sequence[np.ndarray]):
    list_q_pi_hyper = copy.deepcopy(self.list_base_q_pi_hyper)

    # unlabeled data
    for idx in range(len(self.unlabeled_data)):
      q_x = list_q_x[idx]
      traj = self.unlabeled_data[idx]
      for state, joint_action in traj:
        for lstate in range(self.num_lstates):
          if self.num_agents == 1:
            list_q_pi_hyper[0][state][lstate][joint_action] += (q_x[0][lstate])
          else:
            for i_a in range(self.num_agents):
              ind_act = joint_action[i_a]
              list_q_pi_hyper[i_a][state][lstate][ind_act] += (q_x[i_a][lstate])

    return list_q_pi_hyper

  def do_inference(self,
                   callback: Optional[Callable[[int, Sequence[np.ndarray]],
                                               None]] = None):
    count = 0
    list_q_x = []
    for dummy_idx in range(len(self.unlabeled_data)):
      np_q_x = np.full((self.num_agents, self.num_lstates),
                       1 / self.num_lstates)
      list_q_x.append(np_q_x)

    list_q_pi_hyper = [
        np.zeros(
            (self.num_ostates, self.num_lstates, self.tuple_num_actions[i_a]))
        for i_a in range(self.num_agents)
    ]
    list_q_pi_hyper_prev = None

    progress_bar = tqdm(total=self.max_iteration)
    while count < self.max_iteration:
      count += 1
      list_q_pi_hyper_prev = list_q_pi_hyper
      # Don't know exactly why,
      # but the performance is better when we do mstep first
      list_q_pi_hyper = self.mstep_global_variables(list_q_x)
      list_q_x = self.estep_local_variables(list_q_pi_hyper)

      if callback:
        callback(self.num_agents, list_q_pi_hyper)
      delta_team = 0
      for i_a in range(self.num_agents):
        delta = np.max(np.abs(list_q_pi_hyper[i_a] - list_q_pi_hyper_prev[i_a]))
        delta_team = max(delta_team, delta)
      if delta_team < self.epsilon:
        break

      progress_bar.update()
    progress_bar.close()

    for idx in range(self.num_agents):
      numerator = list_q_pi_hyper[idx] - 1
      action_sums = np.sum(numerator, axis=2)
      self.np_policy[idx] = numerator / action_sums[:, :, np.newaxis]

    for idx in range(len(list_q_x)):
      self.list_mental_models.append(tuple(np.amax(list_q_x[idx], axis=1)))