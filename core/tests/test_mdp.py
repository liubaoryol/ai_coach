from enum import Enum
import numpy as np
import models.mdp as mdp_lib
import RL.planning as plan_lib
from utils.mdp_utils import StateSpace, ActionSpace
import utils.test_utils as test_utils


class FL_Actions(Enum):
  LEFT = 0
  DOWN = 1
  RIGHT = 2
  UP = 3


class FrozenLakeMDP(mdp_lib.MDP):
  """
      SFFF
      FHFH
      FFFH
      HFFG
  S : starting point, safe
  F : frozen surface, safe
  H : hole, fall to your doom
  G : goal, where the frisbee is located
  The episode ends when you reach the goal or fall in a hole.
  You receive a reward of 1 if you reach the goal, and zero otherwise.
  """
  def __init__(self, use_sparse=False):
    self.width = 4
    self.height = 4
    self.holes = [(1, 1), (3, 1), (3, 2), (0, 3)]
    self.goal = (3, 3)
    super().__init__(use_sparse=use_sparse)

  def init_statespace(self):
    '''
    To disable dummy states, set self.dummy_states = None
    '''

    self.dict_factored_statespace = {}

    set_state = set()
    for i in range(self.width):
      for j in range(self.height):
        state = (i, j)
        set_state.add(state)

    self.s_space = StateSpace(statespace=set_state)
    self.dict_factored_statespace = {0: self.s_space}
    self.dummy_states = StateSpace(statespace=["terminal"])

  def init_actionspace(self):

    self.dict_factored_actionspace = {}
    self.a_space = ActionSpace(actionspace=FL_Actions)
    self.dict_factored_actionspace = {0: self.a_space}

  def is_terminal(self, state_idx):
    return self.is_dummy_state(state_idx)

  def legal_actions(self, state_idx):
    if self.is_terminal(state_idx):
      return []

    sid, = self.conv_idx_to_state(state_idx)
    state = self.s_space.idx_to_state[sid]

    def action_to_idx(act):
      return self.conv_action_to_idx(int(self.a_space.action_to_idx[act]))

    if state in self.holes:
      return [0]
    if state == self.goal:
      return [0]

    legal_act = []
    if state[0] != 0:
      legal_act.append(action_to_idx(FL_Actions.LEFT))
    if state[0] != self.width - 1:
      legal_act.append(action_to_idx(FL_Actions.RIGHT))
    if state[1] != 0:
      legal_act.append(action_to_idx(FL_Actions.UP))
    if state[1] != self.height - 1:
      legal_act.append(action_to_idx(FL_Actions.DOWN))

    return legal_act

  def transition_model(self, state_idx: int, action_idx: int) -> np.ndarray:
    if self.is_terminal(state_idx):
      return np.array([[1.0, state_idx]])

    # unpack the input
    s_i, = self.conv_idx_to_state(state_idx)
    stt = self.s_space.idx_to_state[s_i]

    a_i, = self.conv_idx_to_action(action_idx)
    act = self.a_space.idx_to_action[a_i]

    terminal_sid = self.conv_dummy_state_to_idx("terminal")

    # obtain next probability values for each factor
    if stt in self.holes:
      return np.array([[1.0, terminal_sid]])
    if stt == self.goal:
      return np.array([[1.0, terminal_sid]])

    def state_to_idx(state):
      return self.conv_state_to_idx(self.s_space.state_to_idx[state])

    list_next_p_state = []
    p_remain = 1
    p_move_correct = 2. / 3.
    p_move_side = 1. / 6.
    if act == FL_Actions.LEFT:
      if stt[0] != 0:
        sttn = (stt[0] - 1, stt[1])
        p_remain -= p_move_correct
        list_next_p_state.append((p_move_correct, state_to_idx(sttn)))
      if stt[1] != self.height - 1:
        sttn = (stt[0], stt[1] + 1)
        p_remain -= p_move_side
        list_next_p_state.append((p_move_side, state_to_idx(sttn)))
      if stt[1] != 0:
        sttn = (stt[0], stt[1] - 1)
        p_remain -= p_move_side
        list_next_p_state.append((p_move_side, state_to_idx(sttn)))
    elif act == FL_Actions.RIGHT:
      if stt[0] != self.width - 1:
        sttn = (stt[0] + 1, stt[1])
        p_remain -= p_move_correct
        list_next_p_state.append((p_move_correct, state_to_idx(sttn)))
      if stt[1] != self.height - 1:
        sttn = (stt[0], stt[1] + 1)
        p_remain -= p_move_side
        list_next_p_state.append((p_move_side, state_to_idx(sttn)))
      if stt[1] != 0:
        sttn = (stt[0], stt[1] - 1)
        p_remain -= p_move_side
        list_next_p_state.append((p_move_side, state_to_idx(sttn)))
    elif act == FL_Actions.UP:
      if stt[1] != 0:
        sttn = (stt[0], stt[1] - 1)
        p_remain -= p_move_correct
        list_next_p_state.append((p_move_correct, state_to_idx(sttn)))
      if stt[0] != 0:
        sttn = (stt[0] - 1, stt[1])
        p_remain -= p_move_side
        list_next_p_state.append((p_move_side, state_to_idx(sttn)))
      if stt[0] != self.width - 1:
        sttn = (stt[0] + 1, stt[1])
        p_remain -= p_move_side
        list_next_p_state.append((p_move_side, state_to_idx(sttn)))
    elif act == FL_Actions.DOWN:
      if stt[1] != self.height - 1:
        sttn = (stt[0], stt[1] + 1)
        p_remain -= p_move_correct
        list_next_p_state.append((p_move_correct, state_to_idx(sttn)))
      if stt[0] != 0:
        sttn = (stt[0] - 1, stt[1])
        p_remain -= p_move_side
        list_next_p_state.append((p_move_side, state_to_idx(sttn)))
      if stt[0] != self.width - 1:
        sttn = (stt[0] + 1, stt[1])
        p_remain -= p_move_side
        list_next_p_state.append((p_move_side, state_to_idx(sttn)))

    if len(list_next_p_state) != 3:
      list_next_p_state.append((p_remain, state_idx))

    return np.array(list_next_p_state)

  def reward(self, state_idx: int, action_idx: int, *args, **kwargs) -> float:
    if self.is_terminal(state_idx):
      return 0

    s_i, = self.conv_idx_to_state(state_idx)
    state = self.s_space.idx_to_state[s_i]
    if state == self.goal:
      return 1
    # elif state in self.holes:
    #   return -1
    else:
      return 0


def test_value_iteration_toy():
  toy_mdp = FrozenLakeMDP()

  gamma = 0.9
  pi, np_v_value, np_q_value = plan_lib.value_iteration(
      toy_mdp.np_transition_model,
      toy_mdp.np_reward_model,
      discount_factor=gamma,
      max_iteration=100,
      epsilon=0.001)

  true_v_values = np.array([
      0.25367878, 0.53623493, 0.33812841, 0.21074621, 0., 0., 0., 0.25719306,
      0.18139626, 0.80532859, 0.35965969, 1., 0.56359908, 0.18139626, 0.,
      0.6630453, 0.
  ])

  assert np.allclose(true_v_values, np_v_value, atol=0.0001, rtol=0.)


def test_soft_value_iteration_toy():
  toy_mdp = FrozenLakeMDP()

  gamma = 0.9
  np_v_value, np_q_value = plan_lib.soft_value_iteration(
      toy_mdp.np_transition_model,
      toy_mdp.np_reward_model,
      discount_factor=gamma,
      max_iteration=100,
      epsilon=0.001)

  true_v_values = np.array([
      4.02017073, 5.01630892, 3.75830779, 4.37395269, 0., 0., 0., 4.54604286,
      3.3518442, 4.9130761, 4.47174755, 1., 4.93311667, 4.04621692, 0.,
      4.87367982, 0.
  ])

  assert np.allclose(true_v_values, np_v_value, atol=0.0001, rtol=0.)


def test_policy_iteration_toy():
  toy_mdp = FrozenLakeMDP()

  gamma = 0.9
  policy_init = np.zeros(toy_mdp.num_states, dtype=int)
  for idx in range(toy_mdp.num_states):
    if len(toy_mdp.legal_actions(idx)) > 0:
      policy_init[idx] = toy_mdp.legal_actions(idx)[0]

  pi, np_v_value, np_q_value = plan_lib.policy_iteration(
      toy_mdp.np_transition_model,
      toy_mdp.np_reward_model,
      discount_factor=gamma,
      max_iteration=20,
      epsilon=0.001,
      policy_initial=policy_init,
  )

  true_v_values = np.array([
      0.25397292, 0.53637719, 0.33818329, 0.21110123, 0., 0., 0., 0.25737311,
      0.1816352, 0.80534705, 0.35988207, 1., 0.56365481, 0.1816352, 0.,
      0.66311948, 0.
  ])

  assert np.allclose(true_v_values, np_v_value, atol=0.0001, rtol=0.)


def test_value_iteration_toy_sparse():
  toy_mdp = FrozenLakeMDP(use_sparse=True)

  gamma = 0.9
  pi, np_v_value, np_q_value = plan_lib.value_iteration(
      toy_mdp.np_transition_model,
      toy_mdp.np_reward_model,
      discount_factor=gamma,
      max_iteration=100,
      epsilon=0.001)

  true_v_values = np.array([
      0.25367878, 0.53623493, 0.33812841, 0.21074621, 0., 0., 0., 0.25719306,
      0.18139626, 0.80532859, 0.35965969, 1., 0.56359908, 0.18139626, 0.,
      0.6630453, 0.
  ])

  assert np.allclose(true_v_values, np_v_value, atol=0.0001, rtol=0.)


def test_soft_value_iteration_toy_sparse():
  toy_mdp = FrozenLakeMDP(use_sparse=True)

  gamma = 0.9
  np_v_value, np_q_value = plan_lib.soft_value_iteration(
      toy_mdp.np_transition_model,
      toy_mdp.np_reward_model,
      discount_factor=gamma,
      max_iteration=100,
      epsilon=0.001)

  true_v_values = np.array([
      4.02017073, 5.01630892, 3.75830779, 4.37395269, 0., 0., 0., 4.54604286,
      3.3518442, 4.9130761, 4.47174755, 1., 4.93311667, 4.04621692, 0.,
      4.87367982, 0.
  ])

  assert np.allclose(true_v_values, np_v_value, atol=0.0001, rtol=0.)


def test_policy_iteration_toy_sparse():
  toy_mdp = FrozenLakeMDP(use_sparse=True)

  gamma = 0.9
  policy_init = np.zeros(toy_mdp.num_states, dtype=int)
  for idx in range(toy_mdp.num_states):
    if len(toy_mdp.legal_actions(idx)) > 0:
      policy_init[idx] = toy_mdp.legal_actions(idx)[0]

  pi, np_v_value, np_q_value = plan_lib.policy_iteration(
      toy_mdp.np_transition_model,
      toy_mdp.np_reward_model,
      discount_factor=gamma,
      max_iteration=20,
      epsilon=0.001,
      policy_initial=policy_init,
  )

  true_v_values = np.array([
      0.25397292, 0.53637719, 0.33818329, 0.21110123, 0., 0., 0., 0.25737311,
      0.1816352, 0.80534705, 0.35988207, 1., 0.56365481, 0.1816352, 0.,
      0.66311948, 0.
  ])

  assert np.allclose(true_v_values, np_v_value, atol=0.0001, rtol=0.)


def test_transition_validity_toy():
  toy_mdp = FrozenLakeMDP()
  assert test_utils.check_transition_validity(toy_mdp)


if __name__ == "__main__":
  toy_mdp = FrozenLakeMDP(use_sparse=True)

  gamma = 0.9

  def get_grid_v_values(pi=None, np_v_value_input=None):
    v_table = np.zeros((toy_mdp.width, toy_mdp.height))
    pi_table = np.zeros((toy_mdp.width, toy_mdp.height), dtype=object)
    for i in range(toy_mdp.width):
      for j in range(toy_mdp.height):
        state = (i, j)
        si = toy_mdp.s_space.state_to_idx[state]
        if np_v_value_input is not None:
          v_table[i, j] = np_v_value_input[si]
        if pi is not None:
          if pi[si] == 0:
            pi_table[i, j] = "Left"
          elif pi[si] == 1:
            pi_table[i, j] = "Down"
          elif pi[si] == 2:
            pi_table[i, j] = "Right"
          elif pi[si] == 3:
            pi_table[i, j] = "Up"
    pi_table[(1, 1)] = "Hole"
    pi_table[(3, 1)] = "Hole"
    pi_table[(3, 2)] = "Hole"
    pi_table[(0, 3)] = "Hole"
    return pi_table.transpose(), v_table.transpose()

  pi, np_v_value, np_q_value = plan_lib.value_iteration(
      toy_mdp.np_transition_model,
      toy_mdp.np_reward_model,
      discount_factor=gamma,
      max_iteration=100,
      epsilon=0.001)

  print("Value Iteration")
  print(pi)
  print(np_v_value)
  pi_table, v_table = get_grid_v_values(pi, np_v_value)
  print(pi_table)
  print(v_table)

  np_v_value, np_q_value = plan_lib.soft_value_iteration(
      toy_mdp.np_transition_model,
      toy_mdp.np_reward_model,
      discount_factor=gamma,
      max_iteration=100,
      epsilon=0.001,
      temperature=1.)

  pi = mdp_lib.deterministic_policy_from_q_value(np_q_value)

  print("Soft Value Iteration")
  print(np_v_value)
  pi_table, v_table = get_grid_v_values(pi, np_v_value)
  print(pi_table)
  print(v_table)

  policy_init = np.zeros(toy_mdp.num_states, dtype=int)
  for idx in range(toy_mdp.num_states):
    if len(toy_mdp.legal_actions(idx)) > 0:
      policy_init[idx] = toy_mdp.legal_actions(idx)[0]

  pi, np_v_value, np_q_value = plan_lib.policy_iteration(
      toy_mdp.np_transition_model,
      toy_mdp.np_reward_model,
      discount_factor=gamma,
      max_iteration=20,
      epsilon=0.001,
      policy_initial=policy_init,
  )

  print("Policy Iteration")
  print(pi)
  print(np_v_value)
  pi_table, v_table = get_grid_v_values(pi, np_v_value)
  print(pi_table)
  print(v_table)
