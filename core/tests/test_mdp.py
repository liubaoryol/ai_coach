from enum import Enum
import numpy as np
import models.mdp as mdp_lib
import RL.planning as plan_lib
from utils.mdp_utils import StateSpace, ActionSpace


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
  def __init__(self):
    self.width = 4
    self.height = 4
    self.holes = [(1, 1), (3, 1), (3, 2), (0, 3)]
    self.goal = (3, 3)
    super().__init__()

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
    p_move = 1. / 3.
    if act == FL_Actions.LEFT:
      if stt[0] != 0:
        sttn = (stt[0] - 1, stt[1])
        p_remain -= p_move
        list_next_p_state.append((p_move, state_to_idx(sttn)))
      if stt[1] != self.height - 1:
        sttn = (stt[0], stt[1] + 1)
        p_remain -= p_move
        list_next_p_state.append((p_move, state_to_idx(sttn)))
      if stt[1] != 0:
        sttn = (stt[0], stt[1] - 1)
        p_remain -= p_move
        list_next_p_state.append((p_move, state_to_idx(sttn)))
    elif act == FL_Actions.RIGHT:
      if stt[0] != self.width - 1:
        sttn = (stt[0] + 1, stt[1])
        p_remain -= p_move
        list_next_p_state.append((p_move, state_to_idx(sttn)))
      if stt[1] != self.height - 1:
        sttn = (stt[0], stt[1] + 1)
        p_remain -= p_move
        list_next_p_state.append((p_move, state_to_idx(sttn)))
      if stt[1] != 0:
        sttn = (stt[0], stt[1] - 1)
        p_remain -= p_move
        list_next_p_state.append((p_move, state_to_idx(sttn)))
    elif act == FL_Actions.UP:
      if stt[1] != 0:
        sttn = (stt[0], stt[1] - 1)
        p_remain -= p_move
        list_next_p_state.append((p_move, state_to_idx(sttn)))
      if stt[0] != 0:
        sttn = (stt[0] - 1, stt[1])
        p_remain -= p_move
        list_next_p_state.append((p_move, state_to_idx(sttn)))
      if stt[0] != self.width - 1:
        sttn = (stt[0] + 1, stt[1])
        p_remain -= p_move
        list_next_p_state.append((p_move, state_to_idx(sttn)))
    elif act == FL_Actions.DOWN:
      if stt[1] != self.height - 1:
        sttn = (stt[0], stt[1] + 1)
        p_remain -= p_move
        list_next_p_state.append((p_move, state_to_idx(sttn)))
      if stt[0] != 0:
        sttn = (stt[0] - 1, stt[1])
        p_remain -= p_move
        list_next_p_state.append((p_move, state_to_idx(sttn)))
      if stt[0] != self.width - 1:
        sttn = (stt[0] + 1, stt[1])
        p_remain -= p_move
        list_next_p_state.append((p_move, state_to_idx(sttn)))

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
      0.04407126, 0.19842536, 0.08759133, 0.02816701, 0., 0., 0., 0.04684832,
      0.01991426, 0.53364746, 0.10347795, 1., 0.24566218, 0.02235332, 0.,
      0.31333924, 0.
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
      3.3775173, 4.48612681, 3.53670304, 3.90625381, 0., 0., 0., 4.2822264,
      2.82530511, 4.55421947, 4.09057361, 1., 4.39429656, 3.42715074, 0.,
      4.46914926, 0.
  ])

  assert np.allclose(true_v_values, np_v_value, atol=0.0001, rtol=0.)


if __name__ == "__main__":
  toy_mdp = FrozenLakeMDP()

  gamma = 0.9
  pi, np_v_value, np_q_value = plan_lib.value_iteration(
      toy_mdp.np_transition_model,
      toy_mdp.np_reward_model,
      discount_factor=gamma,
      max_iteration=100,
      epsilon=0.001)

  v_table = np.zeros((toy_mdp.width, toy_mdp.height))
  for i in range(toy_mdp.width):
    for j in range(toy_mdp.height):
      state = (i, j)
      si = toy_mdp.s_space.state_to_idx[state]
      v_table[i, j] = np_v_value[si]

  print("Value Iteration")
  print(np_v_value)
  print(v_table.transpose())

  np_v_value, np_q_value = plan_lib.soft_value_iteration(
      toy_mdp.np_transition_model,
      toy_mdp.np_reward_model,
      discount_factor=gamma,
      max_iteration=100,
      epsilon=0.001,
      temperature=1.)

  v_table = np.zeros((toy_mdp.width, toy_mdp.height))
  for i in range(toy_mdp.width):
    for j in range(toy_mdp.height):
      state = (i, j)
      si = toy_mdp.s_space.state_to_idx[state]
      v_table[i, j] = np_v_value[si]

  print("Soft Value Iteration")
  print(np_v_value)
  print(v_table.transpose())
