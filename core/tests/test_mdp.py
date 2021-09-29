import numpy as np
import logging
import ai_coach_core.models.mdp as mdp_lib
import ai_coach_core.RL.planning as plan_lib
from ai_coach_core.utils.mdp_utils import StateSpace, ActionSpace
import ai_coach_core.utils.test_utils as test_utils


###############################################################################
# Test Models
###############################################################################
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
  LEFT = 0
  DOWN = 1
  RIGHT = 2
  UP = 3

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
    self.a_space = ActionSpace(actionspace=[
        FrozenLakeMDP.LEFT, FrozenLakeMDP.DOWN, FrozenLakeMDP.RIGHT,
        FrozenLakeMDP.UP
    ])
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
      legal_act.append(action_to_idx(FrozenLakeMDP.LEFT))
    if state[0] != self.width - 1:
      legal_act.append(action_to_idx(FrozenLakeMDP.RIGHT))
    if state[1] != 0:
      legal_act.append(action_to_idx(FrozenLakeMDP.UP))
    if state[1] != self.height - 1:
      legal_act.append(action_to_idx(FrozenLakeMDP.DOWN))

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
    if act == FrozenLakeMDP.LEFT:
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
    elif act == FrozenLakeMDP.RIGHT:
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
    elif act == FrozenLakeMDP.UP:
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
    elif act == FrozenLakeMDP.DOWN:
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

  def reward(self, state_idx: int, action_idx: int) -> float:
    if self.is_terminal(state_idx):
      return 0

    s_i, = self.conv_idx_to_state(state_idx)
    state = self.s_space.idx_to_state[s_i]
    if state == self.goal:
      return 1
    elif state in self.holes:
      return -1
    else:
      return 0


class FrozenLakeMDPLargeReward(FrozenLakeMDP):
  def reward(self, state_idx: int, action_idx: int) -> float:
    if self.is_terminal(state_idx):
      return 0

    s_i, = self.conv_idx_to_state(state_idx)
    state = self.s_space.idx_to_state[s_i]
    if state == self.goal:
      return 100
    elif state in self.holes:
      return -100
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

  true_v_values = [
      -0.10419714, 0.44722222, -0.07147563, -0.12002533, -1., -1., -1.,
      -0.12161469, -0.2624726, 0.77309292, 0.10260353, 1., 0.38091399,
      -0.26156758, -1., 0.62459513, 0.
  ]

  assert np.allclose(true_v_values, np_v_value, atol=0.0001, rtol=0.)

  soft_pi = mdp_lib.softmax_policy_from_q_value(np_q_value)
  true_soft_pi = [[0., 0.40285736, 0.24471172, 0.35243092],
                  [0.23122445, 0.35976952, 0.27326633, 0.1357397],
                  [0.2066682, 0.33723388, 0.2066682, 0.24942972],
                  [0., 0.51772445, 0.48227555, 0.], [1., 0., 0., 0.],
                  [1., 0., 0., 0.], [1., 0., 0., 0.],
                  [0.32611822, 0.34788286, 0.32599892, 0.],
                  [0.59753234, 0.40246766, 0., 0.],
                  [0.31448087, 0., 0.39394004, 0.2915791],
                  [0., 0.21878471, 0.40677427, 0.37444101], [1., 0., 0., 0.],
                  [0.32959284, 0.33204518, 0.13832293, 0.20003904],
                  [0.37221262, 0.25587745, 0.37190993, 0.], [1., 0., 0., 0.],
                  [0.17069074, 0., 0.49457589, 0.33473337], [1., 0., 0., 0.]]

  assert np.allclose(true_soft_pi, soft_pi, atol=0.0001, rtol=0.)


def test_soft_value_iteration_toy():
  toy_mdp = FrozenLakeMDP()

  gamma = 0.9
  np_v_value, np_q_value = plan_lib.soft_value_iteration(
      toy_mdp.np_transition_model,
      toy_mdp.np_reward_model,
      discount_factor=gamma,
      max_iteration=100,
      epsilon=0.001)

  true_v_values = [
      3.49651174, 4.60783148, 3.15563615, 3.90679818, -1., -1., -1., 4.0435257,
      2.79330297, 4.61366732, 4.00216669, 1., 4.51553684, 3.51108191, -1.,
      4.52279111, 0.
  ]

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

  true_v_values = [
      -0.1038797, 0.44737575, -0.07141636, -0.11934478, -1., -1., -1.,
      -0.12141501, -0.26241833, 0.77311281, 0.10284152, 1., 0.38097357,
      -0.26072438, -1., 0.6246746, 0.
  ]

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

  true_v_values = [
      -0.10419714, 0.44722222, -0.07147563, -0.12002533, -1., -1., -1.,
      -0.12161469, -0.2624726, 0.77309292, 0.10260353, 1., 0.38091399,
      -0.26156758, -1., 0.62459513, 0.
  ]

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

  true_v_values = [
      3.49651174, 4.60783148, 3.15563615, 3.90679818, -1., -1., -1., 4.0435257,
      2.79330297, 4.61366732, 4.00216669, 1., 4.51553684, 3.51108191, -1.,
      4.52279111, 0.
  ]

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

  true_v_values = [
      -0.1038797, 0.44737575, -0.07141636, -0.11934478, -1., -1., -1.,
      -0.12141501, -0.26241833, 0.77311281, 0.10284152, 1., 0.38097357,
      -0.26072438, -1., 0.6246746, 0.
  ]

  assert np.allclose(true_v_values, np_v_value, atol=0.0001, rtol=0.)


def test_value_iteration_toy_large_r():
  toy_mdp = FrozenLakeMDPLargeReward()

  gamma = 0.9
  pi, np_v_value, np_q_value = plan_lib.value_iteration(
      toy_mdp.np_transition_model,
      toy_mdp.np_reward_model,
      discount_factor=gamma,
      max_iteration=100,
      epsilon=0.001)

  true_v_values = [
      -10.38749348, 44.7378058, -7.14154661, -11.93329614, -100., -100., -100.,
      -12.12675995, -26.20729746, 77.31131124, 10.28450564, 100., 38.09744522,
      -26.07084431, -100., 62.46757813, 0.
  ]

  assert np.allclose(true_v_values, np_v_value, atol=0.0001, rtol=0.)


def test_soft_value_iteration_toy_large_r():
  toy_mdp = FrozenLakeMDPLargeReward()

  gamma = 0.9
  np_v_value, np_q_value = plan_lib.soft_value_iteration(
      toy_mdp.np_transition_model,
      toy_mdp.np_reward_model,
      discount_factor=gamma,
      max_iteration=100,
      epsilon=0.001)

  true_v_values = [
      -10.32205029, 44.87600258, -6.84860371, -11.71808827, -100., -100., -100.,
      -11.76685238, -25.95336616, 77.39746581, 10.37739743, 100., 38.58577821,
      -25.12065333, -100., 62.55272753, 0.
  ]

  assert np.allclose(true_v_values, np_v_value, atol=0.0001, rtol=0.)


def test_transition_validity_toy():
  toy_mdp = FrozenLakeMDP()
  assert test_utils.check_transition_validity(toy_mdp)


if __name__ == "__main__":
  logging.basicConfig(filename='myapp.log',
                      level=logging.INFO,
                      format='%(asctime)s:[%(levelname)s]%(message)s')
  logging.info("Started")

  toy_mdp = FrozenLakeMDP(use_sparse=False)

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

  soft_pi = mdp_lib.softmax_policy_from_q_value(np_q_value, temperature=1)
  print(soft_pi)

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

  toy_mdp_large_reward = FrozenLakeMDPLargeReward(use_sparse=False)

  pi, np_v_value, np_q_value = plan_lib.value_iteration(
      toy_mdp_large_reward.np_transition_model,
      toy_mdp_large_reward.np_reward_model,
      discount_factor=gamma,
      max_iteration=100,
      epsilon=0.001)

  print("Value Iteration-large")
  print(pi)
  print(np_v_value)
  pi_table, v_table = get_grid_v_values(pi, np_v_value)
  print(pi_table)
  print(v_table)

  np_v_value, np_q_value = plan_lib.soft_value_iteration(
      toy_mdp_large_reward.np_transition_model,
      toy_mdp_large_reward.np_reward_model,
      discount_factor=gamma,
      max_iteration=100,
      epsilon=0.001,
      temperature=1.)

  pi = mdp_lib.deterministic_policy_from_q_value(np_q_value)

  print("Soft Value Iteration-large")
  print(np_v_value)
  pi_table, v_table = get_grid_v_values(pi, np_v_value)
  print(pi_table)
  print(v_table)
