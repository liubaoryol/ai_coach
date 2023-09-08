import numpy as np
import logging
import aic_core.models.mdp as mdp_lib
import aic_core.RL.planning as plan_lib
from aic_core.utils.mdp_utils import StateSpace, ActionSpace
import aic_core.utils.test_utils as test_utils


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

    list_state = []
    for i in range(self.width):
      for j in range(self.height):
        state = (i, j)
        list_state.append(state)

    self.s_space = StateSpace(statespace=list_state)
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


def get_grid_v_values(mdp: FrozenLakeMDP,
                      pi=None,
                      np_v_value_input=None,
                      soft_pi_value=None):
  v_table = np.zeros((mdp.width, mdp.height))
  pi_table = np.zeros((mdp.width, mdp.height), dtype=object)
  np_soft_pi = np.zeros((mdp.width, mdp.height, mdp.num_actions))
  for i in range(mdp.width):
    for j in range(mdp.height):
      state = (i, j)
      si = mdp.s_space.state_to_idx[state]
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
      if soft_pi_value is not None:
        np_soft_pi[state][:] = soft_pi_value[si, :]
  pi_table[(1, 1)] = "Hole"
  pi_table[(3, 1)] = "Hole"
  pi_table[(3, 2)] = "Hole"
  pi_table[(0, 3)] = "Hole"
  return (pi_table.transpose(), v_table.transpose(),
          np_soft_pi.transpose((1, 0, 2)))


VALUE_ITERATION_TRUE = [[-0.12002533, -0.26156758, -0.12161469, -0.2624726],
                        [-0.10419714, -1., -0.07147563, -1.],
                        [0.10260353, 0.44722222, 0.38091399, -1.],
                        [-1., 0.62459513, 0.77309292, 1.]]

SOFT_VALUE_ITERATION_TRUE = [
    [10.70806268, 11.17584681, 11.30611006, 10.86642034],
    [11.24755151, 11.47466212, 11.83938858, 11.47466212],
    [11.53350319, 11.975386, 12.1304413, 11.47466212],
    [11.47466212, 11.87194297, 12.39893502, 13.47466212]
]

POLICY_ITERATION_TRUE = [[-0.11934478, -0.26072438, -0.12141501, -0.26241833],
                         [-0.1038797, -1., -0.07141636, -1.],
                         [0.10284152, 0.44737575, 0.38097357, -1.],
                         [-1., 0.6246746, 0.77311281, 1.]]


def test_value_iteration_toy():
  toy_mdp = FrozenLakeMDP()

  gamma = 0.9
  pi, np_v_value, np_q_value = plan_lib.value_iteration(
      toy_mdp.np_transition_model,
      toy_mdp.np_reward_model,
      discount_factor=gamma,
      max_iteration=100,
      epsilon=0.001)

  soft_pi = mdp_lib.softmax_policy_from_q_value(np_q_value)

  pi_table, v_table, soft_pi_table = get_grid_v_values(toy_mdp, pi, np_v_value,
                                                       soft_pi)

  assert np.allclose(VALUE_ITERATION_TRUE, v_table, atol=0.0001, rtol=0.)

  true_soft_pi = [[[0., 0.51772445, 0.48227555, 0.],
                   [0.37221262, 0.25587745, 0.37190993, 0.],
                   [0.32611822, 0.34788286, 0.32599892, 0.],
                   [0.59753234, 0.40246766, 0., 0.]],
                  [[0., 0.40285736, 0.24471172, 0.35243092], [1., 0., 0., 0.],
                   [0.2066682, 0.33723388, 0.2066682, 0.24942972],
                   [1., 0., 0., 0.]],
                  [[0., 0.21878471, 0.40677427, 0.37444101],
                   [0.23122445, 0.35976952, 0.27326633, 0.1357397],
                   [0.32959284, 0.33204518, 0.13832293, 0.20003904],
                   [1., 0., 0., 0.]],
                  [[1., 0., 0., 0.], [0.17069074, 0., 0.49457589, 0.33473337],
                   [0.31448087, 0., 0.39394004, 0.2915791], [1., 0., 0., 0.]]]

  assert np.allclose(true_soft_pi, soft_pi_table, atol=0.0001, rtol=0.)


def test_value_iteration_toy_sparse():
  toy_mdp = FrozenLakeMDP(use_sparse=True)

  gamma = 0.9
  pi, np_v_value, np_q_value = plan_lib.value_iteration(
      toy_mdp.np_transition_model,
      toy_mdp.np_reward_model,
      discount_factor=gamma,
      max_iteration=100,
      epsilon=0.001)

  _, v_table, _ = get_grid_v_values(toy_mdp, None, np_v_value)

  assert np.allclose(VALUE_ITERATION_TRUE, v_table, atol=0.0001, rtol=0.)


def test_soft_value_iteration_toy():
  toy_mdp = FrozenLakeMDP()

  gamma = 0.9
  np_v_value, np_q_value = plan_lib.soft_value_iteration(
      toy_mdp.np_transition_model,
      toy_mdp.np_reward_model,
      discount_factor=gamma,
      max_iteration=100,
      epsilon=0.001)

  _, v_table, _ = get_grid_v_values(toy_mdp, None, np_v_value)

  assert np.allclose(SOFT_VALUE_ITERATION_TRUE, v_table, atol=0.0001, rtol=0.)


def test_soft_value_iteration_toy_sparse():
  toy_mdp = FrozenLakeMDP(use_sparse=True)

  gamma = 0.9
  np_v_value, np_q_value = plan_lib.soft_value_iteration(
      toy_mdp.np_transition_model,
      toy_mdp.np_reward_model,
      discount_factor=gamma,
      max_iteration=100,
      epsilon=0.001)

  _, v_table, _ = get_grid_v_values(toy_mdp, None, np_v_value)

  assert np.allclose(SOFT_VALUE_ITERATION_TRUE, v_table, atol=0.0001, rtol=0.)


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

  _, v_table, _ = get_grid_v_values(toy_mdp, None, np_v_value)

  assert np.allclose(POLICY_ITERATION_TRUE, v_table, atol=0.0001, rtol=0.)


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

  _, v_table, _ = get_grid_v_values(toy_mdp, None, np_v_value)

  assert np.allclose(POLICY_ITERATION_TRUE, v_table, atol=0.0001, rtol=0.)


def test_value_iteration_toy_large_r():
  toy_mdp = FrozenLakeMDPLargeReward()

  gamma = 0.9
  pi, np_v_value, np_q_value = plan_lib.value_iteration(
      toy_mdp.np_transition_model,
      toy_mdp.np_reward_model,
      discount_factor=gamma,
      max_iteration=100,
      epsilon=0.001)

  _, v_table, _ = get_grid_v_values(toy_mdp, None, np_v_value)

  true_v_values = [[-11.93329614, -26.07084431, -12.12675995, -26.20729746],
                   [-10.38749348, -100., -7.14154661, -100.],
                   [10.28450564, 44.7378058, 38.09744522, -100.],
                   [-100., 62.46757813, 77.31131124, 100.]]

  assert np.allclose(true_v_values, v_table, atol=0.0001, rtol=0.)


def test_soft_value_iteration_toy_large_r():
  toy_mdp = FrozenLakeMDPLargeReward()

  gamma = 0.9
  np_v_value, np_q_value = plan_lib.soft_value_iteration(
      toy_mdp.np_transition_model,
      toy_mdp.np_reward_model,
      discount_factor=gamma,
      max_iteration=100,
      epsilon=0.001)

  _, v_table, _ = get_grid_v_values(toy_mdp, None, np_v_value)

  true_v_values = [[-5.07378462, -17.80361963, -4.06680446, -18.31656778],
                   [-2.73851924, -87.52533788, 2.24941921, -87.52533788],
                   [18.00209525, 52.56959288, 47.51203204, -87.52533788],
                   [-87.52533788, 71.23778488, 87.77812468, 112.47466212]]

  assert np.allclose(true_v_values, v_table, atol=0.0001, rtol=0.)


def test_transition_validity_toy():
  toy_mdp = FrozenLakeMDP()
  assert test_utils.check_transition_validity(toy_mdp)


if __name__ == "__main__":
  import sys
  logging.basicConfig(  # filename='myapp.log',
      level=logging.DEBUG,
      stream=sys.stdout,
      format='%(asctime)s:[%(levelname)s]%(message)s')
  logging.info("Started")

  toy_mdp = FrozenLakeMDP(use_sparse=False)

  gamma = 0.9
  pi, np_v_value, np_q_value = plan_lib.value_iteration(
      toy_mdp.np_transition_model,
      toy_mdp.np_reward_model,
      discount_factor=gamma,
      max_iteration=100,
      epsilon=0.001)

  logging.debug("Value Iteration")

  soft_pi = mdp_lib.softmax_policy_from_q_value(np_q_value, temperature=1)
  pi_table, v_table, soft_pi_table = get_grid_v_values(toy_mdp, pi, np_v_value,
                                                       soft_pi)
  logging.debug(pi_table)
  logging.debug(v_table)
  logging.debug(soft_pi_table)

  np_v_value, np_q_value = plan_lib.soft_value_iteration(
      toy_mdp.np_transition_model,
      toy_mdp.np_reward_model,
      discount_factor=gamma,
      max_iteration=100,
      epsilon=0.001,
      temperature=1.)

  pi = mdp_lib.deterministic_policy_from_q_value(np_q_value)

  logging.debug("Soft Value Iteration")
  pi_table, v_table, _ = get_grid_v_values(toy_mdp, pi, np_v_value)
  logging.debug(pi_table)
  logging.debug(v_table)

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

  logging.debug("Policy Iteration")
  pi_table, v_table, _ = get_grid_v_values(toy_mdp, pi, np_v_value)
  logging.debug(pi_table)
  logging.debug(v_table)

  toy_mdp_large_reward = FrozenLakeMDPLargeReward(use_sparse=False)

  pi, np_v_value, np_q_value = plan_lib.value_iteration(
      toy_mdp_large_reward.np_transition_model,
      toy_mdp_large_reward.np_reward_model,
      discount_factor=gamma,
      max_iteration=100,
      epsilon=0.001)

  logging.debug("Value Iteration-large")
  pi_table, v_table, _ = get_grid_v_values(toy_mdp_large_reward, pi, np_v_value)
  logging.debug(pi_table)
  logging.debug(v_table)

  np_v_value, np_q_value = plan_lib.soft_value_iteration(
      toy_mdp_large_reward.np_transition_model,
      toy_mdp_large_reward.np_reward_model,
      discount_factor=gamma,
      max_iteration=100,
      epsilon=0.001,
      temperature=1.)

  pi = mdp_lib.deterministic_policy_from_q_value(np_q_value)

  logging.debug("Soft Value Iteration-large")
  pi_table, v_table, _ = get_grid_v_values(toy_mdp_large_reward, pi, np_v_value)
  logging.debug(pi_table)
  logging.debug(v_table)
