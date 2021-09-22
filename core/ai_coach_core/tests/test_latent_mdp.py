import numpy as np
import ai_coach_core.models.latent_mdp as mdp_lib
import ai_coach_core.RL.planning as plan_lib
from ai_coach_core.utils.mdp_utils import StateSpace, ActionSpace


class LatentFrozenLakeMDP(mdp_lib.LatentMDP):
  """
      S F F I
      F H F H
      F F F H
      H F F G
  S : starting point, safe
  F : frozen surface, safe
  H : hole, fall to your doom
  I : intermediate stop
  G : goal
  The episode ends when you reach the goal or fall in a hole.
  You receive a reward of 1 if you reach the goal, and zero otherwise.
  """
  LEFT = 0
  DOWN = 1
  RIGHT = 2
  UP = 3
  STAY = 4

  def __init__(self, use_sparse=False):
    self.width = 4
    self.height = 4
    self.holes = [(1, 1), (3, 1), (3, 2), (0, 3)]
    self.inter = (3, 0)
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
        LatentFrozenLakeMDP.LEFT, LatentFrozenLakeMDP.DOWN, LatentFrozenLakeMDP.
        RIGHT, LatentFrozenLakeMDP.UP, LatentFrozenLakeMDP.STAY
    ])
    self.dict_factored_actionspace = {0: self.a_space}

  def init_latentspace(self):
    """Defines MDP latent state space. """
    self.latent_space = StateSpace([0, 1])

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

    legal_act = [LatentFrozenLakeMDP.STAY]
    if state[0] != 0:
      legal_act.append(action_to_idx(LatentFrozenLakeMDP.LEFT))
    if state[0] != self.width - 1:
      legal_act.append(action_to_idx(LatentFrozenLakeMDP.RIGHT))
    if state[1] != 0:
      legal_act.append(action_to_idx(LatentFrozenLakeMDP.UP))
    if state[1] != self.height - 1:
      legal_act.append(action_to_idx(LatentFrozenLakeMDP.DOWN))

    return legal_act

  def transition_model(self, state_idx: int, action_idx: int) -> np.ndarray:
    if self.is_terminal(state_idx):
      return np.array([[1.0, state_idx]])

    # unpack the input
    s_i, = self.conv_idx_to_state(state_idx)
    stt = self.s_space.idx_to_state[s_i]

    a_i, = self.conv_idx_to_action(action_idx)
    act = self.a_space.idx_to_action[a_i]
    if act == LatentFrozenLakeMDP.STAY:
      return np.array([[1.0, state_idx]])

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
    if act == LatentFrozenLakeMDP.LEFT:
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
    elif act == LatentFrozenLakeMDP.RIGHT:
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
    elif act == LatentFrozenLakeMDP.UP:
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
    elif act == LatentFrozenLakeMDP.DOWN:
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

  def reward(self, latent_idx: int, state_idx: int, action_idx: int) -> float:
    if self.is_terminal(state_idx):
      return 0

    s_i, = self.conv_idx_to_state(state_idx)
    state = self.s_space.idx_to_state[s_i]
    a_i, = self.conv_idx_to_action(action_idx)
    act = self.a_space.idx_to_action[a_i]
    reward = 0
    if act == LatentFrozenLakeMDP.STAY:
      reward -= 0.1

    lat = self.latent_space.idx_to_state[latent_idx]
    if lat == 0:
      if state == self.inter:
        if act == LatentFrozenLakeMDP.STAY:
          return reward + 0.2
        else:
          return -np.inf
      elif state in self.holes:
        return reward + -1
      else:
        return reward + 0
    elif lat == 1:
      if state == self.goal:
        return reward + 1
      elif state in self.holes:
        return reward + -1
      else:
        return reward + 0

    return -np.inf


def test_value_iteration_toy():
  toy_mdp = LatentFrozenLakeMDP()
  gamma = 0.9

  list_np_v_values = []
  for idx in range(toy_mdp.num_latents):
    pi, np_v_value, np_q_value = plan_lib.value_iteration(
        toy_mdp.np_transition_model,
        toy_mdp.np_reward_model[idx],
        discount_factor=gamma,
        max_iteration=100,
        epsilon=0.001)
    list_np_v_values.append(np_v_value)

  true_v_values_0 = [
      -0.01097586, 0.01453205, 0.13558896, 0.23484232, -1., -1., -1.,
      0.72640975, 0.99582544, 0.02182906, -0.00544168, 0., 0.03216008,
      0.33592657, -1., 0.01780635, 0.
  ]

  true_v_values_1 = [
      -0.10419266, 0.4472244, -0.07147478, -0.12001599, -1., -1., -1.,
      -0.12160987, -0.2624663, 0.7730932, 0.10260667, 1., 0.38091477,
      -0.26155562, -1., 0.62459619, 0.
  ]

  assert np.allclose(true_v_values_0, list_np_v_values[0], atol=0.0001, rtol=0.)
  assert np.allclose(true_v_values_1, list_np_v_values[1], atol=0.0001, rtol=0.)


def test_value_iteration_toy_sparse():
  toy_mdp = LatentFrozenLakeMDP(use_sparse=True)
  gamma = 0.9

  list_np_v_values = []
  for idx in range(toy_mdp.num_latents):
    pi, np_v_value, np_q_value = plan_lib.value_iteration(
        toy_mdp.np_transition_model,
        toy_mdp.np_reward_model[idx],
        discount_factor=gamma,
        max_iteration=100,
        epsilon=0.001)
    list_np_v_values.append(np_v_value)

  true_v_values_0 = [
      -0.01097586, 0.01453205, 0.13558896, 0.23484232, -1., -1., -1.,
      0.72640975, 0.99582544, 0.02182906, -0.00544168, 0., 0.03216008,
      0.33592657, -1., 0.01780635, 0.
  ]

  true_v_values_1 = [
      -0.10419266, 0.4472244, -0.07147478, -0.12001599, -1., -1., -1.,
      -0.12160987, -0.2624663, 0.7730932, 0.10260667, 1., 0.38091477,
      -0.26155562, -1., 0.62459619, 0.
  ]

  assert np.allclose(true_v_values_0, list_np_v_values[0], atol=0.0001, rtol=0.)
  assert np.allclose(true_v_values_1, list_np_v_values[1], atol=0.0001, rtol=0.)


if __name__ == "__main__":
  toy_mdp = LatentFrozenLakeMDP(use_sparse=False)
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
          elif pi[si] == 4:
            pi_table[i, j] = "Stay"
    pi_table[(1, 1)] = "Hole"
    pi_table[(3, 1)] = "Hole"
    pi_table[(3, 2)] = "Hole"
    pi_table[(0, 3)] = "Hole"
    return pi_table.transpose(), v_table.transpose()

  for idx in range(toy_mdp.num_latents):
    pi, np_v_value, np_q_value = plan_lib.value_iteration(
        toy_mdp.np_transition_model,
        toy_mdp.np_reward_model[idx],
        discount_factor=gamma,
        max_iteration=100,
        epsilon=0.001)

    print("V-Value latent-%d" % (idx, ))
    print(pi)
    print(np_v_value)
    pi_table, v_table = get_grid_v_values(pi, np_v_value)
    print(pi_table)
    print(v_table)
