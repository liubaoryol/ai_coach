import numpy as np
from ai_coach_core.models.latent_mdp import LatentMDP
from ai_coach_core.utils.mdp_utils import StateSpace, ActionSpace
from ai_coach_domain.box_push.helper import EventType


def transition_navi(box_states: list, a1_pos, a2_pos, a1_act, a2_act,
                    box_locations: list, goals: list, walls: list, drops: list,
                    x_bound, y_bound):

  if a1_pos in goals[0:2] and a2_pos in goals[2:4]:
    return [[1.0, None, None]]

  def get_moved_coord_impl(coord, action, x_bound, y_bound, walls):
    x, y = coord
    coord_new = None
    if action == EventType.UP:
      coord_new = (x, y - 1)
    elif action == EventType.DOWN:
      coord_new = (x, y + 1)
    elif action == EventType.LEFT:
      coord_new = (x - 1, y)
    elif action == EventType.RIGHT:
      coord_new = (x + 1, y)
    else:
      return coord

    def is_wall_impl(coord_l, x_bound, y_bound, walls):
      xl, yl = coord_l
      if xl < 0 or xl >= x_bound or yl < 0 or yl >= y_bound:
        return True

      if coord_l in walls:
        return True

      return False

    if is_wall_impl(coord_new, x_bound, y_bound, walls):
      return coord
      # coord_new = coord

    return coord_new

  a1_pos_n = get_moved_coord_impl(a1_pos, a1_act, x_bound, y_bound, walls)
  a2_pos_n = get_moved_coord_impl(a2_pos, a2_act, x_bound, y_bound, walls)

  if a1_pos == a2_pos:
    P_MOVE = 0.1
    list_np_next_p_state = []
    if a1_pos == a1_pos_n and a2_pos == a2_pos_n:
      list_np_next_p_state.append([1, a1_pos, a2_pos])
    else:
      list_np_next_p_state.append([P_MOVE, a1_pos_n, a2_pos_n])
      list_np_next_p_state.append([1 - P_MOVE, a1_pos, a2_pos])
    return list_np_next_p_state
  else:
    return [[1.0, a1_pos_n, a2_pos_n]]


class NavigationMDP(LatentMDP):
  def __init__(self, x_grid, y_grid, goals, walls, *args, **kwargs):
    self.x_grid = x_grid
    self.y_grid = y_grid
    self.goals = goals
    self.walls = walls
    super().__init__(use_sparse=True)

  def init_statespace(self):
    '''
    To disable dummy states, set self.dummy_states = None
    '''

    self.dict_factored_statespace = {}

    set_grid = set()
    for i in range(self.x_grid):
      for j in range(self.y_grid):
        state = (i, j)
        if state not in self.walls:
          set_grid.add(state)
    self.a1_pos_space = StateSpace(statespace=set_grid)
    self.a2_pos_space = StateSpace(statespace=set_grid)
    self.dict_factored_statespace = {0: self.a1_pos_space, 1: self.a2_pos_space}

    self.dummy_states = StateSpace(statespace=["terminal"])

  def init_actionspace(self):
    self.dict_factored_actionspace = {}
    action_states = [EventType(i) for i in range(5)]

    self.a1_a_space = ActionSpace(actionspace=action_states)
    self.a2_a_space = ActionSpace(actionspace=action_states)
    self.dict_factored_actionspace = {0: self.a1_a_space, 1: self.a2_a_space}

  def init_latentspace(self):
    latent_states = [("GH1", 1), ("GH2", 1), ("GH1", 2), ("GH2", 2)]

    self.latent_space = StateSpace(latent_states)

  def is_terminal(self, state_idx):
    if self.is_dummy_state(state_idx):
      return True

    return False

  def legal_actions(self, state_idx):
    if self.is_terminal(state_idx):
      return []

    return range(self.num_actions)

  def transition_model(self, state_idx: int, action_idx: int) -> np.ndarray:
    if self.is_terminal(state_idx):
      return np.array([[1.0, state_idx]])

    sidx1, sidx2 = self.conv_idx_to_state(state_idx)
    a1_pos = self.a1_pos_space.idx_to_state[sidx1]
    a2_pos = self.a2_pos_space.idx_to_state[sidx2]

    aidx1, aidx2 = self.conv_idx_to_action(action_idx)
    a1_act = self.a1_a_space.idx_to_action[aidx1]
    a2_act = self.a2_a_space.idx_to_action[aidx2]

    list_p_next_env = transition_navi([], a1_pos, a2_pos, a1_act, a2_act, [],
                                      self.goals, self.walls, [], self.x_grid,
                                      self.y_grid)

    list_next_p_state = []
    map_next_state = {}
    for p, a1_pos_n, a2_pos_n in list_p_next_env:
      if a1_pos_n is None or a2_pos_n is None:
        return np.array([[1.0, self.conv_dummy_state_to_idx("terminal")]])

      sidx_n1 = self.a1_pos_space.state_to_idx[a1_pos_n]
      sidx_n2 = self.a2_pos_space.state_to_idx[a2_pos_n]
      sidx = self.conv_state_to_idx(tuple([sidx_n1, sidx_n2]))

      map_next_state[sidx] = map_next_state.get(sidx, 0) + p

    for key in map_next_state:
      val = map_next_state[key]
      list_next_p_state.append([val, key])

    return np.array(list_next_p_state)

  def reward(self, latent_idx: int, state_idx: int, action_idx: int) -> float:
    if self.is_terminal(state_idx):
      return 0

    sidx1, sidx2 = self.conv_idx_to_state(state_idx)
    a1_pos = self.a1_pos_space.idx_to_state[sidx1]
    a2_pos = self.a2_pos_space.idx_to_state[sidx2]

    aidx1, aidx2 = self.conv_idx_to_action(action_idx)
    a1_act = self.a1_a_space.idx_to_action[aidx1]
    a2_act = self.a2_a_space.idx_to_action[aidx2]

    latent = self.latent_space.idx_to_state[latent_idx]

    reward = -1
    if latent[0] == "GH1" and latent[1] == 1:
      if a1_pos == self.goals[0] and a2_pos == self.goals[2]:
        return -300
    if latent[0] == "GH1" and latent[1] == 2:
      if a1_pos == self.goals[0] and a2_pos == self.goals[3]:
        return -200
    if latent[0] == "GH2" and latent[1] == 1:
      if a1_pos == self.goals[1] and a2_pos == self.goals[2]:
        return -200
    if latent[0] == "GH2" and latent[1] == 2:
      if a1_pos == self.goals[1] and a2_pos == self.goals[3]:
        return -200

    # if a1_pos in self.goals[0:2] and a2_pos in self.goals[2:4]:
    #   return 1000

    return reward


# class NavigationMDP_Agent1(LatentMDP):
#   def __init__(self, x_grid, y_grid, goals, walls, *args, **kwargs):
#     self.x_grid = x_grid
#     self.y_grid = y_grid
#     self.goals = goals
#     self.walls = walls
#     super().__init__(use_sparse=True)

#   def init_statespace(self):
#     '''
#     To disable dummy states, set self.dummy_states = None
#     '''

#     self.dict_factored_statespace = {}

#     set_grid = set()
#     for i in range(self.x_grid):
#       for j in range(self.y_grid):
#         state = (i, j)
#         if state not in self.walls:
#           set_grid.add(state)
#     self.a1_pos_space = StateSpace(statespace=set_grid)
#     self.a2_pos_space = StateSpace(statespace=set_grid)
#     self.dict_factored_statespace = {0: self.a1_pos_space, 1: self.a2_pos_space}

#     self.dummy_states = StateSpace(statespace=["terminal"])

#   def init_actionspace(self):
#     self.dict_factored_actionspace = {}
#     action_states = [EventType(i) for i in range(5)]

#     self.a1_a_space = ActionSpace(actionspace=action_states)
#     self.a2_a_space = ActionSpace(actionspace=action_states)
#     self.dict_factored_actionspace = {0: self.a1_a_space, 1: self.a2_a_space}

#   def init_latentspace(self):
#     latent_states = [("Goal1", 0), ("Goal2", 0)]

#     self.latent_space = StateSpace(latent_states)

#   def is_terminal(self, state_idx):
#     if self.is_dummy_state(state_idx):
#       return True

#     return False

#   def legal_actions(self, state_idx):
#     if self.is_terminal(state_idx):
#       return []

#     return range(self.num_actions)

#   def transition_model(self, state_idx: int, action_idx: int) -> np.ndarray:
#     if self.is_terminal(state_idx):
#       return np.array([[1.0, state_idx]])

#     sidx1, sidx2 = self.conv_idx_to_state(state_idx)
#     a1_pos = self.a1_pos_space.idx_to_state[sidx1]
#     a2_pos = self.a2_pos_space.idx_to_state[sidx2]

#     aidx1, aidx2 = self.conv_idx_to_action(action_idx)
#     a1_act = self.a1_a_space.idx_to_action[aidx1]
#     a2_act = self.a2_a_space.idx_to_action[aidx2]

#     list_p_next_env = transition_navi([], a1_pos, a2_pos, a1_act, a2_act, [],
#                                       self.goals, self.walls, [], self.x_grid,
#                                       self.y_grid)

#     list_next_p_state = []
#     map_next_state = {}
#     for p, a1_pos_n, a2_pos_n in list_p_next_env:
#       if a1_pos_n is None or a2_pos_n is None:
#         return np.array([[1.0, self.conv_dummy_state_to_idx("terminal")]])

#       sidx_n1 = self.a1_pos_space.state_to_idx[a1_pos_n]
#       sidx_n2 = self.a2_pos_space.state_to_idx[a2_pos_n]
#       sidx = self.conv_state_to_idx(tuple([sidx_n1, sidx_n2]))

#       map_next_state[sidx] = map_next_state.get(sidx, 0) + p

#     for key in map_next_state:
#       val = map_next_state[key]
#       list_next_p_state.append([val, key])

#     return np.array(list_next_p_state)

#   def reward(self, latent_idx: int, state_idx: int, action_idx: int) -> float:
#     if self.is_terminal(state_idx):
#       return 0

#     sidx1, sidx2 = self.conv_idx_to_state(state_idx)
#     a1_pos = self.a1_pos_space.idx_to_state[sidx1]
#     a2_pos = self.a2_pos_space.idx_to_state[sidx2]

#     aidx1, aidx2 = self.conv_idx_to_action(action_idx)
#     a1_act = self.a1_a_space.idx_to_action[aidx1]
#     a2_act = self.a2_a_space.idx_to_action[aidx2]

#     latent = self.latent_space.idx_to_state[latent_idx]

#     reward = -1
#     if latent[0] == "GH1" and latent[1] == 1:
#       if a1_pos == self.goals[0] and a2_pos == self.goals[2]:
#         return -300
#     if latent[0] == "GH1" and latent[1] == 2:
#       if a1_pos == self.goals[0] and a2_pos == self.goals[3]:
#         return -200
#     if latent[0] == "GH2" and latent[1] == 1:
#       if a1_pos == self.goals[1] and a2_pos == self.goals[2]:
#         return -200
#     if latent[0] == "GH2" and latent[1] == 2:
#       if a1_pos == self.goals[1] and a2_pos == self.goals[3]:
#         return -200

#     # if a1_pos in self.goals[0:2] and a2_pos in self.goals[2:4]:
#     #   return 1000

#     return reward