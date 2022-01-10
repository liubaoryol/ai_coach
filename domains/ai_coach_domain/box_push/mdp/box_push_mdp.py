from typing import Sequence, Tuple, Union
import abc
from ai_coach_core.utils.mdp_utils import StateSpace
from ai_coach_core.models.mdp import LatentMDP
from ai_coach_domain.box_push import (BoxState, conv_box_state_2_idx,
                                      conv_box_idx_2_state)
from ai_coach_domain.box_push.helper import get_possible_latent_states


class BoxPushMDP(LatentMDP):
  def __init__(self, x_grid, y_grid, boxes, goals, walls, drops, cb_transition,
               **kwargs):
    self.x_grid = x_grid
    self.y_grid = y_grid
    self.boxes = boxes
    self.goals = goals
    self.walls = walls
    self.drops = drops
    self.transition_fn = cb_transition
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
    self.pos1_space = StateSpace(statespace=set_grid)
    self.pos2_space = StateSpace(statespace=set_grid)
    self.dict_factored_statespace = {0: self.pos1_space, 1: self.pos2_space}

    box_states = self.get_possible_box_states()

    for dummy_i in range(len(self.boxes)):
      self.dict_factored_statespace[dummy_i +
                                    2] = StateSpace(statespace=box_states)

    self.dummy_states = None

  def init_latentspace(self):
    latent_states = get_possible_latent_states(len(self.boxes), len(self.drops),
                                               len(self.goals))
    self.latent_space = StateSpace(latent_states)

  def is_terminal(self, state_idx):
    len_s_space = len(self.dict_factored_statespace)
    state_vec = self.conv_idx_to_state(state_idx)

    for idx in range(2, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.dict_factored_statespace[idx].idx_to_state[box_sidx]
      if box_state[0] != BoxState.OnGoalLoc:
        return False
    return True

  def legal_actions(self, state_idx):
    if self.is_terminal(state_idx):
      return []

    len_s_space = len(self.dict_factored_statespace)
    state_vec = self.conv_idx_to_state(state_idx)

    a1_pos = self.pos1_space.idx_to_state[state_vec[0]]
    a2_pos = self.pos2_space.idx_to_state[state_vec[1]]

    box_states = []
    for idx in range(2, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.dict_factored_statespace[idx].idx_to_state[box_sidx]
      box_states.append(box_state)

    holding_box = -1
    for idx, bstate in enumerate(box_states):
      if bstate[0] == BoxState.WithBoth:
        holding_box = idx

    if holding_box >= 0 and a1_pos != a2_pos:  # illegal state
      return []

    return super().legal_actions(state_idx)

  @abc.abstractmethod
  def get_possible_box_states(
      self) -> Sequence[Tuple[BoxState, Union[int, None]]]:
    raise NotImplementedError

  def conv_sim_states_to_mdp_sidx(self, pos1, pos2, box_states):
    len_s_space = len(self.dict_factored_statespace)
    pos1_idx = self.pos1_space.state_to_idx[pos1]
    pos2_idx = self.pos2_space.state_to_idx[pos2]
    list_states = [int(pos1_idx), int(pos2_idx)]
    for idx in range(2, len_s_space):
      box_sidx_n = box_states[idx - 2]
      box_state_n = conv_box_idx_2_state(box_sidx_n, len(self.drops),
                                         len(self.goals))
      box_sidx = self.dict_factored_statespace[idx].state_to_idx[box_state_n]
      list_states.append(int(box_sidx))

    return self.conv_state_to_idx(tuple(list_states))

  def conv_mdp_sidx_to_sim_states(self, state_idx):
    len_s_space = len(self.dict_factored_statespace)
    state_vec = self.conv_idx_to_state(state_idx)

    pos1 = self.pos1_space.idx_to_state[state_vec[0]]
    pos2 = self.pos2_space.idx_to_state[state_vec[1]]

    box_states = []
    for idx in range(2, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.dict_factored_statespace[idx].idx_to_state[box_sidx]
      box_states.append(conv_box_state_2_idx(box_state, len(self.drops)))

    return pos1, pos2, box_states

  def conv_mdp_aidx_to_sim_actions(self, action_idx):
    vector_aidx = self.conv_idx_to_action(action_idx)
    list_actions = []
    for idx, aidx in enumerate(vector_aidx):
      list_actions.append(
          self.dict_factored_actionspace[idx].idx_to_action[aidx])
    return tuple(list_actions)

  def conv_sim_actions_to_mdp_aidx(self, tuple_actions):
    list_aidx = []
    for idx, act in enumerate(tuple_actions):
      list_aidx.append(self.dict_factored_actionspace[idx].action_to_idx[act])

    return self.np_action_to_idx[tuple(list_aidx)]