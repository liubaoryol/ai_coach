import os
import abc
import pickle
from typing import Sequence, Tuple, Union
import numpy as np
from aic_core.models.mdp import MDP
from aic_core.utils.mdp_utils import StateSpace
from aic_core.RL.planning import value_iteration
from aic_domain.box_push_v2 import BoxState, AGENT_ACTIONSPACE
from aic_domain.box_push_v2 import (conv_box_state_2_idx, conv_box_idx_2_state)
from aic_domain.box_push_v2.transition import transition_mixed
from aic_domain.box_push_v2.maps import (MAP_MOVERS, MAP_CLEANUP_V2,
                                         MAP_CLEANUP_V3)
from aic_domain.box_push_v2.simulator import BoxPushSimulatorV2


class MMDP_BoxPush(MDP):

  def __init__(self, x_grid, y_grid, boxes, goals, walls, drops, box_types,
               a1_init, a2_init, **kwargs):

    self.box_types = box_types
    self.a1_init = a1_init
    self.a2_init = a2_init
    self.x_grid = x_grid
    self.y_grid = y_grid
    self.boxes = boxes
    self.goals = goals
    self.walls = walls
    self.drops = drops
    super().__init__(use_sparse=True)

  def is_terminal(self, state_idx):
    len_s_space = len(self.dict_factored_statespace)
    state_vec = self.conv_idx_to_state(state_idx)

    for idx in range(2, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.dict_factored_statespace[idx].idx_to_state[box_sidx]
      if box_state[0] != BoxState.OnGoalLoc:
        return False
    return True

  def init_statespace(self):
    '''
    To disable dummy states, set self.dummy_states = None
    '''

    self.dict_factored_statespace = {}

    list_grid = []
    for i in range(self.x_grid):
      for j in range(self.y_grid):
        state = (i, j)
        if state not in self.walls:
          list_grid.append(state)
    self.pos1_space = StateSpace(statespace=list_grid)
    self.pos2_space = StateSpace(statespace=list_grid)
    self.dict_factored_statespace = {0: self.pos1_space, 1: self.pos2_space}

    box_states = self.get_possible_box_states()

    for dummy_i in range(len(self.boxes)):
      self.dict_factored_statespace[dummy_i +
                                    2] = StateSpace(statespace=box_states)

    self.dummy_states = None

  def _transition_impl(self, box_states, a1_pos, a2_pos, a1_action, a2_action):
    return transition_mixed(box_states, a1_pos, a2_pos, a1_action, a2_action,
                            self.boxes, self.goals, self.walls, self.drops,
                            self.x_grid, self.y_grid, self.box_types,
                            self.a1_init, self.a2_init)

  def map_to_str(self):
    BASE36 = 36
    assert self.x_grid < BASE36 and self.y_grid < BASE36

    x_36 = np.base_repr(self.x_grid, BASE36)
    y_36 = np.base_repr(self.y_grid, BASE36)

    np_map = np.zeros((self.x_grid, self.y_grid), dtype=int)
    if self.boxes:
      np_map[tuple(zip(*self.boxes))] = 1
    if self.goals:
      np_map[tuple(zip(*self.goals))] = 2
    if self.walls:
      np_map[tuple(zip(*self.walls))] = 3
    if self.drops:
      np_map[tuple(zip(*self.drops))] = 4

    map_5 = "".join(np_map.reshape((-1, )).astype(str))
    map_int = int(map_5, base=5)
    map_36 = np.base_repr(map_int, base=BASE36)

    tup_map = (x_36, y_36, map_36)

    str_map = "%s_%s_%s" % tup_map
    boxes_w_type = list(zip(self.boxes, self.box_types))
    boxes_w_type.sort()
    _, sorted_box_types = zip(*boxes_w_type)
    np_btype = np.array(sorted_box_types) - 1
    str_btype = "".join(np_btype.astype(str))
    int_btype = int(str_btype, base=2)
    base36_btype = np.base_repr(int_btype, base=36)

    return str_map + "_" + base36_btype

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

    a1_hold = False
    a2_hold = False
    both_hold = False
    for idx, bstate in enumerate(box_states):
      if bstate[0] == BoxState.WithBoth:
        both_hold = True
      elif bstate[0] == BoxState.WithAgent1:
        a1_hold = True  # noqa: F841
      elif bstate[0] == BoxState.WithAgent2:
        a2_hold = True  # noqa: F841

    if both_hold and a1_pos != a2_pos:  # illegal state
      return []

    # if not (a1_hold or both_hold) and a1_pos in self.goals:  # illegal state
    #   return []

    # if not (a2_hold or both_hold) and a2_pos in self.goals:  # illegal state
    #   return []

    return super().legal_actions(state_idx)

  @abc.abstractmethod
  def get_possible_box_states(
      self) -> Sequence[Tuple[BoxState, Union[int, None]]]:
    raise NotImplementedError

  def init_actionspace(self):
    self.dict_factored_actionspace = {}
    self.a1_a_space = AGENT_ACTIONSPACE
    self.a2_a_space = AGENT_ACTIONSPACE
    self.dict_factored_actionspace = {0: self.a1_a_space, 1: self.a2_a_space}

  def transition_model(self, state_idx: int, action_idx: int) -> np.ndarray:
    if self.is_terminal(state_idx):
      return np.array([[1.0, state_idx]])

    box_states, a1_pos, a2_pos = self.conv_mdp_sidx_to_sim_states(state_idx)

    act1, act2 = self.conv_mdp_aidx_to_sim_actions(action_idx)

    list_p_next_env = self._transition_impl(box_states, a1_pos, a2_pos, act1,
                                            act2)
    list_next_p_state = []
    map_next_state = {}
    for p, box_states_list, a1_pos_n, a2_pos_n in list_p_next_env:
      sidx_n = self.conv_sim_states_to_mdp_sidx(
          [box_states_list, a1_pos_n, a2_pos_n])
      map_next_state[sidx_n] = map_next_state.get(sidx_n, 0) + p

    for key in map_next_state:
      val = map_next_state[key]
      list_next_p_state.append([val, key])

    return np.array(list_next_p_state)

  def conv_sim_states_to_mdp_sidx(self, tup_states):
    box_states, pos1, pos2 = tup_states
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

    return box_states, pos1, pos2

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

  def reward(self, state_idx: int, action_idx: int, *args, **kwargs) -> float:
    if self.is_terminal(state_idx):
      return 0

    return -1


class MMDP_Movers(MMDP_BoxPush):

  def get_possible_box_states(self):
    box_states = [(BoxState.Original, None), (BoxState.WithBoth, None)]
    num_drops = len(self.drops)
    num_goals = len(self.goals)
    if num_drops != 0:
      for idx in range(num_drops):
        box_states.append((BoxState.OnDropLoc, idx))
    for idx in range(num_goals):
      box_states.append((BoxState.OnGoalLoc, idx))
    return box_states


class MMDP_Cleanup(MMDP_BoxPush):

  def get_possible_box_states(self):
    box_states = [(BoxState.Original, None), (BoxState.WithAgent1, None),
                  (BoxState.WithAgent2, None)]
    num_drops = len(self.drops)
    num_goals = len(self.goals)
    if num_drops != 0:
      for idx in range(num_drops):
        box_states.append((BoxState.OnDropLoc, idx))
    for idx in range(num_goals):
      box_states.append((BoxState.OnGoalLoc, idx))
    return box_states


if __name__ == "__main__":
  domain_name = "movers"
  num_runs = 100

  DATA_DIR = os.path.join(os.path.dirname(__file__), "data/")

  mmdp = None
  if domain_name == "movers":
    game_map = MAP_MOVERS
    mmdp = MMDP_Movers(**game_map)
  elif domain_name == "cleanup_v2":
    game_map = MAP_CLEANUP_V2
    mmdp = MMDP_Cleanup(**game_map)
  elif domain_name == "cleanup_v3":
    game_map = MAP_CLEANUP_V3
    mmdp = MMDP_Cleanup(**game_map)

  if mmdp is not None:

    # mmdp transition
    mmdp_transition_file_name = domain_name + "_mmdp_transition"
    pickle_mmdp_trans = os.path.join(DATA_DIR,
                                     mmdp_transition_file_name + ".pickle")

    if os.path.exists(pickle_mmdp_trans):
      with open(pickle_mmdp_trans, 'rb') as handle:
        np_transition_model = pickle.load(handle)
    else:
      np_transition_model = mmdp.np_transition_model
      with open(pickle_mmdp_trans, 'wb') as handle:
        pickle.dump(np_transition_model,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    # mmdp reward
    mmdp_reward_file_name = domain_name + "_mmdp_reward"
    pickle_mmdp_reward = os.path.join(DATA_DIR,
                                      mmdp_reward_file_name + ".pickle")

    if os.path.exists(pickle_mmdp_reward):
      with open(pickle_mmdp_reward, 'rb') as handle:
        np_reward_model = pickle.load(handle)
    else:
      np_reward_model = mmdp.np_reward_model
      with open(pickle_mmdp_reward, 'wb') as handle:
        pickle.dump(np_reward_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    policy, v_value, q_value = value_iteration(np_transition_model,
                                               np_reward_model,
                                               discount_factor=0.99,
                                               max_iteration=500,
                                               epsilon=0.001)

    game = BoxPushSimulatorV2(0)
    game.init_game(**game_map)
    game.set_autonomous_agent()
    INCREASE_STEP = True
    list_score = []
    list_steps = []
    for _ in range(num_runs):
      game.reset_game()
      while not game.is_finished():
        tup_state = game.get_state_for_each_agent(0)
        sidx = mmdp.conv_sim_states_to_mdp_sidx(tup_state)
        aidx = policy[sidx]
        act1, act2 = mmdp.conv_mdp_aidx_to_sim_actions(aidx)

        game.event_input(0, act1, None)
        game.event_input(1, act2, None)
        if INCREASE_STEP:
          game.current_step += 1
          if game.is_finished():
            break

        map_agent_2_action = game.get_joint_action()
        game.take_a_step(map_agent_2_action)
      list_score.append(game.get_score())
      list_steps.append(game.get_current_step())

    np_score = np.array(list_score)
    np_steps = np.array(list_steps)

    print(np_score.mean(), np_score.std(), np_steps.mean(), np_steps.std())
