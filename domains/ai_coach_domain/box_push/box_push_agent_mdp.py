import numpy as np
from ai_coach_core.models.latent_mdp import LatentMDP
from ai_coach_core.utils.mdp_utils import StateSpace, ActionSpace
from ai_coach_domain.box_push import (BoxState, EventType, conv_box_state_2_idx,
                                      conv_box_idx_2_state)
from ai_coach_domain.box_push.box_push_helper import (
    transition_alone_and_together, transition_always_alone)


class BoxPushAgentMDP(LatentMDP):
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
    self.my_pos_space = StateSpace(statespace=set_grid)
    self.teammate_pos_space = StateSpace(statespace=set_grid)
    self.dict_factored_statespace = {
        0: self.my_pos_space,
        1: self.teammate_pos_space
    }

    box_states = self.get_possible_box_states()

    for dummy_i in range(len(self.boxes)):
      self.dict_factored_statespace[dummy_i +
                                    2] = StateSpace(statespace=box_states)

    self.dummy_states = None

  def init_actionspace(self):
    self.dict_factored_actionspace = {}
    action_states = [EventType(idx) for idx in range(6)]
    self.my_act_space = ActionSpace(actionspace=action_states)
    self.dict_factored_actionspace = {0: self.my_act_space}

  def init_latentspace(self):
    latent_states = []
    for idx in range(len(self.boxes)):
      latent_states.append(("pickup", idx))

    latent_states.append(("origin", None))  # drop at its original position
    for idx in range(len(self.drops)):
      latent_states.append(("drop", idx))
    for idx in range(len(self.goals)):
      latent_states.append(("goal", idx))

    self.latent_space = StateSpace(latent_states)

  def get_possible_box_states(self):
    raise NotImplementedError
    # box_states = [(BoxState(idx), None) for idx in range(4)]
    # num_drops = len(self.drops)
    # num_goals = len(self.goals)
    # if num_drops != 0:
    #   for idx in range(num_drops):
    #     box_states.append((BoxState.OnDropLoc, idx))
    # for idx in range(num_goals):
    #   box_states.append((BoxState.OnGoalLoc, idx))
    # return box_states

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

    my_pos = self.my_pos_space.idx_to_state[state_vec[0]]
    teammate_pos = self.teammate_pos_space.idx_to_state[state_vec[1]]

    box_states = []
    for idx in range(2, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.dict_factored_statespace[idx].idx_to_state[box_sidx]
      box_states.append(box_state)

    holding_box = -1
    for idx, bstate in enumerate(box_states):
      if bstate[0] == BoxState.WithBoth:
        holding_box = idx

    if holding_box >= 0 and my_pos != teammate_pos:  # illegal state
      return []

    return super().legal_actions(state_idx)

  def transition_model(self, state_idx: int, action_idx: int) -> np.ndarray:
    if self.is_terminal(state_idx):
      return np.array([[1.0, state_idx]])

    my_pos, teammate_pos, box_states = self.conv_mdp_sidx_to_sim_states(
        state_idx)
    my_act = self.conv_mdp_aidx_to_sim_action(action_idx)

    # assume a2 has the same possible actions as a1
    list_p_next_env = []
    for teammate_act in self.my_act_space.actionspace:
      list_p_next_env = list_p_next_env + self.transition_fn(
          box_states, my_pos, teammate_pos, my_act, teammate_act, self.boxes,
          self.goals, self.walls, self.drops, self.x_grid, self.y_grid)

    list_next_p_state = []
    map_next_state = {}
    for p, box_states_list, my_pos_n, teammate_pos_n in list_p_next_env:
      sidx_n = self.conv_sim_states_to_mdp_sidx(my_pos_n, teammate_pos_n,
                                                box_states_list)
      map_next_state[sidx_n] = (map_next_state.get(sidx_n, 0) +
                                p / self.num_actions)

    for key in map_next_state:
      val = map_next_state[key]
      list_next_p_state.append([val, key])

    return np.array(list_next_p_state)

  def conv_sim_states_to_mdp_sidx(self, my_pos, teammate_pos, box_states):
    len_s_space = len(self.dict_factored_statespace)
    my_next_idx = self.my_pos_space.state_to_idx[my_pos]
    teammate_next_idx = self.teammate_pos_space.state_to_idx[teammate_pos]
    list_states = [int(my_next_idx), int(teammate_next_idx)]
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

    my_pos = self.my_pos_space.idx_to_state[state_vec[0]]
    teammate_pos = self.teammate_pos_space.idx_to_state[state_vec[1]]

    box_states = []
    for idx in range(2, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.dict_factored_statespace[idx].idx_to_state[box_sidx]
      box_states.append(conv_box_state_2_idx(box_state, len(self.drops)))

    return my_pos, teammate_pos, box_states

  def conv_mdp_aidx_to_sim_action(self, action_idx):
    aidx, = self.conv_idx_to_action(action_idx)
    return self.my_act_space.idx_to_action[aidx]

  def conv_sim_action_to_mdp_aidx(self, action):
    return self.my_act_space.action_to_idx[action]

  def reward(self, latent_idx: int, state_idx: int, action_idx: int) -> float:
    if self.is_terminal(state_idx):
      return 0

    len_s_space = len(self.dict_factored_statespace)
    state_vec = self.conv_idx_to_state(state_idx)

    my_pos = self.my_pos_space.idx_to_state[state_vec[0]]
    teammate_pos = self.teammate_pos_space.idx_to_state[state_vec[1]]

    box_states = []
    for idx in range(2, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.dict_factored_statespace[idx].idx_to_state[box_sidx]
      box_states.append(box_state)

    my_act = self.conv_mdp_aidx_to_sim_action(action_idx)
    latent = self.latent_space.idx_to_state[latent_idx]

    holding_box = -1
    for idx, bstate in enumerate(box_states):
      if bstate[0] in [BoxState.WithAgent1, BoxState.WithBoth]:
        holding_box = idx

    panelty = -1

    if latent[0] == "pickup":
      # if already holding a box, set every action but stay as illegal
      if holding_box >= 0:
        if my_act == EventType.STAY:
          return 0
        else:
          return -np.inf
      else:
        idx = latent[1]
        bstate = box_states[idx]
        box_pos = None
        if bstate[0] == BoxState.Original:
          box_pos = self.boxes[latent[1]]
        elif bstate[0] == BoxState.WithTeammate:
          box_pos = teammate_pos
        elif bstate[0] == BoxState.OnDropLoc:
          box_pos = self.drops[bstate[1]]
        elif bstate[0] == BoxState.OnGoalLoc:
          box_pos = self.goals[bstate[1]]

        if my_pos == box_pos and my_act == EventType.HOLD:
          return 100
    elif holding_box >= 0:  # not "pickup" and holding a box --> drop the box
      desired_loc = None
      if latent[0] == "origin":
        desired_loc = self.boxes[holding_box]
      elif latent[0] == "drop":
        desired_loc = self.drops[latent[1]]
      else:  # latent[0] == "goal"
        desired_loc = self.goals[latent[1]]

      if my_pos == desired_loc and my_act == EventType.UNHOLD:
        return 100
    else:  # "drop the box" but not having a box (illegal state)
      if my_act == EventType.STAY:
        return 0
      else:
        return -np.inf

    return panelty


class BoxPushAgentMDP_AloneOrTogether(BoxPushAgentMDP):
  def __init__(self, x_grid, y_grid, boxes, goals, walls, drops, **kwargs):
    super().__init__(x_grid, y_grid, boxes, goals, walls, drops,
                     transition_alone_and_together)

  def get_possible_box_states(self):
    box_states = [(BoxState(idx), None) for idx in range(4)]
    num_drops = len(self.drops)
    num_goals = len(self.goals)
    if num_drops != 0:
      for idx in range(num_drops):
        box_states.append((BoxState.OnDropLoc, idx))
    for idx in range(num_goals):
      box_states.append((BoxState.OnGoalLoc, idx))
    return box_states


class BoxPushAgentMDP_AlwaysAlone(BoxPushAgentMDP):
  def __init__(self, x_grid, y_grid, boxes, goals, walls, drops, **kwargs):
    super().__init__(x_grid, y_grid, boxes, goals, walls, drops,
                     transition_always_alone)

  def get_possible_box_states(self):
    box_states = [(BoxState(idx), None) for idx in range(3)]
    num_drops = len(self.drops)
    num_goals = len(self.goals)
    if num_drops != 0:
      for idx in range(num_drops):
        box_states.append((BoxState.OnDropLoc, idx))
    for idx in range(num_goals):
      box_states.append((BoxState.OnGoalLoc, idx))
    return box_states


def get_agent_switched_boxstates(box_states, num_drops, num_goals):
  switched_box_states = list(box_states)
  for idx, bidx in enumerate(box_states):
    bstate = conv_box_idx_2_state(bidx, num_drops, num_goals)
    if bstate[0] == BoxState.WithAgent1:
      switched_box_states[idx] = conv_box_state_2_idx(
          (BoxState.WithAgent2, bstate[1]), num_drops)
    elif bstate[0] == BoxState.WithAgent2:
      switched_box_states[idx] = conv_box_state_2_idx(
          (BoxState.WithAgent1, bstate[1]), num_drops)

  return switched_box_states


if __name__ == "__main__":
  import os
  import pickle
  import ai_coach_core.RL.planning as plan_lib
  from ai_coach_domain.box_push.box_push_maps import TUTORIAL_MAP
  game_map = TUTORIAL_MAP

  box_push_mdp = BoxPushAgentMDP_AlwaysAlone(**game_map)
  GAMMA = 0.95

  CHECK_VALIDITY = True
  VALUE_ITER = False

  if CHECK_VALIDITY:
    from ai_coach_core.utils.test_utils import check_transition_validity
    assert check_transition_validity(box_push_mdp)

  if VALUE_ITER:
    pi, np_v_value, np_q_value = plan_lib.value_iteration(
        box_push_mdp.np_transition_model,
        box_push_mdp.np_reward_model[0],
        discount_factor=GAMMA,
        max_iteration=500,
        epsilon=0.01)

    cur_dir = os.path.dirname(__file__)
    str_v_val = os.path.join(cur_dir,
                             "data/box_push_np_v_value_tutorial.pickle")
    with open(str_v_val, "wb") as f:
      pickle.dump(np_v_value, f, pickle.HIGHEST_PROTOCOL)

    str_q_val = os.path.join(cur_dir,
                             "data/box_push_np_q_value_tutorial.pickle")
    with open(str_q_val, "wb") as f:
      pickle.dump(np_q_value, f, pickle.HIGHEST_PROTOCOL)
