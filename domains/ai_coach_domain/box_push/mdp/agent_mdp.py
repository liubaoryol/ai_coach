import numpy as np
from ai_coach_domain.box_push import (BoxState, EventType, conv_box_state_2_idx,
                                      conv_box_idx_2_state, AGENT_ACTIONSPACE)
from ai_coach_domain.box_push.transition import (transition_alone_and_together,
                                                 transition_always_alone)
from ai_coach_domain.box_push.mdp import BoxPushMDP


class BoxPushAgentMDP(BoxPushMDP):
  def init_actionspace(self):
    self.dict_factored_actionspace = {}
    self.my_act_space = AGENT_ACTIONSPACE
    self.dict_factored_actionspace = {0: self.my_act_space}

  def transition_model(self, state_idx: int, action_idx: int) -> np.ndarray:
    if self.is_terminal(state_idx):
      return np.array([[1.0, state_idx]])

    box_states, my_pos, teammate_pos = self.conv_mdp_sidx_to_sim_states(
        state_idx)
    my_act, = self.conv_mdp_aidx_to_sim_actions(action_idx)

    # assume a2 has the same possible actions as a1
    list_p_next_env = []
    for teammate_act in self.my_act_space.actionspace:
      list_p_next_env = list_p_next_env + self.transition_fn(
          box_states, my_pos, teammate_pos, my_act, teammate_act, self.boxes,
          self.goals, self.walls, self.drops, self.x_grid, self.y_grid)

    list_next_p_state = []
    map_next_state = {}
    for p, box_states_list, my_pos_n, teammate_pos_n in list_p_next_env:
      sidx_n = self.conv_sim_states_to_mdp_sidx(
          [box_states_list, my_pos_n, teammate_pos_n])
      # assume a2 choose an action uniformly
      map_next_state[sidx_n] = (map_next_state.get(sidx_n, 0) +
                                p / self.num_actions)

    for key in map_next_state:
      val = map_next_state[key]
      list_next_p_state.append([val, key])

    return np.array(list_next_p_state)

  def reward(self, latent_idx: int, state_idx: int, action_idx: int) -> float:
    if self.is_terminal(state_idx):
      return 0

    len_s_space = len(self.dict_factored_statespace)
    state_vec = self.conv_idx_to_state(state_idx)

    my_pos = self.pos1_space.idx_to_state[state_vec[0]]
    # teammate_pos = self.pos2_space.idx_to_state[state_vec[1]]

    box_states = []
    for idx in range(2, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.dict_factored_statespace[idx].idx_to_state[box_sidx]
      box_states.append(box_state)

    my_act, = self.conv_mdp_aidx_to_sim_actions(action_idx)
    latent = self.latent_space.idx_to_state[latent_idx]

    holding_box = -1
    for idx, bstate in enumerate(box_states):
      if bstate[0] in [BoxState.WithMe, BoxState.WithBoth]:
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
        # elif bstate[0] == BoxState.WithTeammate:
        #   box_pos = teammate_pos
        elif bstate[0] == BoxState.OnDropLoc:
          box_pos = self.drops[bstate[1]]
        # elif bstate[0] == BoxState.OnGoalLoc:
        #   box_pos = self.goals[bstate[1]]

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
  from ai_coach_domain.box_push.maps import TEST_MAP3
  game_map = TEST_MAP3

  box_push_mdp = BoxPushAgentMDP_AlwaysAlone(**game_map)
  GAMMA = 0.95

  CHECK_VALIDITY = False
  VALUE_ITER = True

  if CHECK_VALIDITY:
    from ai_coach_core.utils.test_utils import check_transition_validity
    assert check_transition_validity(box_push_mdp)

  if VALUE_ITER:
    for idx in range(box_push_mdp.num_latents):
      pi, np_v_value, np_q_value = plan_lib.value_iteration(
          box_push_mdp.np_transition_model,
          box_push_mdp.np_reward_model[idx],
          discount_factor=GAMMA,
          max_iteration=500,
          epsilon=0.01)
      # print(box_push_mdp.latent_space.idx_to_state[idx])

      # for sidx in range(box_push_mdp.num_states):
      #   imp, itp, ibox = box_push_mdp.conv_idx_to_state(sidx)
      #   mp = box_push_mdp.my_pos_space.idx_to_state[imp]
      #   tp = box_push_mdp.teammate_pos_space.idx_to_state[itp]
      #   box = box_push_mdp.dict_factored_statespace[2].idx_to_state[ibox]
      #   for aidx in range(box_push_mdp.num_actions):
      #     act = box_push_mdp.my_act_space.idx_to_action[aidx]
      #     print(
      #         str(mp) + "; " + str(tp) + "; " + box[0].name + "; " +
      # act.name +
      #         "; " + "%f" % (np_q_value[sidx, aidx], ))

      cur_dir = os.path.dirname(__file__)
      str_v_val = os.path.join(cur_dir,
                               "data/box_push_np_v_value_tutorial.pickle")
      with open(str_v_val, "wb") as f:
        pickle.dump(np_v_value, f, pickle.HIGHEST_PROTOCOL)

      str_q_val = os.path.join(cur_dir,
                               "data/box_push_np_q_value_tutorial.pickle")
      with open(str_q_val, "wb") as f:
        pickle.dump(np_q_value, f, pickle.HIGHEST_PROTOCOL)
