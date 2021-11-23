import numpy as np
from ai_coach_core.utils.mdp_utils import ActionSpace
from ai_coach_domain.box_push import BoxState, EventType
from ai_coach_domain.box_push.helper import (transition_alone_and_together,
                                             transition_always_together,
                                             transition_always_alone)
from ai_coach_domain.box_push.mdp import BoxPushMDP


class BoxPushTeamMDP(BoxPushMDP):
  def init_actionspace(self):
    self.dict_factored_actionspace = {}
    action_states = [EventType(idx) for idx in range(6)]
    self.a1_a_space = ActionSpace(actionspace=action_states)
    self.a2_a_space = ActionSpace(actionspace=action_states)
    self.dict_factored_actionspace = {0: self.a1_a_space, 1: self.a2_a_space}

  def transition_model(self, state_idx: int, action_idx: int) -> np.ndarray:
    if self.is_terminal(state_idx):
      return np.array([[1.0, state_idx]])

    a1_pos, a2_pos, box_states = self.conv_mdp_sidx_to_sim_states(state_idx)

    act1, act2 = self.conv_mdp_aidx_to_sim_actions(action_idx)

    list_p_next_env = self.transition_fn(box_states, a1_pos, a2_pos, act1, act2,
                                         self.boxes, self.goals, self.walls,
                                         self.drops, self.x_grid, self.y_grid)
    list_next_p_state = []
    map_next_state = {}
    for p, box_states_list, a1_pos_n, a2_pos_n in list_p_next_env:
      sidx_n = self.conv_sim_states_to_mdp_sidx(a1_pos_n, a2_pos_n,
                                                box_states_list)
      map_next_state[sidx_n] = map_next_state.get(sidx_n, 0) + p

    for key in map_next_state:
      val = map_next_state[key]
      list_next_p_state.append([val, key])

    return np.array(list_next_p_state)


class BoxPushTeamMDP_AloneOrTogether(BoxPushTeamMDP):
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

  def reward(self, latent_idx: int, state_idx: int, action_idx: int) -> float:
    if self.is_terminal(state_idx):
      return 0

    len_s_space = len(self.dict_factored_statespace)
    state_vec = self.conv_idx_to_state(state_idx)

    a1_pos = self.pos1_space.idx_to_state[state_vec[0]]
    a2_pos = self.pos2_space.idx_to_state[state_vec[1]]

    box_states = []
    for idx in range(2, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.dict_factored_statespace[idx].idx_to_state[box_sidx]
      box_states.append(box_state)

    act1, act2 = self.conv_mdp_aidx_to_sim_actions(action_idx)

    # a1 drops a box
    a1_drop = False
    if a1_pos in self.goals and act1 == EventType.UNHOLD:
      for bstate in box_states:
        if bstate[0] == BoxState.WithAgent1:
          a1_drop = True

    a2_drop = False
    if a2_pos in self.goals and act2 == EventType.UNHOLD:
      for bstate in box_states:
        if bstate[0] == BoxState.WithAgent2:
          a2_drop = True

    both_drop = False
    if (a1_pos in self.goals and a1_pos == a2_pos and act1 == EventType.UNHOLD
        and act2 == EventType.UNHOLD):
      for bstate in box_states:
        if bstate[0] == BoxState.WithBoth:
          both_drop = True

    reward = -1

    if a1_drop or a2_drop or both_drop:
      reward += 10

    return reward


class BoxPushTeamMDP_AlwaysTogether(BoxPushTeamMDP):
  def __init__(self, x_grid, y_grid, boxes, goals, walls, drops, **kwargs):
    super().__init__(x_grid, y_grid, boxes, goals, walls, drops,
                     transition_always_together)

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

  def reward(self, latent_idx: int, state_idx: int, action_idx: int) -> float:
    if self.is_terminal(state_idx):
      return 0

    len_s_space = len(self.dict_factored_statespace)
    state_vec = self.conv_idx_to_state(state_idx)

    a1_pos = self.pos1_space.idx_to_state[state_vec[0]]
    a2_pos = self.pos2_space.idx_to_state[state_vec[1]]

    box_states = []
    for idx in range(2, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.dict_factored_statespace[idx].idx_to_state[box_sidx]
      box_states.append(box_state)

    act1, act2 = self.conv_mdp_aidx_to_sim_actions(action_idx)
    latent = self.latent_space.idx_to_state[latent_idx]

    holding_box = -1
    for idx, bstate in enumerate(box_states):
      if bstate[0] == BoxState.WithBoth:
        holding_box = idx

    panelty = -1

    if latent[0] == "pickup":
      # if they are already holding a box,
      # set every action but stay as illegal
      if holding_box >= 0:
        if act1 == EventType.STAY and act2 == EventType.STAY:
          return 0
        else:
          return -np.inf
      # if they are not holding a box,
      # give a reward when pickup the target box
      else:
        if (a1_pos == a2_pos and a1_pos == self.boxes[latent[1]]
            and act1 == EventType.HOLD and act2 == EventType.HOLD):
          return 100

        # if get close to the target, don't deduct
        box_pos = self.boxes[latent[1]]
        dist1 = abs(a1_pos[0] - box_pos[0]) + abs(a1_pos[1] - box_pos[1])
        dist2 = abs(a2_pos[0] - box_pos[0]) + abs(a2_pos[1] - box_pos[1])
        panelty += 1 / (dist1 + dist2 + 1)
    elif holding_box >= 0:  # not "pickup" and holding a box --> drop the box
      desired_loc = None
      if latent[0] == "origin":
        desired_loc = self.boxes[holding_box]
      elif latent[0] == "drop":
        desired_loc = self.drops[latent[1]]
      else:  # latent[0] == "goal"
        desired_loc = self.goals[latent[1]]

      if (a1_pos == a2_pos and a1_pos == desired_loc
          and act1 == EventType.UNHOLD and act2 == EventType.UNHOLD):
        return 100
    else:  # "drop the box" but not having a box (illegal state)
      if act1 == EventType.STAY and act2 == EventType.STAY:
        return 0
      else:
        return -np.inf

    return panelty


class BoxPushTeamMDP_AlwaysAlone(BoxPushTeamMDP):
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


if __name__ == "__main__":
  import os
  import pickle
  import ai_coach_core.RL.planning as plan_lib
  from ai_coach_domain.box_push.maps import TUTORIAL_MAP

  game_map = TUTORIAL_MAP

  box_push_mdp = BoxPushTeamMDP_AlwaysTogether(**game_map)
  GAMMA = 0.95

  CHECK_VALIDITY = False
  VALUE_ITER = True
  QLEARN = False

  if CHECK_VALIDITY:
    from ai_coach_core.utils.test_utils import check_transition_validity
    assert check_transition_validity(box_push_mdp)

  cur_dir = os.path.dirname(__file__)
  if VALUE_ITER:
    for idx in range(box_push_mdp.num_latents):
      pi, np_v_value, np_q_value = plan_lib.value_iteration(
          box_push_mdp.np_transition_model,
          box_push_mdp.np_reward_model[idx],
          discount_factor=GAMMA,
          max_iteration=500,
          epsilon=0.01)

      str_v_val = os.path.join(
          cur_dir, "data/box_push_team_np_v_value_tutorial_%d.pickle" % (idx, ))
      with open(str_v_val, "wb") as f:
        pickle.dump(np_v_value, f, pickle.HIGHEST_PROTOCOL)

      str_q_val = os.path.join(
          cur_dir, "data/box_push_team_np_q_value_tutorial_%d.pickle" % (idx, ))
      with open(str_q_val, "wb") as f:
        pickle.dump(np_q_value, f, pickle.HIGHEST_PROTOCOL)

  if QLEARN:
    from tqdm import tqdm
    import ai_coach_core.RL.qlearning as qlearn_lib
    NUM_TRAIN = 1000
    ALPHA = 0.1
    box_push_qlearn = qlearn_lib.QLearningSoftmax(
        beta=1,
        num_states=box_push_mdp.num_states,
        num_actions=box_push_mdp.num_actions,
        num_training=NUM_TRAIN,
        alpha=ALPHA,
        gamma=GAMMA)

    for i in range(NUM_TRAIN):
      box_states = [0] * len(game_map["boxes"])
      sidx = box_push_mdp.conv_sim_states_to_mdp_sidx(game_map["a1_init"],
                                                      game_map["a2_init"],
                                                      box_states)
      pbar = tqdm()
      box_push_qlearn.start_episode()
      count = 0
      while True:
        if box_push_mdp.is_terminal(sidx):
          break
        aidx = box_push_qlearn.get_action(sidx)
        sidx_n = box_push_mdp.transition(sidx, aidx)
        reward = box_push_mdp.reward(0, sidx, aidx)
        box_push_qlearn.observe_transition(sidx, aidx, sidx_n, reward)
        sidx = sidx_n
        count += 1
        if count % 10000 == 0:
          pbar.update(10000)
      pbar.close()

      box_push_qlearn.stop_episode()
      print("episodes: %d, count: %d, reward: %.2f" %
            (box_push_qlearn.get_episodes_sofar(), count,
             box_push_qlearn.episode_rewards))

    str_qlearn_file = os.path.join(
        cur_dir, "data/box_push_team_qlearn_np_q_val_tutorial.pickle")
    with open(str_qlearn_file, "wb") as f:
      pickle.dump(box_push_qlearn.np_q_values, f, pickle.HIGHEST_PROTOCOL)
