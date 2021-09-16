import numpy as np
import models.mdp as mdp_lib
from utils.mdp_utils import StateSpace, ActionSpace
from ai_coach_domain.box_push.box_push_helper import (BoxState, EventType,
                                                      transition,
                                                      conv_box_state_2_idx,
                                                      conv_box_idx_2_state)


class BoxPushMDP(mdp_lib.MDP):
  def __init__(self, x_grid, y_grid, boxes, goals, walls, drops):
    self.x_grid = x_grid
    self.y_grid = y_grid
    self.boxes = boxes
    self.goals = goals
    self.walls = walls
    self.drops = drops
    super().__init__()

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

    box_states = [(BoxState(idx), None) for idx in range(4)]
    num_drops = len(self.drops)
    num_goals = len(self.goals)
    if num_drops != 0:
      for idx in range(num_drops):
        box_states.append((BoxState.OnDropLoc, idx))
    for idx in range(num_goals):
      box_states.append((BoxState.OnGoalLoc, idx))

    self.a1_pos_space = StateSpace(statespace=set_grid)
    self.a2_pos_space = StateSpace(statespace=set_grid)
    self.dict_factored_statespace = {0: self.a1_pos_space, 1: self.a2_pos_space}
    for dummy_i in range(len(self.boxes)):
      self.dict_factored_statespace[dummy_i +
                                    2] = StateSpace(statespace=box_states)

    self.dummy_states = None

  def init_actionspace(self):

    self.dict_factored_actionspace = {}
    action_states = [EventType(dummy_i) for dummy_i in range(6)]
    self.a1_a_space = ActionSpace(actionspace=action_states)
    self.a2_a_space = ActionSpace(actionspace=action_states)
    self.dict_factored_actionspace = {0: self.a1_a_space, 1: self.a2_a_space}

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

    return super().legal_actions(state_idx)

  def transition_model(self, state_idx: int, action_idx: int) -> np.ndarray:
    if self.is_terminal(state_idx):
      return np.array([[1.0, state_idx]])

    len_s_space = len(self.dict_factored_statespace)
    state_vec = self.conv_idx_to_state(state_idx)

    a1_pos = self.a1_pos_space.idx_to_state[state_vec[0]]
    a2_pos = self.a2_pos_space.idx_to_state[state_vec[1]]

    box_states = []
    for idx in range(2, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.dict_factored_statespace[idx].idx_to_state[box_sidx]
      box_states.append(conv_box_state_2_idx(box_state, len(self.drops)))

    a1, a2 = self.conv_idx_to_action(action_idx)
    act1 = self.a1_a_space.idx_to_action[a1]
    act2 = self.a2_a_space.idx_to_action[a2]

    list_p_next_env = transition(box_states, a1_pos, a2_pos, act1, act2,
                                 self.boxes, self.goals, self.walls, self.drops,
                                 self.x_grid, self.y_grid)
    list_next_p_state = []
    map_next_state = {}
    for p, box_states_list, a1_pos_n, a2_pos_n in list_p_next_env:
      a1_next_idx = self.a1_pos_space.state_to_idx[a1_pos_n]
      a2_next_idx = self.a2_pos_space.state_to_idx[a2_pos_n]
      list_states = [int(a1_next_idx), int(a2_next_idx)]
      for idx in range(2, len_s_space):
        box_sidx_n = box_states_list[idx - 2]
        box_state_n = conv_box_idx_2_state(box_sidx_n, len(self.drops),
                                           len(self.goals))
        box_sidx = self.dict_factored_statespace[idx].state_to_idx[box_state_n]
        list_states.append(int(box_sidx))

      sidx_n = self.conv_state_to_idx(tuple(list_states))
      map_next_state[sidx_n] = map_next_state.get(sidx_n, 0) + p

    for key in map_next_state:
      val = map_next_state[key]
      list_next_p_state.append([val, key])

    return np.array(list_next_p_state)

  def reward(self, state_idx: int, action_idx: int, *args, **kwargs) -> float:
    if self.is_terminal(state_idx):
      return 0

    len_s_space = len(self.dict_factored_statespace)
    state_vec = self.conv_idx_to_state(state_idx)

    a1_pos = self.a1_pos_space.idx_to_state[state_vec[0]]
    a2_pos = self.a2_pos_space.idx_to_state[state_vec[0]]

    box_states = []
    for idx in range(2, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.dict_factored_statespace[idx].idx_to_state[box_sidx]
      box_states.append(conv_box_state_2_idx(box_state, len(self.drops)))

    a1, a2 = self.conv_idx_to_action(action_idx)
    act1 = self.a1_a_space.idx_to_action[a1]
    act2 = self.a2_a_space.idx_to_action[a2]

    # a1 drops a box
    a1_drop = False
    if a1_pos in self.goals and act1 == EventType.HOLD:
      for bstate in box_states:
        if bstate[0] == BoxState.WithAgent1:
          a1_drop = True

    a2_drop = False
    if a2_pos in self.goals and act2 == EventType.HOLD:
      for bstate in box_states:
        if bstate[0] == BoxState.WithAgent2:
          a2_drop = True

    both_drop = False
    if (a1_pos in self.goals and act1 == EventType.HOLD
        and act2 == EventType.HOLD):
      for bstate in box_state:
        if bstate[0] == BoxState.WithBoth:
          both_drop = True

    reward = -1

    if a1_drop or a2_drop or both_drop:
      reward += 10

    return reward


if __name__ == "__main__":
  from tqdm import tqdm
  import pickle
  import RL.qlearning as qlearn_lib

  GRID_X = 6
  GRID_Y = 6
  game_map = {
      "boxes": [(0, 1), (3, 1)],
      "goals": [(GRID_X - 1, GRID_Y - 1)],
      "walls": [(GRID_X - 2, GRID_Y - i - 1) for i in range(3)],
      "wall_dir": [0 for dummy_i in range(3)],
      "drops": []
  }

  box_push_mdp = BoxPushMDP(GRID_X, GRID_Y, game_map["boxes"],
                            game_map["goals"], game_map["walls"],
                            game_map["drops"])

  NUM_TRAIN = 10000
  ALPHA = 0.5
  GAMMA = 0.95
  box_push_ql = qlearn_lib.QLearningGreedy(epsilon=0.1,
                                           num_states=box_push_mdp.num_states,
                                           num_actions=box_push_mdp.num_actions,
                                           num_training=NUM_TRAIN,
                                           alpha=ALPHA,
                                           gamma=GAMMA)

  for i in range(NUM_TRAIN):
    a1idx = box_push_mdp.a1_pos_space.state_to_idx[(GRID_X - 1, 0)]
    a2idx = box_push_mdp.a2_pos_space.state_to_idx[(0, GRID_X - 1)]
    states = [int(a1idx), int(a2idx)]
    for idx in range(len(game_map["boxes"])):
      bidx = int(box_push_mdp.dict_factored_statespace[idx + 2].state_to_idx[(
          BoxState.Original, None)])
      states.append(bidx)

    pbar = tqdm()
    sidx = box_push_mdp.np_state_to_idx[tuple(states)]
    box_push_ql.start_episode()
    count = 0
    while True:
      if box_push_mdp.is_terminal(sidx):
        break
      aidx = box_push_ql.get_action(sidx)
      sidx_n = box_push_mdp.transition(sidx, aidx)
      reward = box_push_mdp.reward(sidx, aidx)
      box_push_ql.observe_transition(sidx, aidx, sidx_n, reward)
      count += 1
      if count % 10000 == 0:
        pbar.update(10000)
    pbar.close()

    box_push_ql.stop_episode()
    print(
        "episodes: %d, count: %d, reward: %.2f" %
        (box_push_ql.get_episodes_sofar(), count, box_push_ql.episode_rewards))

  with open("box_push_6by6.pickle", "wb") as f:
    pickle.dump(box_push_ql.np_q_values, f, pickle.HIGHEST_PROTOCOL)
