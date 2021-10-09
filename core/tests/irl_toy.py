import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from ai_coach_core.model_inference.IRL.maxent_irl import (CMaxEntIRL,
                                                          compute_relative_freq,
                                                          cal_reward_error,
                                                          cal_policy_error)
from ai_coach_core.models.mdp import MDP
from ai_coach_core.RL.planning import value_iteration
from ai_coach_core.utils.mdp_utils import StateSpace, ActionSpace
from ai_coach_core.utils.exceptions import InvalidTransitionError

DANGER_GRIDS = [(1, 3), (4, 1), (4, 2)]
TERMINAL_STATE = -1
PICK = (9, 9)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data/irl_toy_trajectories/")


def get_neighborhood(stt, set_state):
  set_neighbor = set()
  x, y, h = stt
  for i in range(-1, 2):
    for j in range(-1, 2):
      if i == 0 and j == 0:
        continue
      state = (x + i, y + j, h)
      if state in set_state:
        set_neighbor.add(state)
  return set_neighbor


class ToyMDP(MDP):
  def __init__(self):
    super().__init__(use_sparse=False)

  def init_statespace(self):
    '''
    To disable dummy states, set self.dummy_states = None
    '''

    self.dict_factored_statespace = {}

    set_state = set()
    for i in range(5):
      for j in range(5):
        state = (i, j)
        if state not in [(2, 1), (2, 2), (2, 3)]:
          set_state.add((state[0], state[1], 0))
          set_state.add((state[0], state[1], 1))

    self.s_space = StateSpace(statespace=set_state)
    self.dict_factored_statespace = {0: self.s_space}
    self.dummy_states = StateSpace(statespace=[TERMINAL_STATE])

  def init_actionspace(self):
    '''
        We use a factored representation for the joint action:
        aAne : anesthesiologist action
        aSur : surgeon action
        '''

    set_actions = set()
    set_actions.add((-1, 0))
    set_actions.add((1, 0))
    set_actions.add((0, -1))
    set_actions.add((0, 1))
    set_actions.add((0, 0))
    set_actions.add((-1, -1))
    set_actions.add((-1, 1))
    set_actions.add((1, -1))
    set_actions.add((1, 1))
    set_actions.add(PICK)

    self.dict_factored_actionspace = {}
    self.a_space = ActionSpace(actionspace=set_actions)
    self.dict_factored_actionspace = {0: self.a_space}

  def legal_actions(self, state_idx):
    if self.is_dummy_state(state_idx):
      return []

    sid, = self.conv_idx_to_state(state_idx)
    # print(sid)
    state = self.s_space.idx_to_state[sid]

    def action_to_idx(act):
      return self.np_action_to_idx[int(self.a_space.action_to_idx[act])]

    if state == (0, 0, 1):
      return [action_to_idx((0, 0))]
    elif (state[0], state[1]) in DANGER_GRIDS:
      return [action_to_idx((0, 0))]
    else:
      legal_acts = []
      for act in self.a_space.actionspace:
        if act == PICK:
          if state == (3, 3, 0):
            legal_acts.append(action_to_idx(act))
        else:
          s_n = (state[0] + act[0], state[1] + act[1], state[2])
          if s_n not in self.s_space.statespace:
            continue
          legal_acts.append(action_to_idx(act))
      return legal_acts

  def transition_model(self, state_idx: int, action_idx: int) -> np.ndarray:
    '''
        Returns a np array with two columns and at least one row.
        The first column corresponds to the probability for the next state.
        The second column corresponds to the index of the next state.
        '''

    # unpack the input
    s_i, = self.np_idx_to_state[state_idx]
    a_i, = self.np_idx_to_action[action_idx]
    stt = self.s_space.idx_to_state[s_i]
    act = self.a_space.idx_to_action[a_i]

    def state_to_idx(sttx):
      return self.np_state_to_idx[int(self.s_space.state_to_idx[sttx])]

    terminal_sid = self.conv_dummy_state_to_idx(TERMINAL_STATE)

    # obtain next probability values for each factor
    list_next_p_state = []
    # np.array([[1., sInit]])
    if stt == (0, 0, 1):
      if act == (0, 0):
        list_next_p_state.append((1., terminal_sid))
      else:
        InvalidTransitionError
    elif (stt[0], stt[1]) in DANGER_GRIDS:
      if act == (0, 0):
        list_next_p_state.append((1., terminal_sid))
      else:
        InvalidTransitionError
    elif act == PICK:
      if stt == (3, 3, 0):
        stt_n_idx = state_to_idx((3, 3, 1))
        list_next_p_state.append((1., stt_n_idx))
      else:
        InvalidTransitionError
    else:
      stt_n = (stt[0] + act[0], stt[1] + act[1], stt[2])
      l1norm = abs(act[0]) + abs(act[1])
      if stt_n not in self.s_space.statespace:
        # pass
        InvalidTransitionError
      elif l1norm == 0:
        stt_n_idx = state_to_idx(stt_n)
        list_next_p_state.append((1., stt_n_idx))
      elif l1norm == 1:
        stt_n_idx = state_to_idx(stt_n)
        list_next_p_state.append((0.9, stt_n_idx))
        set_nbr = get_neighborhood(stt, self.s_space.statespace)
        len_nbr = len(set_nbr) - 1
        for s_nbr in set_nbr:
          if s_nbr != stt_n:
            s_nbr_idx = state_to_idx(s_nbr)
            list_next_p_state.append((0.1 / len_nbr, s_nbr_idx))
      elif l1norm == 2:
        stt_n_idx = state_to_idx(stt_n)
        list_next_p_state.append((0.7, stt_n_idx))
        set_nbr = get_neighborhood(stt, self.s_space.statespace)
        len_nbr = len(set_nbr) - 1
        for s_nbr in set_nbr:
          if s_nbr != stt_n:
            s_nbr_idx = state_to_idx(s_nbr)
            list_next_p_state.append((0.3 / len_nbr, s_nbr_idx))
      else:
        InvalidTransitionError
    if len(list_next_p_state) == 0:
      return []

    np_next_p_state = np.array(list_next_p_state)
    # print(np_next_p_state)
    # print(np.sum(np_next_p_state[:,0]))
    # assert( np.sum(np_next_p_state[:,0]) == 1. )

    return np_next_p_state

  def reward(self, state_idx: int, action_idx: int, *args, **kwargs) -> float:
    if self.is_terminal(state_idx):
      return 0

    sid, = self.np_idx_to_state[state_idx]
    aid, = self.np_idx_to_action[action_idx]
    state = self.s_space.idx_to_state[sid]
    action = self.a_space.idx_to_action[aid]
    x, y, h = state
    if (x, y) in DANGER_GRIDS:  # terminal state
      return -100

    if h == 0:
      if (x, y) == (3, 3):
        if action == PICK:
          return 100
        else:
          return -1
      elif x == 3 or x == 4:
        return -10
      else:
        return -1
    else:
      if (x, y) == (0, 0):  # terminal state
        return 100
      elif x == 0 or x == 1:
        return -10
      else:
        return -1

  def is_terminal(self, state_idx):
    return self.is_dummy_state(state_idx)


def get_stochastic_policy(mdp, deter_pi):
  sto_pi = np.zeros((mdp.num_states, mdp.num_actions))
  for s_idx in range(mdp.num_states):
    # sto_pi[s_idx] =
    deter_act_idx = deter_pi[s_idx]

    possible_acts = mdp.legal_actions(s_idx)
    if len(possible_acts) == 1:
      sto_pi[s_idx, deter_act_idx] = 1.0
    else:
      for act in possible_acts:
        if act == deter_act_idx:
          sto_pi[s_idx, act] = 0.95
        else:
          sto_pi[s_idx, act] = 0.05 / (len(possible_acts) - 1)

  return sto_pi


def gen_trajectory(mdp, stochastic_pi):

  init_state_idx = mdp.s_space.state_to_idx[(0, 0, 0)]

  trajectory = []
  cur_s_idx = init_state_idx
  while True:
    if mdp.is_terminal(cur_s_idx):
      break

    act = random.choices(range(stochastic_pi.shape[1]),
                         weights=stochastic_pi[cur_s_idx],
                         k=1)[0]
    trajectory.append((cur_s_idx, act))

    # transition to next state
    np_next_dist = mdp.transition_model(cur_s_idx, act)

    cur_s_idx = int(
        random.choices(np_next_dist[:, 1], weights=np_next_dist[:, 0], k=1)[0])

  return trajectory


def save_trajectory(trajectory, file_name):
  dir_path = os.path.dirname(file_name)
  if dir_path != '' and not os.path.exists(dir_path):
    os.makedirs(dir_path)

  with open(file_name, 'w', newline='') as txtfile:
    # sequence
    txtfile.write('# state action sequence\n')
    for idx in range(len(trajectory)):
      state, action = trajectory[idx]
      txtfile.write('%d, %d' % (state, action))
      txtfile.write('\n')


def read_trajectory(file_name):
  traj = []
  with open(file_name, newline='') as txtfile:
    lines = txtfile.readlines()
    for i_r in range(1, len(lines)):
      line = lines[i_r]
      row_elem = [int(elem) for elem in line.rstrip().split(", ")]
      state = row_elem[0]
      action = row_elem[1]
      traj.append((state, action))
  return traj


def feature_extract(mdp, s_idx, a_idx):
  if mdp.is_terminal(s_idx):
    return {}

  sid, = mdp.np_idx_to_state[s_idx]
  aid, = mdp.np_idx_to_action[a_idx]
  state = mdp.s_space.idx_to_state[sid]
  action = mdp.a_space.idx_to_action[aid]

  x, y, h = state

  def manhattan_dist(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

  DIST_ORIG = "dist_orig"
  DIST_TOOL = "dist_tool"
  TOOL_STAT = "tool_stat"
  GRID_TYPE = "grid_type"
  GRID_ZONE = "grid_zone"
  TOOL_PICK = "pick_tool"

  feature = {}
  # feature[BAIS] = 1
  feature[DIST_ORIG] = manhattan_dist(x, y, 0, 0)
  feature[DIST_TOOL] = manhattan_dist(x, y, 3, 3) if h == 0 else 0
  feature[TOOL_STAT] = h
  if (x, y) in DANGER_GRIDS:
    feature[GRID_TYPE] = 2
  elif (x, y) in [(2, 1), (2, 2), (2, 3)]:
    feature[GRID_TYPE] = 1
  else:
    feature[GRID_TYPE] = 0

  if x == 0 or x == 1:
    feature[GRID_ZONE] = 0
  elif x == 2:
    feature[GRID_ZONE] = 1
  else:
    feature[GRID_ZONE] = 2

  feature[TOOL_PICK] = 1 if state == (3, 3, 0) and action == PICK else 0

  np_feature = np.zeros(len(feature))
  for idx, key in enumerate(feature):
    np_feature[idx] = feature[key]

  return np_feature


def feature_extract_full_state(mdp, s_idx, a_idx):
  np_feature = np.zeros(mdp.num_states)
  np_feature[s_idx] = 1
  return np_feature


def feature_extract_updated(state, action):
  x, y, h = state

  def manhattan_dist(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

  DIST_D1 = "dist_d1"
  DIST_D2 = "dist_d2"
  DIST_D3 = "dist_d3"

  DIST_ORIG = "dist_orig"
  DIST_TOOL = "dist_tool"
  TOOL_STAT = "tool_stat"
  GRID_TYPE = "grid_type"
  GRID_ZONE = "grid_zone"
  TOOL_PICK = "pick_tool"
  BAIS = "bais"

  feature = {}
  feature[BAIS] = 1
  feature[DIST_ORIG] = manhattan_dist(x, y, 0, 0)
  feature[DIST_TOOL] = manhattan_dist(x, y, 3, 3) if h == 0 else 0
  feature[DIST_D1] = manhattan_dist(x, y, 1, 3)
  feature[DIST_D2] = manhattan_dist(x, y, 4, 1)
  feature[DIST_D3] = manhattan_dist(x, y, 4, 2)
  feature[TOOL_STAT] = h
  if (x, y) in DANGER_GRIDS:
    feature[GRID_TYPE] = 2
  elif (x, y) in [(2, 1), (2, 2), (2, 3)]:
    feature[GRID_TYPE] = 1
  else:
    feature[GRID_TYPE] = 0

  if x == 0 or x == 1:
    feature[GRID_ZONE] = 0
  elif x == 2:
    feature[GRID_ZONE] = 1
  else:
    feature[GRID_ZONE] = 2

  feature[TOOL_PICK] = 1 if state == (3, 3, 0) and action == PICK else 0

  return feature


if __name__ == "__main__":
  # data_dir = './toy_traj/'
  # file_prefix = 'toy_'
  toy_mdp = ToyMDP()

  num_agents = 1
  num_ostates = toy_mdp.num_states
  num_actions = toy_mdp.num_actions

  print(num_ostates)
  print(num_actions)

  gamma = 0.9

  pi, np_v_value, np_q_value = value_iteration(toy_mdp.np_transition_model,
                                               toy_mdp.np_reward_model,
                                               discount_factor=gamma,
                                               max_iteration=100,
                                               epsilon=0.001)

  sto_pi = get_stochastic_policy(toy_mdp, pi)

  GENERATE_DATA = False

  if GENERATE_DATA:
    for dummy in range(100):
      sample = gen_trajectory(toy_mdp, sto_pi)
      file_path = os.path.join(DATA_DIR, str(dummy) + '.txt')
      save_trajectory(sample, file_path)
      # print(sample)
    print("data generated")

  trajectories = []
  len_sum = 0
  file_names = glob.glob(os.path.join(DATA_DIR, '*.txt'))
  for file_nm in file_names:
    traj = read_trajectory(file_nm)
    len_sum += len(traj)
    trajectories.append(traj)

  avg_len = int(len_sum / len(trajectories)) + 1
  # print(avg_len)

  init_prop = np.zeros((toy_mdp.num_states))
  sid = toy_mdp.s_space.state_to_idx[(0, 0, 0)]
  s_idx = toy_mdp.np_state_to_idx[sid]
  init_prop[s_idx] = 1

  irl = CMaxEntIRL(trajectories,
                   toy_mdp,
                   feature_extractor=feature_extract,
                   max_value_iter=100,
                   initial_prop=init_prop)

  rel_freq = compute_relative_freq(num_ostates, trajectories)
  reward_error = []
  policy_error = []

  def compute_errors(reward_fn, policy_fn):
    reward_error.append(cal_reward_error(toy_mdp, reward_fn))
    policy_error.append(cal_policy_error(rel_freq, toy_mdp, policy_fn, sto_pi))

  irl.do_inverseRL(epsilon=0.001,
                   n_max_run=500,
                   callback_reward_pi=compute_errors)

  kl_irl = cal_policy_error(rel_freq, toy_mdp, irl.policy, sto_pi)
  print(kl_irl)
  from ai_coach_core.model_inference.behavior_cloning import behavior_cloning
  pi_bc = behavior_cloning(trajectories, num_ostates, num_actions)
  kl_bc = cal_policy_error(rel_freq, toy_mdp, lambda s, a: pi_bc[s, a], sto_pi)
  print(kl_bc)

  f = plt.figure(figsize=(10, 5))
  ax1 = f.add_subplot(121)
  ax2 = f.add_subplot(122)
  ax1.plot(reward_error)
  ax1.set_ylabel('reward_error')
  # plt.show()

  ax2.plot(policy_error)
  ax2.set_ylabel('policy_error')
  plt.show()
