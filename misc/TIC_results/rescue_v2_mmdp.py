import os
import pickle
import itertools
from typing import Sequence, Tuple, Mapping
import numpy as np
from ai_coach_core.models.mdp import MDP
from ai_coach_core.utils.mdp_utils import StateSpace
from ai_coach_core.RL.planning import value_iteration

from ai_coach_domain.rescue_v2 import (Route, E_EventType, E_Type, Location,
                                       Work, Place, T_Connections,
                                       AGENT_ACTIONSPACE, is_work_done)
from ai_coach_domain.rescue_v2.transition import transition, find_location_index
from ai_coach_domain.rescue_v2.maps import MAP_RESCUE
from ai_coach_domain.rescue_v2.simulator import RescueSimulatorV2


class MMDP_RescueV2(MDP):

  def __init__(self, routes: Sequence[Route], places: Sequence[Place],
               connections: Mapping[int, T_Connections],
               work_locations: Sequence[Location], work_info: Sequence[Work],
               **kwarg):

    self.routes = routes
    self.places = places
    self.connections = connections
    self.work_locations = work_locations
    self.work_info = work_info
    super().__init__(use_sparse=True)

  def _transition_impl(self, work_states, a1_pos, a2_pos, a3_pos, a1_action,
                       a2_action, a3_action):
    return transition(work_states, a1_pos, a2_pos, a3_pos, a1_action, a2_action,
                      a3_action, self.routes, self.connections,
                      self.work_locations, self.work_info)

  def map_to_str(self):
    BASE36 = 36
    num_place = len(self.places)
    num_route = len(self.routes)

    str_map = np.base_repr(num_place, BASE36) + np.base_repr(num_route, BASE36)
    str_place = ","
    for place_id in range(num_place):
      str_node = ""
      for connection in self.connections[place_id]:
        if connection[0] == E_Type.Route:
          str_node += np.base_repr(connection[1], BASE36)
        else:
          str_node += np.base_repr(connection[1] + num_route, BASE36)
      str_place += str_node + "_"
    str_map += str_place[:-1]

    str_route = ","
    for route in self.routes:
      str_route += (np.base_repr(route.start, BASE36) +
                    np.base_repr(route.end, BASE36) +
                    np.base_repr(route.length, BASE36) + "_")

    str_map += str_route[:-1]

    return str_map

  def init_statespace(self):
    '''
    To disable dummy states, set self.dummy_states = None
    '''

    self.dict_factored_statespace = {}

    list_locations = []
    for place_id in self.connections:
      if len(self.connections[place_id]) > 0:
        list_locations.append(Location(E_Type.Place, place_id))

    for route_id, route in enumerate(self.routes):
      for idx in range(route.length):
        list_locations.append(Location(E_Type.Route, route_id, idx))

    self.pos1_space = StateSpace(statespace=list_locations)
    self.pos2_space = StateSpace(statespace=list_locations)
    self.pos3_space = StateSpace(statespace=list_locations)

    num_works = len(self.work_locations)
    self.work_states_space = StateSpace(
        statespace=list(itertools.product([0, 1], repeat=num_works)))

    self.dict_factored_statespace = {
        0: self.pos1_space,
        1: self.pos2_space,
        2: self.pos3_space,
        3: self.work_states_space
    }

    self.dummy_states = None

  def is_terminal(self, state_idx):
    factored_state_idx = self.conv_idx_to_state(state_idx)
    work_states = self.work_states_space.idx_to_state[factored_state_idx[-1]]

    for idx in range(len(work_states)):
      if not is_work_done(idx, work_states, self.work_info[idx].coupled_works):
        return False

    return True

  def conv_sim_states_to_mdp_sidx(self, tup_states) -> int:
    work_states, pos1, pos2, pos3 = tup_states

    pos1_idx = self.pos1_space.state_to_idx[pos1]
    pos2_idx = self.pos2_space.state_to_idx[pos2]
    pos3_idx = self.pos3_space.state_to_idx[pos3]
    work_states_idx = self.work_states_space.state_to_idx[tuple(work_states)]

    return self.conv_state_to_idx(
        (pos1_idx, pos2_idx, pos3_idx, work_states_idx))

  def conv_mdp_sidx_to_sim_states(
      self, state_idx) -> Tuple[Sequence, Location, Location, Location]:
    state_vec = self.conv_idx_to_state(state_idx)
    pos1 = self.pos1_space.idx_to_state[state_vec[0]]
    pos2 = self.pos2_space.idx_to_state[state_vec[1]]
    pos3 = self.pos3_space.idx_to_state[state_vec[2]]
    work_states = self.work_states_space.idx_to_state[state_vec[3]]

    return work_states, pos1, pos2, pos3

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

  def init_actionspace(self):
    self.a1_a_space = AGENT_ACTIONSPACE
    self.a2_a_space = AGENT_ACTIONSPACE
    self.a3_a_space = AGENT_ACTIONSPACE
    self.dict_factored_actionspace = {
        0: self.a1_a_space,
        1: self.a2_a_space,
        2: self.a3_a_space
    }

  def transition_model(self, state_idx: int, action_idx: int) -> np.ndarray:
    if self.is_terminal(state_idx):
      return np.array([[1.0, state_idx]])

    work_states, a1_pos, a2_pos, a3_pos = self.conv_mdp_sidx_to_sim_states(
        state_idx)

    act1, act2, act3 = self.conv_mdp_aidx_to_sim_actions(action_idx)

    list_p_next_env = self._transition_impl(work_states, a1_pos, a2_pos, a3_pos,
                                            act1, act2, act3)
    list_next_p_state = []
    map_next_state = {}
    for p, work_states_n, a1_pos_n, a2_pos_n, a3_pos_n in list_p_next_env:
      sidx_n = self.conv_sim_states_to_mdp_sidx(
          [work_states_n, a1_pos_n, a2_pos_n, a3_pos_n])
      map_next_state[sidx_n] = map_next_state.get(sidx_n, 0) + p

    for key in map_next_state:
      val = map_next_state[key]
      list_next_p_state.append([val, key])

    return np.array(list_next_p_state)

  def legal_actions(self, state_idx):
    if self.is_terminal(state_idx):
      return []

    work_states, pos1, pos2, pos3 = self.conv_mdp_sidx_to_sim_states(state_idx)

    if pos1.type == E_Type.Route:
      a1_actions = [E_EventType(idx) for idx in range(2)]
      a1_actions.append(E_EventType.Stay)
      a1_actions.append(E_EventType.Rescue)
    else:
      a1_actions = [
          E_EventType(idx) for idx in range(len(self.connections[pos1.id]))
      ]
      a1_actions.append(E_EventType.Stay)
      a1_actions.append(E_EventType.Rescue)

    if pos2.type == E_Type.Route:
      a2_actions = [E_EventType(idx) for idx in range(2)]
      a2_actions.append(E_EventType.Stay)
      a2_actions.append(E_EventType.Rescue)
    else:
      a2_actions = [
          E_EventType(idx) for idx in range(len(self.connections[pos2.id]))
      ]
      a2_actions.append(E_EventType.Stay)
      a2_actions.append(E_EventType.Rescue)

    if pos3.type == E_Type.Route:
      a3_actions = [E_EventType(idx) for idx in range(2)]
      a3_actions.append(E_EventType.Stay)
      a3_actions.append(E_EventType.Rescue)
    else:
      a3_actions = [
          E_EventType(idx) for idx in range(len(self.connections[pos3.id]))
      ]
      a3_actions.append(E_EventType.Stay)
      a3_actions.append(E_EventType.Rescue)

    list_actions = []
    for tuple_actions in itertools.product(a1_actions, a2_actions, a3_actions):
      list_actions.append(self.conv_sim_actions_to_mdp_aidx(tuple_actions))

    return list_actions

  def reward(self, state_idx: int, action_idx: int) -> float:
    if self.is_terminal(state_idx):
      return 0

    work_states, pos1, pos2, pos3 = self.conv_mdp_sidx_to_sim_states(state_idx)
    act1, act2, act3 = self.conv_mdp_aidx_to_sim_actions(action_idx)

    reward = 0
    for idx in range(len(work_states)):
      # if work_states[idx] != 0:
      if not is_work_done(idx, work_states, self.work_info[idx].coupled_works):
        workload = self.work_info[idx].workload

        work1 = find_location_index(self.work_locations, pos1)
        work2 = find_location_index(self.work_locations, pos2)
        work3 = find_location_index(self.work_locations, pos3)
        workforce = 0
        if work1 == idx and act1 == E_EventType.Rescue:
          workforce += 1
        if work2 == idx and act2 == E_EventType.Rescue:
          workforce += 1
        if work3 == idx and act3 == E_EventType.Rescue:
          workforce += 1

        if workload <= workforce:
          place_id = self.work_info[idx].rescue_place
          reward += self.places[place_id].helps

    return reward


if __name__ == "__main__":
  domain_name = "rescue_3"
  num_runs = 100

  DATA_DIR = os.path.join(os.path.dirname(__file__), "data/")

  mmdp = None
  if domain_name == "rescue_3":
    game_map = MAP_RESCUE
    mmdp = MMDP_RescueV2(**game_map)

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

    game = RescueSimulatorV2()
    game.init_game(**game_map)
    game.set_autonomous_agent()

    policy, v_value, q_value = value_iteration(np_transition_model,
                                               np_reward_model,
                                               discount_factor=0.98,
                                               max_iteration=game.max_steps,
                                               epsilon=0.001)

    INCREASE_STEP = True
    list_score = []
    list_steps = []
    for _ in range(num_runs):
      game.reset_game()
      while not game.is_finished():
        tup_state = game.get_state_for_each_agent(0)
        sidx = mmdp.conv_sim_states_to_mdp_sidx(tup_state)
        aidx = policy[sidx]
        act1, act2, act3 = mmdp.conv_mdp_aidx_to_sim_actions(aidx)

        game.event_input(0, act1, None)
        game.event_input(1, act2, None)
        game.event_input(2, act3, None)
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
