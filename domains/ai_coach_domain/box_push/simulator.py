from typing import Hashable, Mapping, Tuple, Sequence, Callable
import os
import numpy as np
from ai_coach_domain.simulator import Simulator
from ai_coach_domain.box_push.helper import (EventType,
                                             transition_alone_and_together,
                                             transition_always_together,
                                             transition_always_alone)

Coord = Tuple[int, int]


class BoxPushSimulator(Simulator):
  AGENT1 = 0
  AGENT2 = 1

  def __init__(self, id: Hashable, cb_transition: Callable) -> None:
    super().__init__(id)
    self.cb_get_A1_action = None
    self.cb_get_A2_action = None
    self.cb_get_A1_mental_state = None
    self.cb_get_A2_mental_state = None
    self.cb_get_init_mental_state = None
    self.transition_fn = cb_transition

  def init_game(self,
                x_grid: Coord,
                y_grid: Coord,
                a1_init: Coord,
                a2_init: Coord,
                boxes: Sequence[Coord] = [],
                goals: Sequence[Coord] = [],
                walls: Sequence[Coord] = [],
                drops: Sequence[Coord] = [],
                **kwargs):
    self.x_grid = x_grid
    self.y_grid = y_grid
    self.a1_init = a1_init
    self.a2_init = a2_init
    self.boxes = boxes
    self.goals = goals
    self.walls = walls
    self.drops = drops

    self.reset_game()

  def set_autonomous_agent(self,
                           cb_get_A1_action=None,
                           cb_get_A2_action=None,
                           cb_get_A1_mental_state=None,
                           cb_get_A2_mental_state=None,
                           cb_get_init_mental_state=None):
    self.cb_get_A1_action = cb_get_A1_action
    self.cb_get_A2_action = cb_get_A2_action
    self.cb_get_A1_mental_state = cb_get_A1_mental_state
    self.cb_get_A2_mental_state = cb_get_A2_mental_state
    self.cb_get_init_mental_state = cb_get_init_mental_state
    if self.cb_get_init_mental_state:
      self.a1_latent, self.a2_latent = self.cb_get_init_mental_state()

  def reset_game(self):
    super().reset_game()

    self.a1_pos = self.a1_init
    self.a2_pos = self.a2_init
    # starts with their original locations
    self.box_states = [0] * len(self.boxes)
    self.a1_action_event = None
    self.a2_action_event = None

    self.a1_latent = None
    self.a2_latent = None
    if self.cb_get_init_mental_state:
      self.a1_latent, self.a2_latent = self.cb_get_init_mental_state()
    self.changed_state = []

  def take_a_step(self, map_agent_2_action: Mapping[Hashable,
                                                    Hashable]) -> None:
    a1_action = None
    if BoxPushSimulator.AGENT1 in map_agent_2_action:
      a1_action = map_agent_2_action[BoxPushSimulator.AGENT1]

    a2_action = None
    if BoxPushSimulator.AGENT2 in map_agent_2_action:
      a2_action = map_agent_2_action[BoxPushSimulator.AGENT2]

    if a1_action is None and a2_action is None:
      return

    if a1_action is None:
      a1_action = EventType.STAY

    if a2_action is None:
      a2_action = EventType.STAY

    a1_lat = tuple(self.a1_latent) if self.a1_latent is not None else ("NA", 0)
    a2_lat = tuple(self.a2_latent) if self.a2_latent is not None else ("NA", 0)

    cur_state = [tuple(self.box_states), tuple(self.a1_pos), tuple(self.a2_pos)]

    state = [
        self.current_step, cur_state[0], cur_state[1], cur_state[2],
        a1_action.value, a2_action.value, a1_lat, a2_lat
    ]
    self.history.append(state)

    self.__transition(a1_action, a2_action)
    super().take_a_step(map_agent_2_action)
    self.changed_state.append("current_step")

    if self.cb_get_A1_mental_state:
      next_state = [self.box_states, self.a1_pos, self.a2_pos]
      self.a1_latent = self.cb_get_A1_mental_state(cur_state, a1_action,
                                                   a2_action, self.a1_latent,
                                                   next_state)
    if self.cb_get_A2_mental_state:
      next_state = [self.box_states, self.a1_pos, self.a2_pos]
      self.a2_latent = self.cb_get_A2_mental_state(cur_state, a1_action,
                                                   a2_action, self.a2_latent,
                                                   next_state)

  def __transition(self, a1_action, a2_action):
    list_next_env = self.transition_fn(self.box_states, self.a1_pos,
                                       self.a2_pos, a1_action, a2_action,
                                       self.boxes, self.goals, self.walls,
                                       self.drops, self.x_grid, self.y_grid)

    list_prop = []
    for item in list_next_env:
      list_prop.append(item[0])

    idx_c = np.random.choice(range(len(list_next_env)), 1, p=list_prop)[0]
    _, box_states, a1_pos, a2_pos = list_next_env[idx_c]
    self.a1_pos = a1_pos
    self.a2_pos = a2_pos
    self.box_states = box_states

    self.changed_state.append("a1_pos")
    self.changed_state.append("a2_pos")
    self.changed_state.append("box_states")

  def get_num_box_state(self):
    '''
    initial location (0), with agent1 (1),
    with agent2 (2), with both agents (3),
    drop locations (4 ~ 3 + len(self.drops)),
    goals (4 + len(self.drops) ~ 3 + len(self.drops) + len(self.goals))
    '''
    return 4 + len(self.drops) + len(self.goals)

  def get_num_agents(self):
    return 2

  def get_num_latents(self):
    '''
    in the order of each box, each drop, and each goal
    '''
    return len(self.boxes) + len(self.drops) + len(self.goals)

  def get_latent_idx(self, type, idx):
    if type == "box":
      return idx
    elif type == "drop":
      return len(self.boxes) + idx
    elif type == "goal":
      return len(self.boxes) + len(self.drops) + idx
    else:
      return -1

  def event_input(self, agent: Hashable, event_type: Hashable, value):
    if (agent is None) or (event_type is None):
      return

    if agent == BoxPushSimulator.AGENT1:
      if event_type != EventType.SET_LATENT:
        self.a1_action_event = event_type
      else:
        self.a1_latent = value
        self.changed_state.append("a1_latent")
    elif agent == BoxPushSimulator.AGENT2:
      if event_type != EventType.SET_LATENT:
        self.a2_action_event = event_type
      else:
        self.a2_latent = value
        self.changed_state.append("a2_latent")

  def get_joint_action(self) -> Mapping[Hashable, Hashable]:
    # TODO: need to think about logic.. for now let's assume at least one agent
    # is always a human
    # if self.a1_action_event is None and self.a2_action_event is None:
    #   return {}

    map_a2a = {}
    if self.cb_get_A1_action:
      map_a2a[BoxPushSimulator.AGENT1] = self.cb_get_A1_action(
          **self.get_env_info())
    else:
      map_a2a[BoxPushSimulator.AGENT1] = self.a1_action_event

    if self.cb_get_A2_action:
      map_a2a[BoxPushSimulator.AGENT2] = self.cb_get_A2_action(
          **self.get_env_info())
    else:
      map_a2a[BoxPushSimulator.AGENT2] = self.a2_action_event

    self.a1_action_event = None
    self.a2_action_event = None

    return map_a2a

  def get_env_info(self):
    return {
        "x_grid": self.x_grid,
        "y_grid": self.y_grid,
        "box_states": self.box_states,
        "boxes": self.boxes,
        "goals": self.goals,
        "drops": self.drops,
        "walls": self.walls,
        "a1_pos": self.a1_pos,
        "a2_pos": self.a2_pos,
        "a1_latent": self.a1_latent,
        "a2_latent": self.a2_latent,
        "current_step": self.current_step
    }

  def get_changed_objects(self):
    dict_changed_obj = {}
    for state in self.changed_state:
      dict_changed_obj[state] = getattr(self, state)
    return dict_changed_obj
    # return {
    #     "box_states": self.box_states,
    #     "a1_pos": self.a1_pos,
    #     "a2_pos": self.a2_pos,
    #     "a1_latent": self.a1_latent,
    #     "a2_latent": self.a2_latent
    # }

  def save_history(self, file_name, header):
    dir_path = os.path.dirname(file_name)
    if dir_path != '' and not os.path.exists(dir_path):
      os.makedirs(dir_path)

    with open(file_name, 'w', newline='') as txtfile:
      # sequence
      txtfile.write(header)
      txtfile.write('\n')
      txtfile.write('# cur_step, box_state, a1_pos, a2_pos, ' +
                    'a1_act, a2_act, a1_latent, a2_latent\n')

      for step, bstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat in self.history:
        txtfile.write('%d; ' % (step, ))  # cur step
        # box states
        for idx in range(len(bstt) - 1):
          txtfile.write('%d, ' % (bstt[idx], ))
        txtfile.write('%d; ' % (bstt[-1], ))

        txtfile.write('%d, %d; ' % a1pos)
        txtfile.write('%d, %d; ' % a2pos)

        txtfile.write('%d; %d; ' % (a1act, a2act))

        txtfile.write('%s, %d; ' % a1lat)
        txtfile.write('%s, %d; ' % a2lat)
        txtfile.write('\n')

      # last state
      txtfile.write('%d; ' % (self.current_step, ))  # cur step
      # box states
      for idx in range(len(self.box_states) - 1):
        txtfile.write('%d, ' % (bstt[idx], ))
      txtfile.write('%d; ' % (bstt[-1], ))

      txtfile.write('%d, %d; ' % self.a1_pos)
      txtfile.write('%d, %d; ' % self.a2_pos)
      txtfile.write('\n')

  def is_finished(self) -> bool:
    if super().is_finished():
      return True

    for state in self.box_states:
      if state is not (self.get_num_box_state() - 1):
        return False

    return True

  @classmethod
  def read_file(cls, file_name):
    traj = []
    with open(file_name, newline='') as txtfile:
      lines = txtfile.readlines()
      i_start = 0
      for i_r, row in enumerate(lines):
        if row == ('# cur_step, box_state, a1_pos, a2_pos, ' +
                   'a1_act, a2_act, a1_latent, a2_latent\n'):
          i_start = i_r
          break

      for i_r in range(i_start + 1, len(lines)):
        line = lines[i_r]
        states = line.rstrip()[:-1].split("; ")
        if len(states) < 8:
          break
        step, bstate, a1pos, a2pos, a1act, a2act, a1lat, a2lat = states
        box_state = tuple([int(elem) for elem in bstate.split(", ")])
        a1_pos = tuple([int(elem) for elem in a1pos.split(", ")])
        a2_pos = tuple([int(elem) for elem in a2pos.split(", ")])
        a1_act = EventType(int(a1act))
        a2_act = EventType(int(a2act))
        a1lat_tmp = a1lat.split(", ")
        a1_lat = (a1lat_tmp[0], int(a1lat_tmp[1]))
        a2lat_tmp = a2lat.split(", ")
        a2_lat = (a2lat_tmp[0], int(a2lat_tmp[1]))
        traj.append([box_state, a1_pos, a2_pos, a1_act, a2_act, a1_lat, a2_lat])

    return traj


class BoxPushSimulator_AloneOrTogether(BoxPushSimulator):
  def __init__(self, id: Hashable) -> None:
    super().__init__(id, transition_alone_and_together)


class BoxPushSimulator_AlwaysTogether(BoxPushSimulator):
  def __init__(self, id: Hashable) -> None:
    super().__init__(id, transition_always_together)


class BoxPushSimulator_AlwaysAlone(BoxPushSimulator):
  def __init__(self, id: Hashable) -> None:
    super().__init__(id, transition_always_alone)


if __name__ == "__main__":
  test = False
  exp1 = False
  if test:
    from ai_coach_domain.box_push.maps import TEST_MAP
    from ai_coach_domain.box_push.policy import get_simple_action

    sim = BoxPushSimulator_AlwaysAlone(0)
    sim.init_game(**TEST_MAP)

    sim.set_autonomous_agent(
        cb_get_A1_action=lambda **kwargs: get_simple_action(
            BoxPushSimulator.AGENT1, **kwargs),
        cb_get_A2_action=lambda **kwargs: get_simple_action(
            BoxPushSimulator.AGENT2, **kwargs))
    sim.run_simulation(1, "test_file_name", "text")

  if exp1:
    from ai_coach_domain.box_push.maps import EXP1_MAP
    from ai_coach_domain.box_push.policy import get_exp1_action
    from ai_coach_domain.box_push.team_mdp import BoxPushTeamMDP_AlwaysTogether

    sim = BoxPushSimulator_AlwaysTogether(0)
    sim.init_game(**EXP1_MAP)
    mdp = BoxPushTeamMDP_AlwaysTogether(**EXP1_MAP)

    sim.set_autonomous_agent(cb_get_A1_action=lambda **kwargs: get_exp1_action(
        mdp, BoxPushSimulator.AGENT1, 0.3, **kwargs),
                             cb_get_A2_action=lambda **kwargs: get_exp1_action(
                                 mdp, BoxPushSimulator.AGENT2, 0.3, **kwargs))
    sim.run_simulation(1, "exp_file_name", "text_exp")
