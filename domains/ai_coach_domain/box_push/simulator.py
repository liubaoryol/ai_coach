from typing import Hashable, Mapping, Tuple, Sequence
import abc
import os
import numpy as np
from ai_coach_domain.simulator import Simulator
from ai_coach_domain.agent import SimulatorAgent, InteractiveAgent
from ai_coach_domain.box_push import EventType, AGENT_ACTIONSPACE
from ai_coach_domain.box_push.transition import (transition_alone_and_together,
                                                 transition_always_together,
                                                 transition_always_alone)

Coord = Tuple[int, int]


class BoxPushSimulator(Simulator):
  AGENT1 = 0  # must be defined in a way that aligns with agent idx for policy
  AGENT2 = 1

  def __init__(
      self,
      id: Hashable,
      tuple_action_when_none: Tuple = (EventType.STAY, EventType.STAY)
  ) -> None:
    #  input1: agent idx
    super().__init__(id)
    self.agent_1 = None
    self.agent_2 = None
    self.tuple_action_when_none = tuple_action_when_none

  def init_game(self,
                x_grid: int,
                y_grid: int,
                a1_init: Coord,
                a2_init: Coord,
                boxes: Sequence[Coord] = [],
                goals: Sequence[Coord] = [],
                walls: Sequence[Coord] = [],
                drops: Sequence[Coord] = [],
                wall_dir: Sequence[int] = [],
                **kwargs):
    self.x_grid = x_grid
    self.y_grid = y_grid
    self.a1_init = a1_init
    self.a2_init = a2_init
    self.boxes = boxes
    self.goals = goals
    self.walls = walls
    self.drops = drops
    self.wall_dir = wall_dir

    self.reset_game()

  def get_state_for_each_agent(self, agent_idx):
    'Redefine this method at subclasses as needed'
    return [self.box_states, self.a1_pos, self.a2_pos]

  def set_autonomous_agent(self,
                           agent1: SimulatorAgent = InteractiveAgent(),
                           agent2: SimulatorAgent = InteractiveAgent()):
    self.agent_1 = agent1
    self.agent_2 = agent2

    self.agents = [agent1, agent2]

    # order can be important as Agent2 state may include Agent1's mental state,
    # or vice versa. here we assume agent2 updates its mental state later
    self.agent_1.init_latent(self.get_state_for_each_agent(self.AGENT1))
    self.agent_2.init_latent(self.get_state_for_each_agent(self.AGENT2))

  def reset_game(self):
    self.current_step = 0
    self.history = []

    self.a1_pos = self.a1_init
    self.a2_pos = self.a2_init
    # starts with their original locations
    self.box_states = [0] * len(self.boxes)

    if self.agent_1 is not None:
      self.agent_1.init_latent(self.get_state_for_each_agent(self.AGENT1))
    if self.agent_2 is not None:
      self.agent_2.init_latent(self.get_state_for_each_agent(self.AGENT2))
    self.changed_state = set()

  def get_score(self):
    return -self.get_current_step()

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
      a1_action = self.tuple_action_when_none[0]
    if a2_action is None:
      a2_action = self.tuple_action_when_none[1]

    a1_lat = self.agent_1.get_current_latent()
    if a1_lat is None:
      a1_lat = ("NA", 0)
    a1_lat_0 = a1_lat[0] if a1_lat[0] is not None else "NA"
    a1_lat_1 = a1_lat[1] if a1_lat[1] is not None else 0
    a1_lat = (a1_lat_0, a1_lat_1)

    a2_lat = self.agent_2.get_current_latent()
    if a2_lat is None:
      a2_lat = ("NA", 0)
    a2_lat_0 = a2_lat[0] if a2_lat[0] is not None else "NA"
    a2_lat_1 = a2_lat[1] if a2_lat[1] is not None else 0
    a2_lat = (a2_lat_0, a2_lat_1)

    a1_cur_state = tuple(self.get_state_for_each_agent(self.AGENT1))
    a2_cur_state = tuple(self.get_state_for_each_agent(self.AGENT2))

    state = [
        self.current_step, self.box_states, self.a1_pos, self.a2_pos, a1_action,
        a2_action, a1_lat, a2_lat
    ]
    self.history.append(state)

    self._transition(a1_action, a2_action)
    self.current_step += 1
    self.changed_state.add("current_step")

    # update mental model
    tuple_actions = (a1_action, a2_action)
    self.agent_1.update_mental_state(a1_cur_state, tuple_actions,
                                     self.get_state_for_each_agent(self.AGENT1))
    self.agent_2.update_mental_state(a2_cur_state, tuple_actions,
                                     self.get_state_for_each_agent(self.AGENT2))
    self.changed_state.add("a1_latent")
    self.changed_state.add("a2_latent")

  def _transition(self, a1_action, a2_action):
    list_next_env = self._get_transition_distribution(a1_action, a2_action)

    list_prop = []
    for item in list_next_env:
      list_prop.append(item[0])

    idx_c = np.random.choice(range(len(list_next_env)), 1, p=list_prop)[0]
    _, box_states, a1_pos, a2_pos = list_next_env[idx_c]
    self.a1_pos = a1_pos
    self.a2_pos = a2_pos
    self.box_states = box_states

    self.changed_state.add("a1_pos")
    self.changed_state.add("a2_pos")
    self.changed_state.add("box_states")

  @abc.abstractmethod
  def _get_transition_distribution(self, a1_action, a2_action):
    pass

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

  def event_input(self, agent: Hashable, event_type: Hashable, value):
    if (agent is None) or (event_type is None):
      return

    if agent == BoxPushSimulator.AGENT1:
      if event_type != EventType.SET_LATENT:
        self.agent_1.set_action(event_type)
      else:
        self.agent_1.set_latent(value)
        self.changed_state.add("a1_latent")
    elif agent == BoxPushSimulator.AGENT2:
      if event_type != EventType.SET_LATENT:
        self.agent_2.set_action(event_type)
      else:
        self.agent_2.set_latent(value)
        self.changed_state.add("a2_latent")

  def get_joint_action(self) -> Mapping[Hashable, Hashable]:

    map_a2a = {}
    map_a2a[BoxPushSimulator.AGENT1] = self.agent_1.get_action(
        self.get_state_for_each_agent(self.AGENT1))
    map_a2a[BoxPushSimulator.AGENT2] = self.agent_2.get_action(
        self.get_state_for_each_agent(self.AGENT2))

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
        "a1_latent": self.agent_1.get_current_latent(),
        "a2_latent": self.agent_2.get_current_latent(),
        "wall_dir": self.wall_dir,
        "current_step": self.current_step
    }

  def get_changed_objects(self):
    dict_changed_obj = {}
    for state in self.changed_state:
      if state == "a1_latent":
        dict_changed_obj[state] = self.agent_1.get_current_latent()
      elif state == "a2_latent":
        dict_changed_obj[state] = self.agent_2.get_current_latent()
      else:
        dict_changed_obj[state] = getattr(self, state)
    self.changed_state = set()
    return dict_changed_obj

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

        txtfile.write('%d; %d; ' % (AGENT_ACTIONSPACE.action_to_idx[a1act],
                                    AGENT_ACTIONSPACE.action_to_idx[a2act]))

        txtfile.write('%s, %d; ' % a1lat)
        txtfile.write('%s, %d; ' % a2lat)
        txtfile.write('\n')

      # last state
      txtfile.write('%d; ' % (self.current_step, ))  # cur step
      # box states
      for idx in range(len(self.box_states) - 1):
        txtfile.write('%d, ' % (self.box_states[idx], ))
      txtfile.write('%d; ' % (self.box_states[-1], ))

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
          for dummy in range(8 - len(states)):
            states.append(None)
        step, bstate, a1pos, a2pos, a1act, a2act, a1lat, a2lat = states
        box_state = tuple([int(elem) for elem in bstate.split(", ")])
        a1_pos = tuple([int(elem) for elem in a1pos.split(", ")])
        a2_pos = tuple([int(elem) for elem in a2pos.split(", ")])
        if a1act is None:
          a1_act = None
        else:
          a1_act = int(a1act)
        if a2act is None:
          a2_act = None
        else:
          a2_act = int(a2act)
        if a1lat is None:
          a1_lat = None
        else:
          a1lat_tmp = a1lat.split(", ")
          a1_lat = (a1lat_tmp[0], int(a1lat_tmp[1]))
        if a2lat is None:
          a2_lat = None
        else:
          a2lat_tmp = a2lat.split(", ")
          a2_lat = (a2lat_tmp[0], int(a2lat_tmp[1]))
        traj.append([box_state, a1_pos, a2_pos, a1_act, a2_act, a1_lat, a2_lat])

    return traj


class BoxPushSimulator_AloneOrTogether(BoxPushSimulator):

  def __init__(
      self,
      id: Hashable,
      tuple_action_when_none: Tuple = (EventType.STAY, EventType.STAY)
  ) -> None:
    super().__init__(id, tuple_action_when_none=tuple_action_when_none)

  def _get_transition_distribution(self, a1_action, a2_action):
    return transition_alone_and_together(self.box_states, self.a1_pos,
                                         self.a2_pos, a1_action, a2_action,
                                         self.boxes, self.goals, self.walls,
                                         self.drops, self.x_grid, self.y_grid)


class BoxPushSimulator_AlwaysTogether(BoxPushSimulator):

  def __init__(
      self,
      id: Hashable,
      tuple_action_when_none: Tuple = (EventType.STAY, EventType.STAY)
  ) -> None:
    super().__init__(id, tuple_action_when_none=tuple_action_when_none)

  def _get_transition_distribution(self, a1_action, a2_action):
    return transition_always_together(self.box_states, self.a1_pos, self.a2_pos,
                                      a1_action, a2_action, self.boxes,
                                      self.goals, self.walls, self.drops,
                                      self.x_grid, self.y_grid)


class BoxPushSimulator_AlwaysAlone(BoxPushSimulator):

  def __init__(
      self,
      id: Hashable,
      tuple_action_when_none: Tuple = (EventType.STAY, EventType.STAY)
  ) -> None:
    super().__init__(id, tuple_action_when_none=tuple_action_when_none)

  def _get_transition_distribution(self, a1_action, a2_action):
    return transition_always_alone(self.box_states, self.a1_pos, self.a2_pos,
                                   a1_action, a2_action, self.boxes, self.goals,
                                   self.walls, self.drops, self.x_grid,
                                   self.y_grid)


if __name__ == "__main__":
  test = False
  exp1 = False
  if test:
    from ai_coach_domain.box_push.maps import TEST_MAP
    from ai_coach_domain.box_push.agent import BoxPushSimpleAgent

    sim = BoxPushSimulator_AlwaysAlone(0)
    sim.init_game(**TEST_MAP)
    agent1 = BoxPushSimpleAgent(0, sim.x_grid, sim.y_grid, sim.boxes, sim.goals,
                                sim.walls, sim.drops)
    agent2 = BoxPushSimpleAgent(1, sim.x_grid, sim.y_grid, sim.boxes, sim.goals,
                                sim.walls, sim.drops)

    sim.set_autonomous_agent(agent1=agent1, agent2=agent2)
    sim.run_simulation(1, "test_file_name", "text")

  if exp1:
    from ai_coach_domain.box_push.maps import EXP1_MAP
    from ai_coach_domain.box_push.agent import (BoxPushAIAgent_Team1,
                                                BoxPushAIAgent_Team2)
    from ai_coach_domain.box_push.policy import BoxPushPolicyTeamExp1
    from ai_coach_domain.box_push.mdp import BoxPushTeamMDP_AlwaysTogether

    sim = BoxPushSimulator_AlwaysTogether(0)
    sim.init_game(**EXP1_MAP)
    mdp = BoxPushTeamMDP_AlwaysTogether(**EXP1_MAP)
    policy1 = BoxPushPolicyTeamExp1(mdp, temperature=1, agent_idx=0)
    policy2 = BoxPushPolicyTeamExp1(mdp, temperature=1, agent_idx=1)
    agent1 = BoxPushAIAgent_Team1(policy1)
    agent2 = BoxPushAIAgent_Team2(policy2)

    sim.set_autonomous_agent(agent1=agent1, agent2=agent2)
    sim.run_simulation(1, "exp_file_name", "text_exp")
