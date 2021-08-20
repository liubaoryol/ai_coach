from typing import Hashable, Mapping
import numpy as np
from domains.interface.simulator import Simulator
from domains.box_push.box_push_helper import EventType, transition


class BoxPushSimulator(Simulator):
  X_GRID = 6
  Y_GRID = 6
  NUM_BOX = 3
  GOAL = (0, 0)
  AGENT1 = 0
  AGENT2 = 1

  def __init__(self, id: Hashable) -> None:
    super().__init__(id)

  def _init_env(self, *args, **kwargs):
    self.reset_game()
    # self.a1_pos = (BoxPushSimulator.X_GRID, 0)
    # self.a2_pos = (0, BoxPushSimulator.Y_GRID)
    # self.a1_hold = False
    # self.a2_hold = False
    # self.a1_latent = None
    # self.a2_latent = None
    # self.boxes = self._generate_boxes()

  def _generate_boxes(self):
    return [(1, 3), (2, 5), (4, 2)]

  def take_a_step(self, map_agent_2_action: Mapping[Hashable,
                                                    Hashable]) -> None:
    self.current_step += 1
    a1_action = None
    if BoxPushSimulator.AGENT1 in map_agent_2_action:
      a1_action = map_agent_2_action[BoxPushSimulator.AGENT1]

    a2_action = None
    if BoxPushSimulator.AGENT2 in map_agent_2_action:
      a2_action = map_agent_2_action[BoxPushSimulator.AGENT2]

    self.__transition(a1_action, a2_action)

  def __transition(self, a1_action, a2_action):
    if a1_action is None:
      a1_action = EventType.STAY

    if a2_action is None:
      a2_action = EventType.STAY

    list_next_env = transition(self.boxes, self.a1_pos, self.a2_pos,
                               self.a1_hold, self.a2_hold, a1_action, a2_action,
                               [BoxPushSimulator.GOAL], BoxPushSimulator.X_GRID,
                               BoxPushSimulator.Y_GRID)

    list_prop = []
    for item in list_next_env:
      list_prop.append(item[0])

    # Not sure of syntax
    idx_c = np.random.choice(range(len(list_next_env)), 1, p=list_prop)[0]
    _, boxes, a1_pos, a2_pos, a1_h, a2_h = list_next_env[idx_c]
    self.a1_pos = a1_pos
    self.a2_pos = a2_pos
    self.a1_hold = a1_h
    self.a2_hold = a2_h
    self.boxes = boxes

  def reset_game(self):
    self.a1_pos = (BoxPushSimulator.X_GRID - 1, 0)
    self.a2_pos = (0, BoxPushSimulator.Y_GRID - 1)
    self.a1_hold = False
    self.a2_hold = False
    self.a1_latent = None
    self.a2_latent = None
    self.boxes = self._generate_boxes()
    self.a1_key = None
    self.a2_key = None

  def get_num_agents(self):
    return 2

  def event_input(self, agent: Hashable, event_type: Hashable, value):
    if agent == BoxPushSimulator.AGENT1:
      if event_type != EventType.SET_LATENT:
        self.a1_key = event_type
      else:
        self.a1_latent = value
    else:  # agent == BoxPushSimulator.AGENT2
      if event_type != EventType.SET_LATENT:
        self.a2_key = event_type
      else:
        self.a2_latent = value

  def get_action(self) -> Mapping[Hashable, Hashable]:
    return {
        BoxPushSimulator.AGENT1: self.a1_key,
        BoxPushSimulator.AGENT2: self.a2_key
    }

  def get_env_info(self):
    return {
        "boxes": self.boxes,
        "a1_pos": self.a1_pos,
        "a2_pos": self.a2_pos,
        "a1_hold": self.a1_hold,
        "a2_hold": self.a2_hold,
        "a1_latent": self.a1_latent,
        "a2_latent": self.a2_latent
    }

  def get_changed_objects(self):
    return self.get_env_info()

  @classmethod
  def read_file(cls, file_name):
    pass

  def save_history(self):
    pass

  def is_finished(self) -> bool:
    if self.current_step > self.max_steps:
      return True

    for item in self.boxes:
      if item is not None:
        return False

    return True
