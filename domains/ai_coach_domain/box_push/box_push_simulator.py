from typing import Hashable, Mapping, Tuple, Sequence
import numpy as np
from ai_coach_domain.simulator import Simulator
from ai_coach_domain.box_push.box_push_helper import EventType, transition

Coord = Tuple[int, int]


class BoxPushSimulator(Simulator):
  AGENT1 = 0
  AGENT2 = 1

  def __init__(self, id: Hashable) -> None:
    super().__init__(id)

  def init_game(self,
                x_grid: Coord,
                y_grid: Coord,
                boxes: Sequence[Coord] = [],
                goals: Sequence[Coord] = [],
                walls: Sequence[Coord] = [],
                wall_dir: Sequence[int] = [],
                drops: Sequence[Coord] = []):
    self.x_grid = x_grid
    self.y_grid = y_grid
    self.boxes = boxes
    self.goals = goals
    self.walls = walls
    self.wall_dir = wall_dir
    self.drops = drops

    self.reset_game()

  def init_game_with_test_map(self, x_grid, y_grid):
    boxes = [(1, 3), (2, 5), (4, 2)]
    goals = [(x_grid - 1, y_grid - 1)]
    walls = [(x_grid - 5, y_grid - 1 - i)
             for i in range(5)] + [(x_grid - 1 - i, y_grid - 5)
                                   for i in range(3)]
    wall_dir = [0 for i in range(5)] + [1 for i in range(3)]
    drops = []

    self.init_game(x_grid, y_grid, boxes, goals, walls, wall_dir, drops)

  def reset_game(self):
    self.a1_pos = (self.x_grid - 1, 0)
    self.a2_pos = (0, self.y_grid - 1)
    # starts with their original locations
    self.box_states = [0] * len(self.boxes)
    self.a1_action = None
    self.a2_action = None
    self.a1_latent = None
    self.a2_latent = None
    self.changed_state = []

  def _generate_map(self):
    self.boxes = [(1, 3), (2, 5), (4, 2)]
    self.goals = [(self.x_grid - 1, self.y_grid - 1)]
    self.walls = [(self.x_grid - 5, self.y_grid - 1 - i)
                  for i in range(5)] + [(self.x_grid - 1 - i, self.y_grid - 5)
                                        for i in range(3)]
    self.wall_dir = [0 for i in range(5)] + [1 for i in range(3)]
    # self.drops = [(X_GRID - 4, Y_GRID - 5)]
    self.drops = []

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

    list_next_env = transition(self.box_states, self.a1_pos, self.a2_pos,
                               a1_action, a2_action, self.boxes, self.goals,
                               self.walls, self.drops, self.x_grid, self.y_grid)

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
        self.a1_action = event_type
      else:
        self.a1_latent = value
        self.changed_state.append("a1_latent")
    elif agent == BoxPushSimulator.AGENT2:
      if event_type != EventType.SET_LATENT:
        self.a2_action = event_type
      else:
        self.a2_latent = value
        self.changed_state.append("a2_latent")

  def get_action(self) -> Mapping[Hashable, Hashable]:
    map_a2a = {
        BoxPushSimulator.AGENT1: self.a1_action,
        BoxPushSimulator.AGENT2: self.a2_action
    }

    self.a1_action = None
    self.a2_action = None

    return map_a2a

  def get_env_info(self):
    return {
        "box_states": self.box_states,
        "boxes": self.boxes,
        "goals": self.goals,
        "drops": self.drops,
        "walls": self.walls,
        "wall_dir": self.wall_dir,
        "a1_pos": self.a1_pos,
        "a2_pos": self.a2_pos,
        "a1_latent": self.a1_latent,
        "a2_latent": self.a2_latent
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

  def save_history(self):
    pass

  def is_finished(self) -> bool:
    if self.current_step > self.max_steps:
      return True

    for state in self.box_states:
      if state is not (self.get_num_box_state() - 1):
        return False

    return True

  @classmethod
  def read_file(cls, file_name):
    pass
