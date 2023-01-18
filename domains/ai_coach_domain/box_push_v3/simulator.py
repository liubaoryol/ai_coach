from typing import Sequence
from ai_coach_domain.box_push.simulator import BoxPushSimulator, Coord
from ai_coach_domain.box_push_v3.transition import transition_mixed_noisy
import random


class BoxPushSimulatorV3(BoxPushSimulator):

  def __init__(self, fix_init: bool = False) -> None:
    super().__init__(0)
    self.fix_init = fix_init

  def init_game(self, x_grid: int, y_grid: int, a1_init: Coord, a2_init: Coord,
                boxes: Sequence[Coord], goals: Sequence[Coord],
                walls: Sequence[Coord], drops: Sequence[Coord],
                wall_dir: Sequence[int], box_types: Sequence[int], **kwargs):
    self.box_types = box_types
    self.possible_positions = []
    for x in range(x_grid):
      for y in range(y_grid):
        pos = (x, y)
        if pos not in walls and pos not in goals:
          self.possible_positions.append(pos)

    super().init_game(x_grid, y_grid, a1_init, a2_init, boxes, goals, walls,
                      drops, wall_dir, **kwargs)

  def _get_transition_distribution(self, a1_action, a2_action):
    return transition_mixed_noisy(self.box_states, self.a1_pos, self.a2_pos,
                                  a1_action, a2_action, self.boxes, self.goals,
                                  self.walls, self.drops, self.x_grid,
                                  self.y_grid, self.box_types, self.a1_init,
                                  self.a2_init)

  def reset_game(self):
    self.current_step = 0
    self.history = []

    if self.fix_init:
      self.a1_pos = self.a1_init
      self.a2_pos = self.a2_init
    else:
      self.a1_pos = random.choice(self.possible_positions)
      self.a2_pos = random.choice(self.possible_positions)
    # starts with their original locations
    self.box_states = [0] * len(self.boxes)

    if self.agent_1 is not None:
      self.agent_1.init_latent(self.get_state_for_each_agent(self.AGENT1))
    if self.agent_2 is not None:
      self.agent_2.init_latent(self.get_state_for_each_agent(self.AGENT2))
    self.changed_state = set()
