from typing import Hashable, Tuple
import random
from stand_alone.app import AppInterface
from ai_coach_domain.box_push import EventType, BoxState, conv_box_idx_2_state
from ai_coach_domain.box_push.box_push_maps import TUTORIAL_MAP
from ai_coach_domain.box_push.box_push_simulator import BoxPushSimulator
from ai_coach_domain.box_push import BoxPushSimulator_AlwaysTogether
from ai_coach_domain.box_push.box_push_team_mdp import (
    BoxPushTeamMDP_AlwaysTogether)
from ai_coach_domain.box_push.box_push_policy import (get_simple_action)
# from ai_coach_domain.box_push import (BoxPushSimulator_AlwaysTogether as
#                                       BoxPushSimulator)
# from ai_coach_domain.box_push.box_push_mdp import BoxPushMDP_AlwaysTogether
# from ai_coach_domain.box_push.box_push_policy import get_exp1_action


class BoxPushApp(AppInterface):
  def __init__(self) -> None:
    super().__init__()

  def _init_game(self):
    'define game related variables and objects'
    GAME_ENV_ID = 0
    game_map = TUTORIAL_MAP
    self.x_grid = game_map["x_grid"]
    self.y_grid = game_map["y_grid"]
    self.game = BoxPushSimulator_AlwaysTogether(GAME_ENV_ID)
    self.mdp = BoxPushTeamMDP_AlwaysTogether(**TUTORIAL_MAP)

    self.game.init_game(**game_map)
    self.game.set_autonomous_agent(
        cb_get_A2_action=lambda **kwargs: get_simple_action(
            BoxPushSimulator.AGENT2, **kwargs))
    self.game.event_input(BoxPushSimulator.AGENT2, EventType.SET_LATENT,
                          ("pickup", 0))

  def _init_gui(self):
    self.main_window.title("Box Push")
    self.canvas_width = 300
    self.canvas_height = 300
    super()._init_gui()

  def _conv_key_to_agent_event(self,
                               key_sym) -> Tuple[Hashable, Hashable, Hashable]:
    agent_id = None
    action = None
    value = None
    # agent1 move
    if key_sym == "Left":
      agent_id = BoxPushSimulator.AGENT1
      action = EventType.LEFT
    elif key_sym == "Right":
      agent_id = BoxPushSimulator.AGENT1
      action = EventType.RIGHT
    elif key_sym == "Up":
      agent_id = BoxPushSimulator.AGENT1
      action = EventType.UP
    elif key_sym == "Down":
      agent_id = BoxPushSimulator.AGENT1
      action = EventType.DOWN
    elif key_sym == "p":
      agent_id = BoxPushSimulator.AGENT1
      action = EventType.HOLD
    elif key_sym == "o":
      agent_id = BoxPushSimulator.AGENT2
      action = EventType.SET_LATENT

      a2_hold = False
      valid_boxes = []
      for idx, bidx in enumerate(self.game.box_states):
        bstate = conv_box_idx_2_state(bidx, len(self.game.drops),
                                      len(self.game.goals))
        if bstate[0] == BoxState.Original:
          valid_boxes.append(idx)
        elif bstate[0] in [BoxState.WithAgent2, BoxState.WithBoth]:
          a2_hold = True
          break

      if a2_hold:
        value = ("goal", 0)
      else:
        idx = random.choice(valid_boxes)
        value = ("pickup", idx)

    return (agent_id, action, value)

  def _conv_mouse_to_agent_event(
      self, is_left: bool,
      cursor_pos: Tuple[float, float]) -> Tuple[Hashable, Hashable, Hashable]:
    # find the target hit by the cursor
    # self.canvas_width
    # self.canvas_height

    latent = 0

    return (BoxPushSimulator.AGENT1, EventType.SET_LATENT, latent)

  def _update_canvas_scene(self):
    data = self.game.get_env_info()
    box_states = data["box_states"]
    boxes = data["boxes"]
    drops = data["drops"]
    goals = data["goals"]
    walls = data["walls"]
    a1_pos = data["a1_pos"]
    a2_pos = data["a2_pos"]
    # a1_latent = data["a1_latent"]
    # a2_latent = data["a2_latent"]

    x_unit = int(self.canvas_width / self.x_grid)
    y_unit = int(self.canvas_height / self.y_grid)

    self.clear_canvas()
    for coord in boxes:
      self.create_rectangle(coord[0] * x_unit, coord[1] * y_unit,
                            (coord[0] + 1) * x_unit, (coord[1] + 1) * y_unit,
                            "gray")

    for coord in goals:
      self.create_rectangle(coord[0] * x_unit, coord[1] * y_unit,
                            (coord[0] + 1) * x_unit, (coord[1] + 1) * y_unit,
                            "gold")

    for coord in walls:
      self.create_rectangle(coord[0] * x_unit, coord[1] * y_unit,
                            (coord[0] + 1) * x_unit, (coord[1] + 1) * y_unit,
                            "black")

    for coord in drops:
      self.create_rectangle(coord[0] * x_unit, coord[1] * y_unit,
                            (coord[0] + 1) * x_unit, (coord[1] + 1) * y_unit,
                            "gray")

    a1_hold = False
    a2_hold = False
    for bidx, sidx in enumerate(box_states):
      state = conv_box_idx_2_state(sidx, len(drops), len(goals))
      box = None
      box_color = "green2"
      if state[0] == BoxState.Original:
        box = boxes[bidx]
      elif state[0] == BoxState.WithAgent1:
        box = a1_pos
        a1_hold = True
        box_color = "green4"
      elif state[0] == BoxState.WithAgent2:
        box = a2_pos
        a2_hold = True
        box_color = "green4"
      elif state[0] == BoxState.WithBoth:
        box = a1_pos
        a1_hold = True
        a2_hold = True
        box_color = "green4"
      elif state[0] == BoxState.OnDropLoc:
        box = drops[state[1]]

      if box is not None:
        self.create_rectangle(box[0] * x_unit, box[1] * y_unit,
                              (box[0] + 1) * x_unit, (box[1] + 1) * y_unit,
                              box_color)

    a1_color = "blue"
    if a1_hold:
      a1_color = "dark slate blue"
    self.create_circle((a1_pos[0] + 0.5) * x_unit, (a1_pos[1] + 0.5) * y_unit,
                       x_unit * 0.5, a1_color)

    a2_color = "red"
    if a2_hold:
      a2_color = "indian red"
    self.create_circle((a2_pos[0] + 0.5) * x_unit, (a2_pos[1] + 0.5) * y_unit,
                       x_unit * 0.5, a2_color)

  def _update_canvas_overlay(self):
    pass

  def _on_game_end(self):
    self.game.reset_game()
    self._update_canvas_scene()
    self._update_canvas_overlay()
    self._on_start_btn_clicked()


if __name__ == "__main__":
  app = BoxPushApp()
  app.run()
