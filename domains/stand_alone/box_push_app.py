from typing import Hashable, Tuple
from stand_alone.app import AppInterface
from ai_coach_domain.box_push import BoxPushSimulator
from ai_coach_domain.box_push import EventType


class BoxPushApp(AppInterface):
  def __init__(self) -> None:
    super().__init__()

  def _init_gui(self):
    self.main_window.title("Box Push")
    self.canvas_width = 300
    self.canvas_height = 300
    super()._init_gui()

  def _conv_key_to_agent_event(self,
                               key_sym) -> Tuple[Hashable, Hashable, Hashable]:
    agent_id = None
    action = None
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

    return (agent_id, action, None)

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
    boxes = data["boxes"]
    a1_pos = data["a1_pos"]
    a2_pos = data["a2_pos"]
    a1_hold = data["a1_hold"]
    a2_hold = data["a2_hold"]
    x_unit = int(self.canvas_width / BoxPushSimulator.X_GRID)
    y_unit = int(self.canvas_height / BoxPushSimulator.Y_GRID)

    self.canvas.delete("all")
    for box in boxes:
      if box is not None:
        self.create_rectangle(box[0] * x_unit, box[1] * y_unit,
                              (box[0] + 1) * x_unit, (box[1] + 1) * y_unit,
                              "black")
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

  def _init_game(self):
    GAME_ENV_ID = 0
    self.game = BoxPushSimulator(GAME_ENV_ID)

  def _on_game_end(self):
    self.game.reset_game()
    self._update_canvas_scene()
    self._update_canvas_overlay()
    self._on_start_btn_clicked()


if __name__ == "__main__":
  app = BoxPushApp()
  app.run()
