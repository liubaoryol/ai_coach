from typing import Hashable, Tuple, Mapping, Sequence
from stand_alone.app import AppInterface
import numpy as np
from ai_coach_domain.rescue import (E_EventType, Work, Location, Place, Route,
                                    E_Type, T_Connections)
from ai_coach_domain.rescue.maps import MAP_RESCUE, MAP_RESCUE_2
from ai_coach_domain.rescue.simulator import RescueSimulator
from ai_coach_domain.agent import InteractiveAgent
from ai_coach_domain.rescue.agent import AIAgent_Rescue
from ai_coach_domain.rescue.policy import Policy_Rescue
from ai_coach_domain.rescue.mdp import MDP_Rescue_Task, MDP_Rescue_Agent

GAME_MAP = MAP_RESCUE_2


class RescueApp(AppInterface):
  def __init__(self) -> None:
    super().__init__()

  def _init_game(self):
    self.game = RescueSimulator()
    self.game.max_steps = 100

    AGENT_2 = RescueSimulator.AGENT2
    TEMPERATURE = 0.3

    task_mdp = MDP_Rescue_Task(**GAME_MAP)
    agent_mdp = MDP_Rescue_Agent(**GAME_MAP)
    policy2 = Policy_Rescue(task_mdp, agent_mdp, TEMPERATURE, AGENT_2)
    agent2 = AIAgent_Rescue(AGENT_2, policy2)

    self.game.init_game(**GAME_MAP)
    self.game.set_autonomous_agent(agent2=agent2)

  def _init_gui(self):
    self.main_window.title("Rescue")
    self.canvas_width = 300
    self.canvas_height = 300
    super()._init_gui()

  def _conv_key_to_agent_event(self,
                               key_sym) -> Tuple[Hashable, Hashable, Hashable]:
    agent_id = None
    action = None
    value = None

    # agent 1 move
    if key_sym == "u":
      agent_id = RescueSimulator.AGENT1
      action = E_EventType.Option0
    elif key_sym == "i":
      agent_id = RescueSimulator.AGENT1
      action = E_EventType.Option1
    elif key_sym == "o":
      agent_id = RescueSimulator.AGENT1
      action = E_EventType.Option2
    elif key_sym == "p":
      agent_id = RescueSimulator.AGENT1
      action = E_EventType.Option3
    elif key_sym == "bracketleft":
      agent_id = RescueSimulator.AGENT1
      action = E_EventType.Stay
    elif key_sym == "bracketright":
      agent_id = RescueSimulator.AGENT1
      action = E_EventType.Rescue
    # agent 2 move
    if key_sym == "q":
      agent_id = RescueSimulator.AGENT2
      action = E_EventType.Option0
    elif key_sym == "w":
      agent_id = RescueSimulator.AGENT2
      action = E_EventType.Option1
    elif key_sym == "e":
      agent_id = RescueSimulator.AGENT2
      action = E_EventType.Option2
    elif key_sym == "r":
      agent_id = RescueSimulator.AGENT2
      action = E_EventType.Option3
    elif key_sym == "t":
      agent_id = RescueSimulator.AGENT2
      action = E_EventType.Stay
    elif key_sym == "y":
      agent_id = RescueSimulator.AGENT2
      action = E_EventType.Rescue

    return (agent_id, action, value)

  def _conv_mouse_to_agent_event(
      self, is_left: bool,
      cursor_pos: Tuple[float, float]) -> Tuple[Hashable, Hashable, Hashable]:
    return (None, None, None)

  def _update_canvas_scene(self):
    data = self.game.get_env_info()
    work_state = data["work_states"]  # type: Sequence[int]
    work_locations = data["work_locations"]  # type: Sequence[Location]
    work_info = data["work_info"]  # type: Sequence[Work]
    places = data["places"]  # type: Sequence[Place]
    connections = data["connections"]  # type: Mapping[int, T_Connections]
    routes = data["routes"]  # type: Sequence[Route]
    a1_pos = data["a1_pos"]  # type: Location
    a2_pos = data["a2_pos"]  # type: Location

    self.clear_canvas()
    for place in places:
      x_s = (place.coord[0] - 0.05) * self.canvas_width
      y_s = (place.coord[1] - 0.05) * self.canvas_height
      x_e = (place.coord[0] + 0.05) * self.canvas_width
      y_e = (place.coord[1] + 0.05) * self.canvas_height
      self.create_rectangle(x_s, y_s, x_e, y_e, "yellow")

      offset = -0.02
      for _ in range(place.helps):
        x_s = (place.coord[0] + offset) * self.canvas_width
        y_s = (place.coord[1] - 0.04) * self.canvas_height
        x_e = (place.coord[0] + offset) * self.canvas_width
        y_e = (place.coord[1] - 0.00) * self.canvas_height
        offset += 0.02
        self.create_line(x_s, y_s, x_e, y_e, "black", 1)

    for route in routes:
      place_s = np.array(places[route.start].coord)
      place_e = np.array(places[route.end].coord)

      vec = (place_e - place_s) / np.linalg.norm(place_e - place_s)

      line_s = place_s + vec * 0.05
      line_e = place_e - vec * 0.05
      x_s = line_s[0] * self.canvas_width
      y_s = line_s[1] * self.canvas_height

      x_e = line_e[0] * self.canvas_width
      y_e = line_e[1] * self.canvas_height

      self.create_line(x_s, y_s, x_e, y_e, "green", 10)

    def get_coord(loc: Location):
      place_size_half = 0.05
      if loc.type == E_Type.Place:
        return places[loc.id].coord
      else:
        route_id = loc.id  # type: int
        route = routes[route_id]
        idx = loc.index

        place_s = np.array(places[route.start].coord)
        place_e = np.array(places[route.end].coord)

        vec = (place_e - place_s) / np.linalg.norm(place_e - place_s)

        line_s = place_s + vec * place_size_half
        line_e = place_e - vec * place_size_half

        line_len = np.linalg.norm(line_e - line_s)
        step_len = line_len / route.length

        pos = line_s + (idx + 0.5) * step_len * vec
        return pos

    for widx, done in enumerate(work_state):
      work_coord = get_coord(work_locations[widx])
      if done == 0:
        continue

      x_s = work_coord[0] * self.canvas_width
      y_s = work_coord[1] * self.canvas_height
      wid = 0.03 * self.canvas_width
      hei = 0.03 * self.canvas_height
      self.create_triangle(x_s, y_s, wid, hei, "black")

    rad = 0.02 * self.canvas_width
    a1_coord = get_coord(a1_pos)
    x_c1 = (a1_coord[0] - 0.03) * self.canvas_width
    y_c1 = a1_coord[1] * self.canvas_height
    self.create_circle(x_c1, y_c1, rad, "blue")

    a2_coord = get_coord(a2_pos)
    x_c2 = (a2_coord[0] + 0.03) * self.canvas_width
    y_c2 = a2_coord[1] * self.canvas_height
    self.create_circle(x_c2, y_c2, rad, "red")

    self.create_text(x_c1, y_c1 + 10,
                     str(self.game.agent_1.get_current_latent()))
    self.create_text(x_c2, y_c2 + 10,
                     str(self.game.agent_2.get_current_latent()))

  def _update_canvas_overlay(self):
    pass

  def _on_game_end(self):
    self.game.reset_game()
    self._update_canvas_scene()
    self._update_canvas_overlay()
    self._on_start_btn_clicked()


if __name__ == "__main__":
  app = RescueApp()
  app.run()
