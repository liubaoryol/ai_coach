from typing import Hashable, Tuple, Sequence
from stand_alone.app import AppInterface
import numpy as np
from aic_domain.rescue import E_EventType, Location, Place, Route, E_Type
from aic_domain.rescue.maps import MAP_RESCUE
from aic_domain.rescue.simulator import RescueSimulator
from aic_domain.agent.cached_agent import BTILCachedPolicy
from aic_domain.rescue.agent import (AIAgent_Rescue_PartialObs,
                                     AIAgent_Rescue_BTIL)
from aic_domain.rescue.policy import Policy_Rescue
from aic_domain.rescue.mdp import MDP_Rescue_Task, MDP_Rescue_Agent
import pickle
from aic_core.intervention.feedback_strategy import (get_sorted_x_combos)

GAME_MAP = MAP_RESCUE

TEST_BTIL_AGENT = True
DATA_DIR = "analysis/TIC_results/data/"
V_VAL_FILE_NAME = "rescue_2_500_0,30_30_merged_v_values_learned.pickle"
# V_VAL_FILE_NAME = None


class RescueApp(AppInterface):
  def __init__(self) -> None:
    super().__init__()

  def _init_game(self):
    self.game = RescueSimulator()
    self.game.max_steps = 100

    self.game.init_game(**GAME_MAP)

    if V_VAL_FILE_NAME is not None:
      with open(DATA_DIR + V_VAL_FILE_NAME, 'rb') as handle:
        self.np_v_values = pickle.load(handle)

    task_mdp = MDP_Rescue_Task(**GAME_MAP)
    agent_mdp = MDP_Rescue_Agent(**GAME_MAP)
    self.mdp = task_mdp

    if not TEST_BTIL_AGENT:
      TEMPERATURE = 0.3
      init_states = ([1] * len(GAME_MAP["work_locations"]), GAME_MAP["a1_init"],
                     GAME_MAP["a2_init"])

      policy1 = Policy_Rescue(task_mdp, agent_mdp, TEMPERATURE, 0)
      agent1 = AIAgent_Rescue_PartialObs(init_states, 0, policy1)

      # agent2 = AIAgent_Rescue(AGENT_2, policy2)
      policy2 = Policy_Rescue(task_mdp, agent_mdp, TEMPERATURE, 1)
      agent2 = AIAgent_Rescue_PartialObs(init_states, 1, policy2)
    else:
      data_dir = "analysis/TIC_results/data/learned_models/"  # noqa: E501
      np_policy_1 = np.load(
          data_dir + "rescue_2_btil2_policy_synth_woTx_FTTT_500_0,30_a1.npy")
      test_policy_1 = BTILCachedPolicy(np_policy_1, task_mdp, 0,
                                       agent_mdp.latent_space)
      np_policy_2 = np.load(
          data_dir + "rescue_2_btil2_policy_synth_woTx_FTTT_500_0,30_a2.npy")
      test_policy_2 = BTILCachedPolicy(np_policy_2, task_mdp, 1,
                                       agent_mdp.latent_space)

      np_tx_1 = np.load(data_dir +
                        "rescue_2_btil2_tx_synth_FTTT_500_0,30_a1.npy")
      np_tx_2 = np.load(data_dir +
                        "rescue_2_btil2_tx_synth_FTTT_500_0,30_a2.npy")
      mask = (False, True, True, True)
      agent1 = AIAgent_Rescue_BTIL(np_tx_1, mask, test_policy_1, 0)
      agent2 = AIAgent_Rescue_BTIL(np_tx_2, mask, test_policy_2, 1)

    self.game.set_autonomous_agent(agent1, agent2)

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
    places = data["places"]  # type: Sequence[Place]
    routes = data["routes"]  # type: Sequence[Route]
    a1_pos = data["a1_pos"]  # type: Location
    a2_pos = data["a2_pos"]  # type: Location

    self.clear_canvas()

    for route in routes:
      x_s, y_s = places[route.start].coord
      x_s = x_s * self.canvas_width
      y_s = y_s * self.canvas_height
      for coord in route.coords:
        x_e, y_e = coord
        x_e = x_e * self.canvas_width
        y_e = y_e * self.canvas_height
        self.create_line(x_s, y_s, x_e, y_e, "green", 10)
        x_s, y_s = x_e, y_e

      x_e, y_e = places[route.end].coord
      x_e = x_e * self.canvas_width
      y_e = y_e * self.canvas_height
      self.create_line(x_s, y_s, x_e, y_e, "green", 10)

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

    def get_coord(loc: Location):
      if loc.type == E_Type.Place:
        return places[loc.id].coord
      else:
        route_id = loc.id  # type: int
        route = routes[route_id]  # type: Route
        idx = loc.index
        return route.coords[idx]

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
    if V_VAL_FILE_NAME is not None:
      game = self.game  # type: RescueSimulator
      tup_state = tuple(game.get_current_state())
      oidx = self.mdp.conv_sim_states_to_mdp_sidx(tup_state)
      list_combos = get_sorted_x_combos(self.np_v_values, oidx)
      print(list_combos)

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
