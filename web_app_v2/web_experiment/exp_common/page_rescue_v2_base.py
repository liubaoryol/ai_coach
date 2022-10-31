from typing import Mapping, Any, Sequence, List
import copy
import numpy as np
from ai_coach_domain.rescue_v2 import (E_EventType, Location, E_Type, Place,
                                       T_Connections, Route)
from ai_coach_domain.rescue_v2.simulator import RescueSimulatorV2
import web_experiment.exp_common.canvas_objects as co
from web_experiment.define import EDomainType
from web_experiment.exp_common.page_base import ExperimentPageBase, Exp1UserData
from web_experiment.exp_common.helper_rescue_v2 import (
    location_2_coord_v2, rescue_v2_game_scene, rescue_v2_game_scene_names,
    RESCUE_V2_PLACE_DRAW_INFO)

RESCUE_MAX_STEP = 15


def human_clear_problem(
    dict_prev_game: Mapping[str, Any],
    dict_cur_game: Mapping[str, Any],
    human_action: E_EventType,
):
  work_states_prev = dict_prev_game["work_states"]
  work_states_cur = dict_cur_game["work_states"]
  work_locations = dict_cur_game["work_locations"]
  a1_pos = dict_prev_game["a1_pos"]

  if a1_pos in work_locations:
    widx = work_locations.index(a1_pos)
    wstate_p = work_states_prev[widx]
    wstate_c = work_states_cur[widx]

    if wstate_p != wstate_c and human_action == E_EventType.Rescue:
      return True

  return False


class RescueV2GamePageBase(ExperimentPageBase):
  OPTION_0 = E_EventType.Option0.name
  OPTION_1 = E_EventType.Option1.name
  OPTION_2 = E_EventType.Option2.name
  OPTION_3 = E_EventType.Option3.name
  STAY = E_EventType.Stay.name
  RESCUE = E_EventType.Rescue.name

  ACTION_BUTTONS = [OPTION_0, OPTION_1, OPTION_2, OPTION_3, STAY, RESCUE]

  def __init__(self,
               manual_latent_selection,
               game_map,
               auto_prompt: bool = True,
               prompt_on_change: bool = True,
               prompt_freq: int = 5) -> None:
    super().__init__(True, True, True, EDomainType.Rescue)
    self._MANUAL_SELECTION = manual_latent_selection
    self._GAME_MAP = game_map

    self._PROMPT_ON_CHANGE = prompt_on_change
    self._PROMPT_FREQ = prompt_freq
    self._AUTO_PROMPT = auto_prompt

    self._AGENT1 = RescueSimulatorV2.AGENT1
    self._AGENT2 = RescueSimulatorV2.AGENT2
    self._AGENT3 = RescueSimulatorV2.AGENT3

  def init_user_data(self, user_game_data: Exp1UserData):
    user_game_data.data[Exp1UserData.GAME_DONE] = False
    user_game_data.data[Exp1UserData.SELECT] = False

    game = user_game_data.get_game_ref()
    if game is None:
      game = RescueSimulatorV2()
      game.max_steps = RESCUE_MAX_STEP

      user_game_data.set_game(game)

    game.init_game(**self._GAME_MAP)
    game.set_autonomous_agent()

    user_game_data.data[Exp1UserData.ACTION_COUNT] = 0

  def get_updated_drawing_info(self,
                               user_data: Exp1UserData,
                               clicked_button: str = None,
                               dict_prev_scene_data: Mapping[str, Any] = None):
    if dict_prev_scene_data is None:
      drawing_objs = self._get_init_drawing_objects(user_data)
      commands = self._get_init_commands(user_data)
      animations = None
    else:
      drawing_objs = self._get_updated_drawing_objects(user_data,
                                                       dict_prev_scene_data)
      commands = self._get_button_commands(clicked_button, user_data)
      game = user_data.get_game_ref()
      animations = None
      if clicked_button in self.ACTION_BUTTONS:
        animations = self._get_animations(dict_prev_scene_data,
                                          game.get_env_info())
    drawing_order = self._get_drawing_order(user_data)

    return commands, drawing_objs, drawing_order, animations

  def button_clicked(self, user_game_data: Exp1UserData, clicked_btn: str):
    '''
    user_game_data: NOTE - values will be updated
    return: commands, drawing_objs, drawing_order, animations
      drawing info
    '''

    if clicked_btn in self.ACTION_BUTTONS:
      game = user_game_data.get_game_ref()
      dict_prev_game = copy.deepcopy(game.get_env_info())
      a1_act, a2_act, a3_act, done = self.action_event(user_game_data,
                                                       clicked_btn)
      if done:
        self._on_game_finished(user_game_data)
      else:
        self._on_action_taken(user_game_data, dict_prev_game,
                              (a1_act, a2_act, a3_act))
      return

    elif clicked_btn == co.BTN_SELECT:
      user_game_data.data[Exp1UserData.SELECT] = True
      return

    elif self.is_sel_latent_btn(clicked_btn):
      latent = self.selbtn2latent(clicked_btn)
      if latent is not None:
        game = user_game_data.get_game_ref()
        game.event_input(self._AGENT1, E_EventType.Set_Latent, latent)
        user_game_data.data[Exp1UserData.SELECT] = False
        user_game_data.data[Exp1UserData.ACTION_COUNT] = 0
        return

    return super().button_clicked(user_game_data, clicked_btn)

  def _get_instruction(self, user_game_data: Exp1UserData):
    if user_game_data.data[Exp1UserData.SELECT]:
      return (
          "Please select your current destination among the circled options. " +
          "It can be the same destination as you had previously selected.")
    else:
      return (
          "Please choose your next action. If your destination has changed, " +
          "please update it using the select destination button.")

  def _get_drawing_order(self, user_game_data: Exp1UserData):
    dict_game = user_game_data.get_game_ref().get_env_info()
    drawing_order = []
    drawing_order.append(self.GAME_BORDER)

    drawing_order = (drawing_order +
                     self._game_scene_names(dict_game, user_game_data))
    drawing_order = (drawing_order +
                     self._game_overlay_names(dict_game, user_game_data))
    drawing_order = drawing_order + self.ACTION_BUTTONS
    drawing_order.append(co.BTN_SELECT)

    drawing_order.append(self.TEXT_SCORE)

    drawing_order.append(self.TEXT_INSTRUCTION)

    return drawing_order

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_game_data)

    dict_game = user_game_data.get_game_ref().get_env_info()

    game_objs = self._game_scene(dict_game,
                                 user_game_data,
                                 include_background=True)
    for obj in game_objs:
      dict_objs[obj.name] = obj

    overlay_objs = self._game_overlay(dict_game, user_game_data)
    for obj in overlay_objs:
      dict_objs[obj.name] = obj

    disable_status = self._get_action_btn_disable_state(user_game_data,
                                                        dict_game)
    objs = self._get_btn_actions(dict_game, *disable_status)
    for obj in objs:
      dict_objs[obj.name] = obj

    obj = self._get_btn_select(user_game_data)
    dict_objs[obj.name] = obj

    return dict_objs

  def _get_action_btn_disable_state(self, user_data: Exp1UserData,
                                    game_env: Mapping[Any, Any]):
    selecting = user_data.data[Exp1UserData.SELECT]
    game_done = user_data.data[Exp1UserData.GAME_DONE]

    if selecting or game_done:
      return True, True, True

    rescue_ok = False

    a1pos = game_env["a1_pos"]  # type: Location
    a1_latent = game_env["a1_latent"]
    work_locations = game_env["work_locations"]
    work_states = game_env["work_states"]

    if user_data.data[Exp1UserData.COLLECT_LATENT]:
      if a1pos in work_locations:
        widx = work_locations.index(a1pos)
        if a1_latent == widx and work_states[widx] != 0:
          rescue_ok = True
    else:
      if a1pos in work_locations:
        widx = work_locations.index(a1pos)
        if work_states[widx] != 0:
          rescue_ok = True

    return False, False, not rescue_ok

  def _get_btn_actions(
      self,
      game_env: Mapping[Any, Any],
      disable_move: bool = False,
      disable_stay: bool = False,
      disable_rescue: bool = False) -> Sequence[co.DrawingObject]:
    a1pos = game_env["a1_pos"]  # type: Location

    x_ctrl_cen = int(self.GAME_RIGHT + (co.CANVAS_WIDTH - self.GAME_RIGHT) / 2)
    y_ctrl_cen = int(co.CANVAS_HEIGHT * 0.65)
    x_joy_cen = int(x_ctrl_cen - 75)
    ctrl_origin = np.array([x_joy_cen, y_ctrl_cen])

    arrow_width = 30
    font_size = 18

    list_buttons = []

    # offset = buttonsize[1] + 3
    connections = game_env["connections"]  # type: Mapping[int, T_Connections]
    places = game_env["places"]  # type: Sequence[Place]
    routes = game_env["routes"]  # type: Sequence[Route]

    offset = 30

    coord_c = np.array(location_2_coord_v2(a1pos, places, routes))
    if a1pos.type == E_Type.Place:
      for idx, connection in enumerate(connections[a1pos.id]):
        if connection[0] == E_Type.Place:
          coord_n = np.array(places[connection[1]].coord)
        else:
          if routes[connection[1]].start == a1pos.id:
            coord_n = np.array(routes[connection[1]].coords[0])
          elif routes[connection[1]].end == a1pos.id:
            coord_n = np.array(routes[connection[1]].coords[-1])
          else:
            raise ValueError("Invalid map")

        direction = coord_n - coord_c
        direction = direction / np.linalg.norm(direction)
        origin = ctrl_origin + direction * offset
        origin = (int(origin[0]), int(origin[1]))
        direction = (direction[0], direction[1])

        btn_obj = co.ThickArrow(self.ACTION_BUTTONS[idx],
                                origin,
                                direction,
                                arrow_width,
                                disable=disable_move)
        list_buttons.append(btn_obj)
    else:
      route = routes[a1pos.id]
      index = a1pos.index

      # moving forward
      if index + 1 == route.length:
        coord_n = places[route.end].coord
      else:
        coord_n = route.coords[index + 1]

      direction = coord_n - coord_c
      direction = direction / np.linalg.norm(direction)
      origin = ctrl_origin + direction * offset
      origin = (int(origin[0]), int(origin[1]))
      direction = (direction[0], direction[1])

      btn_obj = co.ThickArrow(self.OPTION_0,
                              origin,
                              direction,
                              arrow_width,
                              disable=disable_move)
      list_buttons.append(btn_obj)

      # moving backward
      if index - 1 < 0:
        coord_n = places[route.start].coord
      else:
        coord_n = route.coords[index - 1]

      direction = coord_n - coord_c
      direction = direction / np.linalg.norm(direction)
      origin = ctrl_origin + direction * offset
      origin = (int(origin[0]), int(origin[1]))
      direction = (direction[0], direction[1])

      btn_obj = co.ThickArrow(self.OPTION_1,
                              origin,
                              direction,
                              arrow_width,
                              disable=disable_move)
      list_buttons.append(btn_obj)

    obj = co.ButtonCircle(self.STAY, (x_joy_cen, y_ctrl_cen),
                          int(0.7 * offset),
                          font_size,
                          "",
                          disable=disable_stay,
                          fill=True,
                          border=False)
    list_buttons.append(obj)
    btn_rescue = co.ButtonRect(self.RESCUE, (x_ctrl_cen + 75, y_ctrl_cen),
                               (120, 50),
                               font_size,
                               "Rescue",
                               disable=disable_rescue)
    list_buttons.append(btn_rescue)

    return list_buttons

  def _get_btn_select(self, user_game_data: Exp1UserData):
    x_ctrl_cen = int(self.GAME_RIGHT + (co.CANVAS_WIDTH - self.GAME_RIGHT) / 2)
    y_ctrl_cen = int(co.CANVAS_HEIGHT * 0.8)

    buttonsize = (int(self.GAME_WIDTH / 3), int(self.GAME_WIDTH / 15))
    font_size = 18

    selecting = user_game_data.data[Exp1UserData.SELECT]
    select_disable = not self._MANUAL_SELECTION or selecting
    btn_select = co.ButtonRect(co.BTN_SELECT, (x_ctrl_cen, y_ctrl_cen),
                               buttonsize,
                               font_size,
                               "Select Destination",
                               disable=select_disable)
    return btn_select

  def _on_action_taken(self, user_game_data: Exp1UserData,
                       dict_prev_game: Mapping[str, Any],
                       tuple_actions: Sequence[Any]):
    '''
    user_cur_game_data: NOTE - values will be updated
    '''

    game = user_game_data.get_game_ref()
    # set selection prompt status
    # TODO: check work state changed
    work_state_changed = human_clear_problem(dict_prev_game,
                                             game.get_env_info(),
                                             tuple_actions[0])

    select_latent = False
    if self._PROMPT_ON_CHANGE and work_state_changed:
      select_latent = True

    if self._AUTO_PROMPT:
      user_game_data.data[Exp1UserData.ACTION_COUNT] += 1
      if user_game_data.data[Exp1UserData.ACTION_COUNT] >= self._PROMPT_FREQ:
        select_latent = True

    user_game_data.data[Exp1UserData.SELECT] = select_latent
    user_game_data.data[Exp1UserData.SCORE] = (
        user_game_data.get_game_ref().current_step)

    # mental state update
    # possibly change the page to draw

  def _on_game_finished(self, user_game_data: Exp1UserData):
    '''
    user_game_data: NOTE - values will be updated
    '''
    user_game_data.data[Exp1UserData.GAME_DONE] = True

    # update score
    game = user_game_data.get_game_ref()
    user_game_data.data[Exp1UserData.SCORE] = game.current_step

  def _get_updated_drawing_objects(
      self,
      user_data: Exp1UserData,
      dict_prev_game: Mapping[str,
                              Any] = None) -> Mapping[str, co.DrawingObject]:
    dict_game = user_data.get_game_ref().get_env_info()
    dict_objs = {}
    game_updated = (dict_prev_game["current_step"] != dict_game["current_step"])
    if game_updated:
      for obj in self._game_scene(dict_game, user_data, False):
        dict_objs[obj.name] = obj

      obj = self._get_score_obj(user_data)
      dict_objs[obj.name] = obj

    for obj in self._game_overlay(dict_game, user_data):
      dict_objs[obj.name] = obj

    obj = self._get_instruction_objs(user_data)
    dict_objs[obj.name] = obj

    disable_status = self._get_action_btn_disable_state(user_data, dict_game)
    objs = self._get_btn_actions(dict_game, *disable_status)
    for obj in objs:
      dict_objs[obj.name] = obj

    obj = self._get_btn_select(user_data)
    dict_objs[obj.name] = obj

    return dict_objs

  def _get_button_commands(self, clicked_btn, user_data: Exp1UserData):
    return {"delete": self.ACTION_BUTTONS}

  def _get_animations(self, dict_prev_game: Mapping[str, Any],
                      dict_cur_game: Mapping[str, Any]):

    list_animations = []
    a1_pos_p = dict_prev_game["a1_pos"]
    a1_pos_c = dict_cur_game["a1_pos"]

    a2_pos_p = dict_prev_game["a2_pos"]
    a2_pos_c = dict_cur_game["a2_pos"]

    a3_pos_p = dict_prev_game["a3_pos"]
    a3_pos_c = dict_cur_game["a3_pos"]

    if a1_pos_p == a1_pos_c:
      obj_name = co.IMG_POLICE_CAR
      amp = int(self.GAME_WIDTH * 0.01)
      obj = {'type': 'vibrate', 'obj_name': obj_name, 'amplitude': amp}
      if obj not in list_animations:
        list_animations.append(obj)

    if a2_pos_p == a2_pos_c:
      obj_name = co.IMG_FIRE_ENGINE
      amp = int(self.GAME_WIDTH * 0.01)
      obj = {'type': 'vibrate', 'obj_name': obj_name, 'amplitude': amp}
      if obj not in list_animations:
        list_animations.append(obj)

    if a3_pos_p == a3_pos_c:
      obj_name = co.IMG_AMBULANCE
      amp = int(self.GAME_WIDTH * 0.01)
      obj = {'type': 'vibrate', 'obj_name': obj_name, 'amplitude': amp}
      if obj not in list_animations:
        list_animations.append(obj)

    return list_animations

  def action_event(self, user_game_data: Exp1UserData, clicked_btn: str):
    '''
    user_game_data: NOTE - values will be updated
    '''
    action = None
    if clicked_btn == self.OPTION_0:
      action = E_EventType.Option0
    elif clicked_btn == self.OPTION_1:
      action = E_EventType.Option1
    elif clicked_btn == self.OPTION_2:
      action = E_EventType.Option2
    elif clicked_btn == self.OPTION_3:
      action = E_EventType.Option3
    elif clicked_btn == self.STAY:
      action = E_EventType.Stay
    elif clicked_btn == self.RESCUE:
      action = E_EventType.Rescue

    game = user_game_data.get_game_ref()
    # should not happen
    assert action is not None
    assert not game.is_finished()

    game.event_input(self._AGENT1, action, None)

    # take actions
    map_agent2action = game.get_joint_action()
    game.take_a_step(map_agent2action)

    return (map_agent2action[self._AGENT1], map_agent2action[self._AGENT2],
            map_agent2action[self._AGENT3], game.is_finished())

  def _game_overlay(self, game_env,
                    user_data: Exp1UserData) -> List[co.DrawingObject]:

    def coord_2_canvas(coord_x, coord_y):
      x = int(self.GAME_LEFT + coord_x * self.GAME_WIDTH)
      y = int(self.GAME_TOP + coord_y * self.GAME_HEIGHT)
      return (x, y)

    def size_2_canvas(width, height):
      w = int(width * self.GAME_WIDTH)
      h = int(height * self.GAME_HEIGHT)
      return (w, h)

    overlay_obs = []

    places = game_env["places"]  # type: Sequence[Place]
    routes = game_env["routes"]  # type: Sequence[Place]
    work_locations = game_env["work_locations"]

    if user_data.data[Exp1UserData.PARTIAL_OBS]:
      po_outer_ltwh = [
          self.GAME_LEFT, self.GAME_TOP, self.GAME_WIDTH, self.GAME_HEIGHT
      ]

      circles = []
      for place in places:
        if place.visible:
          for circle in RESCUE_V2_PLACE_DRAW_INFO[place.name].circles:
            cen_cnvs = coord_2_canvas(place.coord[0] + circle[0],
                                      place.coord[1] + circle[1])
            rad_cnvs = size_2_canvas(circle[2], 0)[0]
            circles.append((*cen_cnvs, rad_cnvs))

      a1_pos = game_env["a1_pos"]
      a1_coord = coord_2_canvas(*location_2_coord_v2(a1_pos, places, routes))
      radius = size_2_canvas(0.06, 0)[0]
      circles.append((*a1_coord, radius))

      obj = co.ClippedRectangle(co.PO_LAYER, po_outer_ltwh, list_circle=circles)
      overlay_obs.append(obj)

    if (user_data.data[Exp1UserData.SHOW_LATENT]
        and not user_data.data[Exp1UserData.SELECT]):
      a1_latent = game_env["a1_latent"]
      if a1_latent is not None:
        coord = location_2_coord_v2(work_locations[a1_latent], places, routes)
        if coord is not None:
          radius = size_2_canvas(0.05, 0)[0]
          x_cen = coord[0]
          y_cen = coord[1]
          obj = co.BlinkCircle(co.CUR_LATENT,
                               coord_2_canvas(x_cen, y_cen),
                               radius,
                               line_color="red",
                               fill=False,
                               border=True,
                               linewidth=3)
          overlay_obs.append(obj)

    if user_data.data[Exp1UserData.SELECT]:
      obj = co.Rectangle(co.SEL_LAYER, (self.GAME_LEFT, self.GAME_TOP),
                         (self.GAME_WIDTH, self.GAME_HEIGHT),
                         fill_color="white",
                         alpha=0.8)
      overlay_obs.append(obj)

      radius = size_2_canvas(0.05, 0)[0]
      font_size = 20

      for idx, loc in enumerate(work_locations):
        coord = location_2_coord_v2(loc, places, routes)
        obj = co.SelectingCircle(self.latent2selbtn(idx),
                                 coord_2_canvas(*coord), radius, font_size, "")
        overlay_obs.append(obj)

    return overlay_obs

  def _game_overlay_names(self, game_env, user_data: Exp1UserData) -> List:

    overlay_names = []
    work_locations = game_env["work_locations"]
    if user_data.data[Exp1UserData.PARTIAL_OBS]:
      overlay_names.append(co.PO_LAYER)

    if (user_data.data[Exp1UserData.SHOW_LATENT]
        and not user_data.data[Exp1UserData.SELECT]):
      a1_latent = game_env["a1_latent"]
      if a1_latent is not None:
        overlay_names.append(co.CUR_LATENT)

    if user_data.data[Exp1UserData.SELECT]:
      overlay_names.append(co.SEL_LAYER)

      for idx, loc in enumerate(work_locations):
        overlay_names.append(self.latent2selbtn(idx))

    return overlay_names

  def _game_scene(self,
                  game_env,
                  user_data: Exp1UserData,
                  include_background: bool = True) -> List[co.DrawingObject]:

    game_ltwh = (self.GAME_LEFT, self.GAME_TOP, self.GAME_WIDTH,
                 self.GAME_HEIGHT)
    return rescue_v2_game_scene(game_env, game_ltwh, include_background)

  def _game_scene_names(self, game_env, user_data: Exp1UserData) -> List:

    def is_visible(img_name):
      if user_data.data[Exp1UserData.PARTIAL_OBS]:
        if img_name == co.IMG_FIRE_ENGINE:
          a1_pos = game_env["a1_pos"]
          a2_pos = game_env["a2_pos"]
          if a1_pos == a2_pos:
            return True

          places = game_env["places"]  # type: Sequence[Place]
          for idx, place in enumerate(places):
            if place.visible and a2_pos == Location(E_Type.Place, idx):
              return True

          return False

        if img_name == co.IMG_AMBULANCE:
          a1_pos = game_env["a1_pos"]
          a3_pos = game_env["a3_pos"]
          if a1_pos == a3_pos:
            return True

          places = game_env["places"]  # type: Sequence[Place]
          for idx, place in enumerate(places):
            if place.visible and a3_pos == Location(E_Type.Place, idx):
              return True

          return False

      return True

    return rescue_v2_game_scene_names(game_env, is_visible)

  def latent2selbtn(self, latent):
    return "latent" + str(latent)

  def selbtn2latent(self, sel_btn_name):
    if sel_btn_name[:6] == "latent":
      return int(sel_btn_name[6:])

    return None

  def is_sel_latent_btn(self, sel_btn_name):
    return sel_btn_name[:6] == "latent"
