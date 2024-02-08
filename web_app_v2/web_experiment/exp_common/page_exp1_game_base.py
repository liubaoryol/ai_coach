from typing import Mapping, Any, Sequence, List
import copy
from aic_domain.box_push.simulator import BoxPushSimulator
from aic_domain.box_push import conv_box_idx_2_state, BoxState, EventType
from web_experiment.define import EDomainType
import web_experiment.exp_common.canvas_objects as co
from web_experiment.exp_common.page_base import Exp1UserData, ExperimentPageBase
from web_experiment.exp_common.helper import (boxpush_game_scene,
                                              boxpush_game_scene_names,
                                              get_btn_boxpush_actions,
                                              get_select_btn)

TEST_ROBOT_SIGHT = False


def get_holding_box_idx(box_states, num_drops, num_goals):
  a1_box = -1
  a2_box = -1
  for idx in range(len(box_states)):
    state = conv_box_idx_2_state(box_states[idx], num_drops, num_goals)
    if state[0] == BoxState.WithAgent1:  # with a1
      a1_box = idx
    elif state[0] == BoxState.WithAgent2:  # with a2
      a2_box = idx
    elif state[0] == BoxState.WithBoth:  # with both
      a1_box = idx
      a2_box = idx

  return a1_box, a2_box


def are_agent_states_changed(dict_prev_game: Mapping[str, Any],
                             dict_cur_game: Mapping[str, Any]):
  num_drops = len(dict_prev_game["drops"])
  num_goals = len(dict_prev_game["goals"])

  a1_pos_changed = False
  a2_pos_changed = False
  if dict_prev_game["a1_pos"] != dict_cur_game["a1_pos"]:
    a1_pos_changed = True

  if dict_prev_game["a2_pos"] != dict_cur_game["a2_pos"]:
    a2_pos_changed = True

  a1_box_prev, a2_box_prev = get_holding_box_idx(dict_prev_game["box_states"],
                                                 num_drops, num_goals)
  a1_box, a2_box = get_holding_box_idx(dict_cur_game["box_states"], num_drops,
                                       num_goals)

  a1_hold_changed = False
  a2_hold_changed = False

  if a1_box_prev != a1_box:
    a1_hold_changed = True

  if a2_box_prev != a2_box:
    a2_hold_changed = True

  return (a1_pos_changed, a2_pos_changed, a1_hold_changed, a2_hold_changed,
          a1_box, a2_box)


def get_valid_box_to_pickup(game: BoxPushSimulator):
  num_drops = len(game.drops)
  num_goals = len(game.goals)

  valid_box = []

  box_states = game.box_states
  for idx in range(len(box_states)):
    state = conv_box_idx_2_state(box_states[idx], num_drops, num_goals)
    if state[0] in [BoxState.Original, BoxState.OnDropLoc]:  # with a1
      valid_box.append(idx)

  return valid_box


###############################################################################
# canvas page game
###############################################################################
class BoxPushGamePageBase(ExperimentPageBase):
  def __init__(self,
               domain_type: EDomainType,
               game_map,
               latent_collection: bool = True) -> None:
    super().__init__(True, True, True, domain_type)
    self._GAME_MAP = game_map
    self._LATENT_COLLECTION = latent_collection

    # overwrite these flags at child classes
    self._MANUAL_SELECTION = latent_collection
    self._PROMPT_ON_CHANGE = latent_collection
    self._AUTO_PROMPT = latent_collection
    self._PROMPT_FREQ = 5

    self._AGENT1 = BoxPushSimulator.AGENT1
    self._AGENT2 = BoxPushSimulator.AGENT2

    assert domain_type in [EDomainType.Movers, EDomainType.Cleanup]

  def init_user_data(self, user_game_data: Exp1UserData):
    user_game_data.data[Exp1UserData.GAME_DONE] = False
    user_game_data.data[Exp1UserData.SELECT] = False
    # NOTE: game should be defined at the child class

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
      if clicked_button in co.ACTION_BUTTONS:
        animations = self._get_animations(dict_prev_scene_data,
                                          game.get_env_info())

    return commands, drawing_objs, animations

  def button_clicked(self, user_game_data: Exp1UserData, clicked_btn: str):
    '''
    user_game_data: NOTE - values will be updated
    return: commands, drawing_objs, drawing_order, animations
      drawing info
    '''

    if clicked_btn in co.ACTION_BUTTONS:
      game = user_game_data.get_game_ref()
      dict_prev_game = copy.deepcopy(game.get_env_info())
      a1_act, a2_act, done = self.action_event(user_game_data, clicked_btn)
      if done:
        self._on_game_finished(user_game_data)
      else:
        self._on_action_taken(user_game_data, dict_prev_game, (a1_act, a2_act))
      return

    elif clicked_btn == co.BTN_SELECT:
      user_game_data.data[Exp1UserData.SELECT] = True
      return

    elif self.is_sel_latent_btn(clicked_btn):
      latent = self.selbtn2latent(clicked_btn)
      if latent is not None:
        game = user_game_data.get_game_ref()
        game.event_input(self._AGENT1, EventType.SET_LATENT, latent)
        user_game_data.data[Exp1UserData.SELECT] = False
        user_game_data.data[Exp1UserData.ACTION_COUNT] = 0
        user_game_data.data[Exp1UserData.USER_LABELS].append(
            (game.current_step, latent))
        return

    elif clicked_btn == co.BTN_CONFIRM:
      user_game_data.data[Exp1UserData.DURING_INTERVENTION] = False
      return

    return super().button_clicked(user_game_data, clicked_btn)

  def _get_instruction(self, user_game_data: Exp1UserData):
    if user_game_data.data[Exp1UserData.SELECT]:
      return (
          "Please select your current destination among the circled options. " +
          "It can be the same destination as you had previously selected.")
    else:
      txt_inst = "Please choose your next action. "
      if self._LATENT_COLLECTION:
        txt_inst += (
            "You can only pick up or drop a box at the place circled in red. " +
            "If your destination has changed, " +
            "please update it using the \"Select Destination\" button.")

      return txt_inst

  def _get_drawing_order(self, user_game_data: Exp1UserData):
    dict_game = user_game_data.get_game_ref().get_env_info()

    drawing_order = super()._get_drawing_order(user_game_data)

    drawing_order = (drawing_order +
                     self._game_scene_names(dict_game, user_game_data))
    drawing_order = (drawing_order +
                     self._game_overlay_names(dict_game, user_game_data))
    drawing_order = drawing_order + co.ACTION_BUTTONS

    if self._LATENT_COLLECTION:
      drawing_order.append(co.BTN_SELECT)

    drawing_order.append(self.TEXT_SCORE)

    drawing_order.append(self.TEXT_INSTRUCTION)

    filtered_drawing_order = []
    for obj_name in drawing_order:
      if obj_name in user_game_data.data[Exp1UserData.DRAW_OBJ_NAMES]:
        filtered_drawing_order.append(obj_name)

    return filtered_drawing_order

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

    selecting = user_game_data.data[Exp1UserData.SELECT]
    select_disable = not self._MANUAL_SELECTION or selecting
    dis_status = self._get_action_btn_disabled(user_game_data)
    objs = self._get_btn_actions(*dis_status)
    for obj in objs:
      dict_objs[obj.name] = obj

    if self._LATENT_COLLECTION:
      obj = self._get_btn_select(select_disable)
      dict_objs[obj.name] = obj

    return dict_objs

  def _get_btn_actions(self,
                       up_disable=False,
                       down_disable=False,
                       left_disable=False,
                       right_disable=False,
                       stay_disable=False,
                       pickup_disable=False,
                       drop_disable=False):
    return get_btn_boxpush_actions(self.GAME_WIDTH, self.GAME_RIGHT, up_disable,
                                   down_disable, left_disable, right_disable,
                                   stay_disable, pickup_disable, drop_disable)

  def _get_btn_select(self, disable=False):
    return get_select_btn(self.GAME_WIDTH, self.GAME_RIGHT, disable)

  def _on_action_taken(self, user_game_data: Exp1UserData,
                       dict_prev_game: Mapping[str, Any],
                       tuple_actions: Sequence[Any]):
    '''
    user_cur_game_data: NOTE - values will be updated
    '''

    game = user_game_data.get_game_ref()
    # set selection prompt status
    (a1_pos_changed, a2_pos_changed, a1_hold_changed, a2_hold_changed, a1_box,
     a2_box) = are_agent_states_changed(dict_prev_game, game.get_env_info())

    select_latent = False
    if self._PROMPT_ON_CHANGE and a1_hold_changed:
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
    game = user_game_data.get_game_ref()

    # update score
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

    selecting = user_data.data[Exp1UserData.SELECT]
    select_disable = not self._MANUAL_SELECTION or selecting
    dis_status = self._get_action_btn_disabled(user_data)
    action_btns = self._get_btn_actions(*dis_status)
    for obj in action_btns:
      dict_objs[obj.name] = obj

    if self._LATENT_COLLECTION:
      obj = self._get_btn_select(select_disable)
      dict_objs[obj.name] = obj

    return dict_objs

  def _get_button_commands(self, clicked_btn, user_data: Exp1UserData):
    return None

  def _get_animations(self, dict_prev_game: Mapping[str, Any],
                      dict_cur_game: Mapping[str, Any]):
    (a1_pos_changed, a2_pos_changed, a1_hold_changed, a2_hold_changed, a1_box,
     a2_box) = are_agent_states_changed(dict_prev_game, dict_cur_game)

    unchanged_agents = []
    if not a1_pos_changed and not a1_hold_changed:
      unchanged_agents.append(0)
    if not a2_pos_changed and not a2_hold_changed:
      unchanged_agents.append(1)

    list_animations = []

    for agent_idx in unchanged_agents:
      obj_name = ""
      if agent_idx == 0:
        if a1_box < 0:
          obj_name = (co.IMG_WOMAN if self._DOMAIN_TYPE == EDomainType.Movers
                      else co.IMG_MAN)
        elif a1_box == a2_box:
          obj_name = co.IMG_BOTH_BOX
        else:
          obj_name = co.IMG_MAN_BAG
      else:
        if a2_box < 0:
          obj_name = co.IMG_ROBOT
        elif a1_box == a2_box:
          obj_name = co.IMG_BOTH_BOX
        else:
          obj_name = co.IMG_ROBOT_BAG

      amp = int(self.GAME_WIDTH / dict_cur_game["x_grid"] * 0.05)

      obj = {'type': 'vibrate', 'obj_name': obj_name, 'amplitude': amp}
      if obj not in list_animations:
        list_animations.append(obj)

    return list_animations

  def _get_action_btn_disabled(self, user_data: Exp1UserData):
    '''
    output order :
        btn_up, btn_down, btn_left, btn_right, btn_stay, btn_pickup, btn_drop
    '''
    game = user_data.get_game_ref()
    game_env = game.get_env_info()

    selecting = user_data.data[Exp1UserData.SELECT]
    game_done = user_data.data[Exp1UserData.GAME_DONE]
    intervention = user_data.data[Exp1UserData.DURING_INTERVENTION]

    if selecting or game_done or intervention:
      return True, True, True, True, True, True, True

    drop_ok = False
    pickup_ok = False
    num_drops = len(game_env["drops"])
    num_goals = len(game_env["goals"])
    a1_pos = game_env["a1_pos"]
    a1_box, _ = get_holding_box_idx(game_env["box_states"], num_drops,
                                    num_goals)

    if self._LATENT_COLLECTION:
      a1_latent = game_env["a1_latent"]
      if a1_latent is None:
        return False, False, False, False, False, True, True

      if a1_box >= 0:  # set drop action status
        if a1_latent[0] == 'origin' and a1_pos == game_env["boxes"][a1_box]:
          drop_ok = True
        else:
          for idx, coord in enumerate(game_env["goals"]):
            if (a1_latent[0] == 'goal' and a1_latent[1] == idx
                and a1_pos == coord):
              drop_ok = True
              break
      else:  # set pickup action status
        for idx, bidx in enumerate(game_env["box_states"]):
          state = conv_box_idx_2_state(bidx, num_drops, num_goals)
          coord = None
          if state[0] == BoxState.Original:
            coord = game_env["boxes"][idx]
          # elif state[0] == BoxState.WithAgent2:
          #   coord = game_env["a2_pos"]

          if coord is not None:
            if (a1_latent[0] == 'pickup' and a1_latent[1] == idx
                and a1_pos == coord):
              pickup_ok = True
              break

    else:
      if a1_box >= 0:  # set drop action status
        if a1_pos in game_env["goals"]:
          drop_ok = True
        elif a1_pos == game_env["boxes"][a1_box]:
          drop_ok = True
      else:  # set pickup action status
        for idx, bidx in enumerate(game_env["box_states"]):
          bstate = conv_box_idx_2_state(bidx, num_drops, num_goals)
          if bstate[0] == BoxState.Original:
            if game_env["boxes"][idx] == a1_pos:
              pickup_ok = True
          elif bstate[0] == BoxState.WithAgent2:
            if game_env["a2_pos"] == a1_pos:
              pickup_ok = True

    return False, False, False, False, False, not pickup_ok, not drop_ok

  def action_event(self, user_game_data: Exp1UserData, clicked_btn: str):
    '''
    user_game_data: NOTE - values will be updated
    '''
    action = None
    if clicked_btn == co.BTN_LEFT:
      action = EventType.LEFT
    elif clicked_btn == co.BTN_RIGHT:
      action = EventType.RIGHT
    elif clicked_btn == co.BTN_UP:
      action = EventType.UP
    elif clicked_btn == co.BTN_DOWN:
      action = EventType.DOWN
    elif clicked_btn == co.BTN_STAY:
      action = EventType.STAY
    elif clicked_btn == co.BTN_PICK_UP:
      action = EventType.HOLD
    elif clicked_btn == co.BTN_DROP:
      action = EventType.UNHOLD

    game = user_game_data.get_game_ref()
    # should not happen
    assert action is not None
    assert not game.is_finished()

    game.event_input(self._AGENT1, action, None)

    # take actions
    map_agent2action = game.get_joint_action()
    game.take_a_step(map_agent2action)

    return (map_agent2action[self._AGENT1], map_agent2action[self._AGENT2],
            game.is_finished())

  def _get_latent_pos_overlay(self, game_env):
    a1_latent = game_env["a1_latent"]
    if a1_latent is not None:
      coord = None
      if a1_latent[0] == "pickup":
        coord = game_env["boxes"][a1_latent[1]]
      elif a1_latent[0] == "goal":
        coord = game_env["goals"][a1_latent[1]]

      if coord is not None:
        x_grid = game_env["x_grid"]
        y_grid = game_env["y_grid"]
        x_cen = int(self.GAME_LEFT +
                    (coord[0] + 0.5) / x_grid * self.GAME_WIDTH)
        y_cen = int(self.GAME_TOP +
                    (coord[1] + 0.5) / y_grid * self.GAME_HEIGHT)
        radius = int(0.44 / x_grid * self.GAME_WIDTH)

        return (x_cen, y_cen), radius

    return None, None

  def _game_overlay(self, game_env,
                    user_data: Exp1UserData) -> List[co.DrawingObject]:

    x_grid = game_env["x_grid"]
    y_grid = game_env["y_grid"]

    def coord_2_canvas(coord_x, coord_y):
      x = int(self.GAME_LEFT + coord_x / x_grid * self.GAME_WIDTH)
      y = int(self.GAME_TOP + coord_y / y_grid * self.GAME_HEIGHT)
      return (x, y)

    def size_2_canvas(width, height):
      w = int(width / x_grid * self.GAME_WIDTH)
      h = int(height / y_grid * self.GAME_HEIGHT)
      return (w, h)

    num_drops = len(game_env["drops"])
    num_goals = len(game_env["goals"])
    a1_box, _ = get_holding_box_idx(game_env["box_states"], num_drops,
                                    num_goals)
    overlay_obs = []

    if user_data.data[Exp1UserData.PARTIAL_OBS]:
      po_outer_ltwh = [
          self.GAME_LEFT, self.GAME_TOP, self.GAME_WIDTH, self.GAME_HEIGHT
      ]
      cen_pos = game_env["a1_pos"]
      if TEST_ROBOT_SIGHT:
        cen_pos = game_env["a2_pos"]
      x_grid = game_env["x_grid"]
      y_grid = game_env["y_grid"]

      radius = size_2_canvas(3, 0)[0] / 10
      radii = [radius] * 4

      inner_left = max(0, cen_pos[0] - 1)
      inner_top = max(0, cen_pos[1] - 1)
      inner_right = min(cen_pos[0] + 2, x_grid)
      inner_bottom = min(cen_pos[1] + 2, y_grid)
      inner_width = inner_right - inner_left
      inner_height = inner_bottom - inner_top

      pos = coord_2_canvas(inner_left, inner_top)
      size = size_2_canvas(inner_width, inner_height)

      if cen_pos[0] - 1 < 0:
        radii[0] = 0
        radii[1] = 0
      if cen_pos[1] - 1 < 0:
        radii[0] = 0
        radii[3] = 0
      if cen_pos[0] + 2 > x_grid:
        radii[2] = 0
        radii[3] = 0
      if cen_pos[1] + 2 > y_grid:
        radii[1] = 0
        radii[2] = 0

      po_inner_ltwh = [pos[0], pos[1], size[0], size[1]]
      obj = co.ClippedRectangle(co.PO_LAYER,
                                po_outer_ltwh,
                                list_rect=[po_inner_ltwh],
                                list_rect_radii=[radii])
      overlay_obs.append(obj)

    if self._LATENT_COLLECTION and not user_data.data[Exp1UserData.SELECT]:
      center_pos, radius = self._get_latent_pos_overlay(game_env)
      if center_pos is not None:
        obj = co.BlinkCircle(co.CUR_LATENT,
                             center_pos,
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

      radius = size_2_canvas(0.45, 0)[0]
      font_size = 20

      for idx, coord in enumerate(game_env["goals"]):
        x_cen = coord[0] + 0.5
        y_cen = coord[1] + 0.5
        lat = ["goal", idx]
        obj = co.SelectingCircle(self.latent2selbtn(lat),
                                 coord_2_canvas(x_cen, y_cen), radius,
                                 font_size, "")
        overlay_obs.append(obj)

      for idx, bidx in enumerate(game_env["box_states"]):
        coord = game_env["boxes"][idx]

        x_cen = coord[0] + 0.5
        y_cen = coord[1] + 0.5
        lat = ["pickup", idx]
        obj = co.SelectingCircle(self.latent2selbtn(lat),
                                 coord_2_canvas(x_cen, y_cen), radius,
                                 font_size, "")
        overlay_obs.append(obj)

    if user_data.data[Exp1UserData.DURING_INTERVENTION]:
      obj = co.Rectangle(co.SEL_LAYER, (self.GAME_LEFT, self.GAME_TOP),
                         (self.GAME_WIDTH, self.GAME_HEIGHT),
                         fill_color="white",
                         alpha=0.8)
      overlay_obs.append(obj)

      radius = size_2_canvas(0.45, 0)[0]
      font_size = 20

      for idx, bidx in enumerate(game_env["box_states"]):
        if (idx == user_data.data[Exp1UserData.CUR_INFERENCE]):
          coord = game_env["boxes"][idx]

          x_cen = coord[0] + 0.5
          y_cen = coord[1] + 0.5
          obj = co.SelectingCircle("Confirm Intervention",
                                   coord_2_canvas(x_cen, y_cen), radius,
                                   font_size, "")
          overlay_obs.append(obj)

    return overlay_obs

  def _game_overlay_names(self, game_env, user_data: Exp1UserData) -> List:

    num_drops = len(game_env["drops"])
    num_goals = len(game_env["goals"])
    a1_box, _ = get_holding_box_idx(game_env["box_states"], num_drops,
                                    num_goals)
    overlay_names = []

    if user_data.data[Exp1UserData.PARTIAL_OBS]:
      overlay_names.append(co.PO_LAYER)

    if self._LATENT_COLLECTION and not user_data.data[Exp1UserData.SELECT]:
      a1_latent = game_env["a1_latent"]
      if a1_latent is not None:
        coord = None
        if a1_latent[0] == "pickup":
          coord = game_env["boxes"][a1_latent[1]]
        elif a1_latent[0] == "goal":
          coord = game_env["goals"][a1_latent[1]]

        if coord is not None:
          overlay_names.append(co.CUR_LATENT)

    if user_data.data[Exp1UserData.SELECT]:
      overlay_names.append(co.SEL_LAYER)

      for idx, coord in enumerate(game_env["goals"]):
        overlay_names.append(self.latent2selbtn(["goal", idx]))

      for idx, bidx in enumerate(game_env["box_states"]):
        overlay_names.append(self.latent2selbtn(["pickup", idx]))
    if user_data.data[Exp1UserData.DURING_INTERVENTION]:
      overlay_names.append(co.SEL_LAYER)
      for idx, bidx in enumerate(game_env["box_states"]):
        if idx == user_data.data[Exp1UserData.CUR_INFERENCE]:
          overlay_names.append("Confirm Intervention")
    return overlay_names

  def _game_scene(self,
                  game_env,
                  user_data: Exp1UserData,
                  include_background: bool = True) -> List[co.DrawingObject]:

    game_ltwh = (self.GAME_LEFT, self.GAME_TOP, self.GAME_WIDTH,
                 self.GAME_HEIGHT)
    is_movers = self._DOMAIN_TYPE == EDomainType.Movers
    return boxpush_game_scene(game_env, game_ltwh, is_movers,
                              include_background)

  def _game_scene_names(self, game_env, user_data: Exp1UserData) -> List:
    if TEST_ROBOT_SIGHT:

      def is_visible(img_name):
        if user_data.data[Exp1UserData.PARTIAL_OBS]:
          a2_pos = game_env["a2_pos"]
          if (img_name == co.IMG_WOMAN or img_name == co.IMG_MAN_BAG
              or img_name == co.IMG_MAN):
            a1_pos = game_env["a1_pos"]
            diff = max(abs(a1_pos[0] - a2_pos[0]), abs(a1_pos[1] - a2_pos[1]))
            if diff > 1:
              return False
          elif img_name[:-1] == co.IMG_BOX or img_name[:-1] == co.IMG_TRASH_BAG:
            bidx = int(img_name[-1])
            box_pos = game_env["boxes"][bidx]
            diff = max(abs(a2_pos[0] - box_pos[0]), abs(a2_pos[1] - box_pos[1]))
            if diff > 1:
              return False

        return True
    else:

      def is_visible(img_name):
        if user_data.data[Exp1UserData.PARTIAL_OBS]:
          a1_pos = game_env["a1_pos"]
          if img_name == co.IMG_ROBOT or img_name == co.IMG_ROBOT_BAG:
            a2_pos = game_env["a2_pos"]
            diff = max(abs(a1_pos[0] - a2_pos[0]), abs(a1_pos[1] - a2_pos[1]))
            if diff > 1:
              return False
          elif img_name[:-1] == co.IMG_BOX or img_name[:-1] == co.IMG_TRASH_BAG:
            bidx = int(img_name[-1])
            box_pos = game_env["boxes"][bidx]
            diff = max(abs(a1_pos[0] - box_pos[0]), abs(a1_pos[1] - box_pos[1]))
            if diff > 1:
              return False

        return True

    is_movers = self._DOMAIN_TYPE == EDomainType.Movers
    return boxpush_game_scene_names(game_env, is_movers, is_visible)

  def latent2selbtn(self, latent):
    if latent[0] == "pickup":
      return "sel_box" + str(latent[1])
    elif latent[0] == "goal":
      return "sel_goa" + str(latent[1])

    return None

  def selbtn2latent(self, sel_btn_name):
    if sel_btn_name[:7] == "sel_box":
      return ("pickup", int(sel_btn_name[7:]))
    elif sel_btn_name[:7] == "sel_goa":
      return ("goal", int(sel_btn_name[7:]))

    return None

  def is_sel_latent_btn(self, sel_btn_name):
    return sel_btn_name[:7] in ["sel_box", "sel_goa"]
