from typing import Mapping, Any, Sequence, List
import copy
import os
import time
from ai_coach_domain.box_push.simulator import (BoxPushSimulator_AlwaysTogether,
                                                BoxPushSimulator_AlwaysAlone,
                                                BoxPushSimulator)
from ai_coach_domain.box_push import conv_box_idx_2_state, BoxState, EventType
from web_experiment.models import db, User
import web_experiment.exp_common.canvas_objects as co
from web_experiment.exp_common.page_exp1_base import (Exp1UserData,
                                                      Exp1PageBase)
from web_experiment.exp_common.page_game_scene import (game_scene,
                                                       game_scene_names)


def get_file_name(save_path, user_id, session_name):
  traj_dir = os.path.join(save_path, user_id)
  # save somewhere
  if not os.path.exists(traj_dir):
    os.makedirs(traj_dir)

  sec, msec = divmod(time.time() * 1000, 1000)
  time_stamp = '%s.%03d' % (time.strftime('%Y-%m-%d_%H_%M_%S',
                                          time.gmtime(sec)), msec)
  file_name = session_name + '_' + str(user_id) + '_' + time_stamp + '.txt'
  return os.path.join(traj_dir, file_name)


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


class Exp1PageGame(Exp1PageBase):
  def __init__(self,
               is_movers,
               manual_latent_selection,
               game_map,
               auto_prompt: bool = True,
               prompt_on_change: bool = True,
               prompt_freq: int = 5) -> None:
    super().__init__(True, True, True, is_movers)
    self._MANUAL_SELECTION = manual_latent_selection
    self._GAME_MAP = game_map

    self._PROMPT_ON_CHANGE = prompt_on_change
    self._PROMPT_FREQ = prompt_freq
    self._AUTO_PROMPT = auto_prompt

    self._AGENT1 = BoxPushSimulator.AGENT1
    self._AGENT2 = BoxPushSimulator.AGENT2

  def init_user_data(self, user_game_data: Exp1UserData):
    user_game_data.data[Exp1UserData.GAME_DONE] = False
    user_game_data.data[Exp1UserData.SELECT] = False

    game = user_game_data.get_game_ref()
    if game is None:
      if self._IS_MOVERS:
        game = BoxPushSimulator_AlwaysTogether(None)
      else:
        game = BoxPushSimulator_AlwaysAlone(None)

      user_game_data.set_game(game)

    game.init_game(**self._GAME_MAP)

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
      if clicked_button in co.ACTION_BUTTONS:
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

    elif co.is_sel_latent_btn(clicked_btn):
      latent = co.selbtn2latent(clicked_btn)
      if latent is not None:
        game = user_game_data.get_game_ref()
        game.event_input(self._AGENT1, EventType.SET_LATENT, latent)
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
    drawing_order = drawing_order + co.ACTION_BUTTONS
    drawing_order.append(co.BTN_SELECT)

    drawing_order.append(self.TEXT_SCORE)

    drawing_order.append(self.RECT_INSTRUCTION)
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

    selecting = user_game_data.data[Exp1UserData.SELECT]
    select_disable = not self._MANUAL_SELECTION or selecting
    dis_status = self._get_action_btn_disabled(user_game_data)
    objs = self._get_btn_actions(*dis_status, select_disable=select_disable)
    for obj in objs:
      dict_objs[obj.name] = obj

    return dict_objs

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
    if self._PROMPT_ON_CHANGE and (a1_hold_changed or a2_hold_changed):
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
    user = user_game_data.data[Exp1UserData.USER]
    user_id = user.userid

    # save trajectory
    save_path = user_game_data.data[Exp1UserData.SAVE_PATH]
    session_name = user_game_data.data[Exp1UserData.SESSION_NAME]
    file_name = get_file_name(save_path, user_id, session_name)
    header = game.__class__.__name__ + "-" + session_name + "\n"
    header += "User ID: %s\n" % (str(user_id), )
    header += str(self._GAME_MAP)
    game.save_history(file_name, header)

    # update score
    user_game_data.data[Exp1UserData.SCORE] = game.current_step
    if self._IS_MOVERS:
      best_score = user.best_a
    else:
      best_score = user.best_b

    if best_score > game.current_step:
      user = User.query.filter_by(userid=user_id).first()
      if self._IS_MOVERS:
        user.best_a = game.current_step
      else:
        user.best_b = game.current_step

      db.session.commit()
      user_game_data.data[Exp1UserData.USER] = user

    # move to next page
    user_game_data.go_to_next_page()

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

    obj = self._get_instruction_objs(user_data)[0]
    dict_objs[obj.name] = obj

    selecting = user_data.data[Exp1UserData.SELECT]
    select_disable = not self._MANUAL_SELECTION or selecting
    dis_status = self._get_action_btn_disabled(user_data)
    action_btns = self._get_btn_actions(*dis_status,
                                        select_disable=select_disable)
    for obj in action_btns:
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
          obj_name = co.IMG_WOMAN if self._IS_MOVERS else co.IMG_MAN
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

    if selecting or game_done:
      return True, True, True, True, True, True, True

    drop_ok = False
    pickup_ok = False
    num_drops = len(game_env["drops"])
    num_goals = len(game_env["goals"])
    a1_box, _ = get_holding_box_idx(game_env["box_states"], num_drops,
                                    num_goals)
    if a1_box >= 0:  # set drop action status
      drop_ok = True
    else:  # set pickup action status
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
      a1_pos = game_env["a1_pos"]
      x_grid = game_env["x_grid"]
      y_grid = game_env["y_grid"]

      radius = size_2_canvas(3, 0)[0] / 10
      radii = [radius] * 4

      inner_left = max(0, a1_pos[0] - 1)
      inner_top = max(0, a1_pos[1] - 1)
      inner_right = min(a1_pos[0] + 2, x_grid)
      inner_bottom = min(a1_pos[1] + 2, y_grid)
      inner_width = inner_right - inner_left
      inner_height = inner_bottom - inner_top

      pos = coord_2_canvas(inner_left, inner_top)
      size = size_2_canvas(inner_width, inner_height)

      if a1_pos[0] - 1 < 0:
        radii[0] = 0
        radii[1] = 0
      if a1_pos[1] - 1 < 0:
        radii[0] = 0
        radii[3] = 0
      if a1_pos[0] + 2 > x_grid:
        radii[2] = 0
        radii[3] = 0
      if a1_pos[1] + 2 > y_grid:
        radii[1] = 0
        radii[2] = 0

      po_inner_ltwh = [pos[0], pos[1], size[0], size[1]]
      obj = co.RectSpotlight(co.PO_LAYER,
                             po_outer_ltwh,
                             po_inner_ltwh,
                             radii=radii,
                             alpha=0.3)
      overlay_obs.append(obj)

    if (user_data.data[Exp1UserData.SHOW_LATENT]
        and not user_data.data[Exp1UserData.SELECT]):
      a1_latent = game_env["a1_latent"]
      if a1_latent is not None:
        coord = None
        if a1_latent[0] == "pickup":
          bidx = game_env["box_states"][a1_latent[1]]
          state = conv_box_idx_2_state(bidx, num_drops, num_goals)
          if state[0] == BoxState.Original:
            coord = game_env["boxes"][a1_latent[1]]
          elif state[0] == BoxState.WithAgent2:
            coord = game_env["a2_pos"]
        elif a1_latent[0] == "origin":
          if a1_box >= 0:
            coord = game_env["boxes"][a1_box]
        elif a1_latent[0] == "goal":
          coord = game_env["goals"][a1_latent[1]]

        if coord is not None:
          radius = size_2_canvas(0.44, 0)[0]
          x_cen = coord[0] + 0.5
          y_cen = coord[1] + 0.5
          obj = co.Circle(co.CUR_LATENT,
                          coord_2_canvas(x_cen, y_cen),
                          radius,
                          line_color="red",
                          fill=False,
                          border=True)
          overlay_obs.append(obj)

    if user_data.data[Exp1UserData.SELECT]:
      obj = co.Rectangle(co.SEL_LAYER,
                         coord_2_canvas(self.GAME_LEFT, self.GAME_TOP),
                         coord_2_canvas(self.GAME_WIDTH, self.GAME_HEIGHT),
                         fill_color="white",
                         alpha=0.8)
      overlay_obs.append(obj)

      cnt = 0
      radius = size_2_canvas(0.45, 0)[0]
      font_size = 20
      if a1_box >= 0:

        coord = game_env["boxes"][a1_box]
        x_cen = coord[0] + 0.5
        y_cen = coord[1] + 0.5
        lat = ["origin", 0]
        obj = co.SelectingCircle(co.latent2selbtn(lat),
                                 coord_2_canvas(x_cen, y_cen), radius,
                                 font_size, str(cnt))
        overlay_obs.append(obj)
        cnt += 1

        for idx, coord in enumerate(game_env["goals"]):
          x_cen = coord[0] + 0.5
          y_cen = coord[1] + 0.5
          lat = ["goal", idx]
          obj = co.SelectingCircle(co.latent2selbtn(lat),
                                   coord_2_canvas(x_cen, y_cen), radius,
                                   font_size, str(cnt))
          overlay_obs.append(obj)
          cnt += 1
      else:
        for idx, bidx in enumerate(game_env["box_states"]):
          state = conv_box_idx_2_state(bidx, num_drops, num_goals)
          coord = None
          if state[0] == BoxState.Original:
            coord = game_env["boxes"][idx]
          elif state[0] == BoxState.WithAgent2:  # with a2
            coord = game_env["a2_pos"]

          if coord is not None:
            x_cen = coord[0] + 0.5
            y_cen = coord[1] + 0.5
            lat = ["pickup", idx]
            obj = co.SelectingCircle(co.latent2selbtn(lat),
                                     coord_2_canvas(x_cen, y_cen), radius,
                                     font_size, str(cnt))
            overlay_obs.append(obj)
            cnt += 1

    return overlay_obs

  def _game_overlay_names(self, game_env, user_data: Exp1UserData) -> List:

    num_drops = len(game_env["drops"])
    num_goals = len(game_env["goals"])
    a1_box, _ = get_holding_box_idx(game_env["box_states"], num_drops,
                                    num_goals)
    overlay_names = []

    if user_data.data[Exp1UserData.PARTIAL_OBS]:
      overlay_names.append(co.PO_LAYER)

    if not user_data.data[Exp1UserData.SELECT]:
      a1_latent = game_env["a1_latent"]
      if a1_latent is not None:
        coord = None
        if a1_latent[0] == "pickup":
          bidx = game_env["box_states"][a1_latent[1]]
          state = conv_box_idx_2_state(bidx, num_drops, num_goals)
          if state[0] == BoxState.Original:
            coord = game_env["boxes"][a1_latent[1]]
          elif state[0] == BoxState.WithAgent2:
            coord = game_env["a2_pos"]
        elif a1_latent[0] == "origin":
          if a1_box >= 0:
            coord = game_env["boxes"][a1_box]
        elif a1_latent[0] == "goal":
          coord = game_env["goals"][a1_latent[1]]

        if coord is not None:
          overlay_names.append(co.CUR_LATENT)

    if user_data.data[Exp1UserData.SELECT]:
      overlay_names.append(co.SEL_LAYER)
      if a1_box >= 0:
        overlay_names.append(co.latent2selbtn(["origin", 0]))

        for idx, coord in enumerate(game_env["goals"]):
          overlay_names.append(co.latent2selbtn(["goal", idx]))
      else:
        for idx, bidx in enumerate(game_env["box_states"]):
          state = conv_box_idx_2_state(bidx, num_drops, num_goals)
          coord = None
          if state[0] == BoxState.Original:
            coord = game_env["boxes"][idx]
          elif state[0] == BoxState.WithAgent2:  # with a2
            coord = game_env["a2_pos"]

          if coord is not None:
            overlay_names.append(co.latent2selbtn(["pickup", idx]))

    return overlay_names

  def _game_scene(self,
                  game_env,
                  user_data: Exp1UserData,
                  include_background: bool = True) -> List[co.DrawingObject]:

    game_ltwh = (self.GAME_LEFT, self.GAME_TOP, self.GAME_WIDTH,
                 self.GAME_HEIGHT)
    return game_scene(game_env, game_ltwh, self._IS_MOVERS, include_background)

  def _game_scene_names(self, game_env, user_data: Exp1UserData) -> List:
    def is_visible(img_name):
      if user_data.data[Exp1UserData.PARTIAL_OBS]:
        if img_name == co.IMG_ROBOT or img_name == co.IMG_ROBOT_BAG:
          a1_pos = game_env["a1_pos"]
          a2_pos = game_env["a2_pos"]
          diff = max(abs(a1_pos[0] - a2_pos[0]), abs(a1_pos[1] - a2_pos[1]))
          if diff > 1:
            return False

      return True

    return game_scene_names(game_env, self._IS_MOVERS, is_visible)