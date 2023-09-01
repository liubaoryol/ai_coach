import abc
from typing import Optional, List, Mapping, Set
from dataclasses import dataclass, field
import numpy as np
from aic_domain.box_push import conv_box_idx_2_state, BoxState
from aic_domain.box_push.simulator import BoxPushSimulator
import web_experiment.experiment1.canvas_objects as co


@dataclass
class GameFlags:
  select: bool
  done: bool
  action_count: int = 0
  clicked_btn: Set[str] = field(default_factory=set)
  aligned_a2_action: bool = False
  partial_obs: bool = False


class UserGameData:

  def __init__(self, user) -> None:
    self.game = None  # type: Optional[BoxPushSimulator]
    self.cur_page_idx = 0
    self.num_pages = 0
    self.session_name = ""
    self.user = user
    self.score = 0
    self.save_path = ""
    self.flags = GameFlags(False, False, 0)

  def go_to_next_page(self):
    if self.cur_page_idx + 1 < self.num_pages:
      self.cur_page_idx += 1

  def go_to_prev_page(self):
    if self.cur_page_idx - 1 >= 0:
      self.cur_page_idx -= 1


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


def get_objs_as_dictionary(drawing_objs: Mapping[str, co.DrawingObject]):
  return [obj.get_dictionary() for obj in drawing_objs.values()]


###############################################################################
# canvas page base
###############################################################################


class CanvasPageBase(abc.ABC):
  GAME_BORDER = "ls_line"
  TEXT_INSTRUCTION = "text_inst"
  RECT_INSTRUCTION = "rect_inst"
  TEXT_SCORE = "text_score"
  SPOTLIGHT = "Spotlight"

  GAME_LEFT = 0
  GAME_TOP = 0
  GAME_WIDTH = co.CANVAS_HEIGHT
  GAME_HEIGHT = co.CANVAS_HEIGHT
  GAME_RIGHT = GAME_LEFT + GAME_WIDTH
  GAME_BOTTOM = GAME_TOP + GAME_HEIGHT

  def __init__(self, show_border, show_instr, show_score, show_game,
               is_movers) -> None:
    self._SHOW_BORDER = show_border
    self._SHOW_INSTRUCTION = show_instr
    self._SHOW_SCORE = show_score
    self._SHOW_GAME = show_game
    self._IS_MOVERS = is_movers

  def init_user_data_and_get_drawing_info(self, user_game_data: UserGameData):
    '''
    user_game_data: NOTE - values will be updated
    return: commands, drawing_objs, drawing_order, animations
      drawing info
    '''

    self._init_user_data(user_game_data)

    dict_env = None
    score = user_game_data.score
    if user_game_data.game is not None:
      dict_env = user_game_data.game.get_env_info()
      score = dict_env["current_step"]

    if self._IS_MOVERS:
      best_score = user_game_data.user.best_a
    else:
      best_score = user_game_data.user.best_b

    drawing_objs = self._get_init_drawing_objects(dict_env,
                                                  user_game_data.flags, score,
                                                  best_score)
    drawing_order = self._get_drawing_order(dict_env, user_game_data.flags)
    commands = self._get_init_commands(dict_env, user_game_data.flags)

    animations = None

    return commands, drawing_objs, drawing_order, animations

  @abc.abstractmethod
  def _init_user_data(self, user_game_data: UserGameData):
    '''
    user_game_data: NOTE - values will be updated
    '''
    user_game_data.flags.done = False
    user_game_data.flags.aligned_a2_action = False

  @abc.abstractmethod
  def button_clicked(self, user_game_data: UserGameData, clicked_btn):
    '''
    user_game_data: NOTE - values will be updated
    return: commands, drawing_objs, drawing_order, animations
      drawing info
    '''

    if clicked_btn == co.BTN_NEXT:
      user_game_data.go_to_next_page()
      return None, None, None, None

    if clicked_btn == co.BTN_PREV:
      user_game_data.go_to_prev_page()
      return None, None, None, None

    return None, None, None, None

  def _get_instruction(self, flags: GameFlags):
    return ""

  @abc.abstractmethod
  def _get_drawing_order(self, game_env=None, flags: GameFlags = None):
    drawing_order = []
    if self._SHOW_BORDER:
      drawing_order.append(self.GAME_BORDER)

    return drawing_order

  def _get_init_drawing_objects(
      self,
      game_env=None,
      flags: GameFlags = None,
      score: int = 0,
      best_score: int = 9999) -> Mapping[str, co.DrawingObject]:
    dict_objs = {}
    if self._SHOW_BORDER:
      dict_objs[self.GAME_BORDER] = co.LineSegment(
          self.GAME_BORDER, (self.GAME_RIGHT, self.GAME_TOP),
          (self.GAME_RIGHT, self.GAME_BOTTOM))

    if self._SHOW_INSTRUCTION:
      for obj in self._get_instruction_objs(flags):
        dict_objs[obj.name] = obj

    if self._SHOW_SCORE:
      obj = self._get_score_obj(score, best_score)
      dict_objs[obj.name] = obj

    if self._SHOW_GAME and game_env is not None:
      game_objs = self._game_scene(game_env,
                                   self._IS_MOVERS,
                                   flags,
                                   include_background=True)
      for obj in game_objs:
        dict_objs[obj.name] = obj

      overlay_objs = self._game_overlay(game_env, flags, not flags.select)
      for obj in overlay_objs:
        dict_objs[obj.name] = obj

    return dict_objs

  def _get_init_commands(self, game_env=None, flags: GameFlags = None):
    return {"clear": None}

  def _game_overlay(self, game_env, flags: GameFlags,
                    show_latent: bool) -> List[co.DrawingObject]:
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

    if flags.partial_obs:
      po_outer_ltwh = [
          self.GAME_LEFT, self.GAME_TOP, self.GAME_WIDTH, self.GAME_HEIGHT
      ]
      a1_pos = game_env["a1_pos"]
      x_grid = game_env["x_grid"]
      y_grid = game_env["y_grid"]
      inner_left = max(0, a1_pos[0] - 1)
      inner_top = max(0, a1_pos[1] - 1)
      inner_right = min(a1_pos[0] + 2, x_grid)
      inner_bottom = min(a1_pos[1] + 2, y_grid)
      inner_width = inner_right - inner_left
      inner_height = inner_bottom - inner_top

      pos = coord_2_canvas(inner_left, inner_top)
      size = size_2_canvas(inner_width, inner_height)
      po_inner_ltwh = [pos[0], pos[1], size[0], size[1]]
      obj = co.RectSpotlight(co.PO_LAYER,
                             po_outer_ltwh,
                             po_inner_ltwh,
                             alpha=0.3)
      overlay_obs.append(obj)

    if show_latent:
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

    if flags.select:
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

  def _game_overlay_names(self, game_env, flags: GameFlags,
                          show_latent: bool) -> List:
    num_drops = len(game_env["drops"])
    num_goals = len(game_env["goals"])
    a1_box, _ = get_holding_box_idx(game_env["box_states"], num_drops,
                                    num_goals)
    overlay_names = []

    if flags.partial_obs:
      overlay_names.append(co.PO_LAYER)

    if show_latent:
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

    if flags.select:
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
                  is_movers,
                  flags: GameFlags,
                  include_background: bool = True) -> List[co.DrawingObject]:
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

    game_objs = []
    if include_background:
      for idx, coord in enumerate(game_env["boxes"]):
        game_pos = coord_2_canvas(coord[0] + 0.5, coord[1] + 0.7)
        size = size_2_canvas(0.4, 0.2)
        obj = co.Ellipse(co.BOX_ORIGIN + str(idx), game_pos, size, "grey")
        game_objs.append(obj)

      for idx, coord in enumerate(game_env["walls"]):
        wid = 1.4
        hei = 1
        left = coord[0] + 0.5 - 0.5 * wid
        top = coord[1] + 0.5 - 0.5 * hei
        angle = 0 if game_env["wall_dir"][idx] == 0 else 0.5 * np.pi
        obj = co.GameObject(co.IMG_WALL + str(idx), coord_2_canvas(left, top),
                            size_2_canvas(wid, hei), angle, co.IMG_WALL)
        game_objs.append(obj)

      for idx, coord in enumerate(game_env["goals"]):
        hei = 0.8
        wid = 0.724
        left = coord[0] + 0.5 - 0.5 * wid
        top = coord[1] + 0.5 - 0.5 * hei
        obj = co.GameObject(co.IMG_GOAL + str(idx), coord_2_canvas(left, top),
                            size_2_canvas(wid, hei), 0, co.IMG_GOAL)
        game_objs.append(obj)

    num_drops = len(game_env["drops"])
    num_goals = len(game_env["goals"])
    a1_hold_box = -1
    a2_hold_box = -1
    for idx, bidx in enumerate(game_env["box_states"]):
      state = conv_box_idx_2_state(bidx, num_drops, num_goals)
      obj = None
      if state[0] == BoxState.Original:
        coord = game_env["boxes"][idx]
        wid = 0.541
        hei = 0.60
        left = coord[0] + 0.5 - 0.5 * wid
        top = coord[1] + 0.5 - 0.5 * hei
        img_name = co.IMG_BOX if is_movers else co.IMG_TRASH_BAG
        obj = co.GameObject(img_name + str(idx), coord_2_canvas(left, top),
                            size_2_canvas(wid, hei), 0, img_name)
      elif state[0] == BoxState.WithAgent1:  # with a1
        coord = game_env["a1_pos"]
        offset = 0
        if game_env["a2_pos"] == coord:
          offset = -0.2
        hei = 1
        wid = 0.385
        left = coord[0] + 0.5 + offset - 0.5 * wid
        top = coord[1] + 0.5 - 0.5 * hei
        obj = co.GameObject(co.IMG_MAN_BAG, coord_2_canvas(left, top),
                            size_2_canvas(wid, hei), 0, co.IMG_MAN_BAG)
        a1_hold_box = idx
      elif state[0] == BoxState.WithAgent2:  # with a2
        coord = game_env["a2_pos"]
        offset = 0
        if game_env["a1_pos"] == coord:
          offset = -0.2
        hei = 0.8
        wid = 0.476
        left = coord[0] + 0.5 + offset - 0.5 * wid
        top = coord[1] + 0.5 - 0.5 * hei
        obj = co.GameObject(co.IMG_ROBOT_BAG, coord_2_canvas(left, top),
                            size_2_canvas(wid, hei), 0, co.IMG_ROBOT_BAG)
        a2_hold_box = idx
      elif state[0] == BoxState.WithBoth:  # with both
        coord = game_env["a1_pos"]
        hei = 1
        wid = 0.712
        left = coord[0] + 0.5 - 0.5 * wid
        top = coord[1] + 0.5 - 0.5 * hei
        obj = co.GameObject(co.IMG_BOTH_BOX, coord_2_canvas(left, top),
                            size_2_canvas(wid, hei), 0, co.IMG_BOTH_BOX)
        a1_hold_box = idx
        a2_hold_box = idx

      if obj is not None:
        game_objs.append(obj)

    if a1_hold_box < 0:
      coord = game_env["a1_pos"]
      offset = 0
      if coord == game_env["a2_pos"]:
        offset = 0.2 if a2_hold_box >= 0 else -0.2
      hei = 1
      wid = 0.23
      left = coord[0] + 0.5 + offset - 0.5 * wid
      top = coord[1] + 0.5 - 0.5 * hei
      img_name = co.IMG_WOMAN if is_movers else co.IMG_MAN
      obj = co.GameObject(img_name, coord_2_canvas(left, top),
                          size_2_canvas(wid, hei), 0, img_name)
      game_objs.append(obj)

    if a2_hold_box < 0:
      coord = game_env["a2_pos"]
      offset = 0
      if coord == game_env["a1_pos"]:
        offset = 0.2
      hei = 0.8
      wid = 0.422
      left = coord[0] + 0.5 + offset - 0.5 * wid
      top = coord[1] + 0.5 - 0.5 * hei
      obj = co.GameObject(co.IMG_ROBOT, coord_2_canvas(left, top),
                          size_2_canvas(wid, hei), 0, co.IMG_ROBOT)
      game_objs.append(obj)

    # if flags.partial_obs:
    #   a1_pos = game_env["a1_pos"]
    #   a2_pos = game_env["a2_pos"]
    #   diff = max(abs(a1_pos[0] - a2_pos[0]), abs(a1_pos[1] - a2_pos[1]))
    #   if diff > 1:
    #     new_game_obj = []
    #     for obj in game_objs:
    #       if obj.name != co.IMG_ROBOT and obj.name != co.IMG_ROBOT_BAG:
    #         new_game_obj.append(obj)
    #     game_objs = new_game_obj

    return game_objs

  def _game_scene_names(self, game_env, is_movers, flags: GameFlags) -> List:
    drawing_names = []
    for idx, _ in enumerate(game_env["boxes"]):
      drawing_names.append(co.BOX_ORIGIN + str(idx))

    for idx, _ in enumerate(game_env["walls"]):
      drawing_names.append(co.IMG_WALL + str(idx))

    for idx, _ in enumerate(game_env["goals"]):
      drawing_names.append(co.IMG_GOAL + str(idx), )

    num_drops = len(game_env["drops"])
    num_goals = len(game_env["goals"])
    a1_hold_box = -1
    a2_hold_box = -1
    for idx, bidx in enumerate(game_env["box_states"]):
      state = conv_box_idx_2_state(bidx, num_drops, num_goals)
      img_name = None
      if state[0] == BoxState.Original:
        img_type = co.IMG_BOX if is_movers else co.IMG_TRASH_BAG
        img_name = img_type + str(idx)
      elif state[0] == BoxState.WithAgent1:  # with a1
        img_name = co.IMG_MAN_BAG
        a1_hold_box = idx
      elif state[0] == BoxState.WithAgent2:  # with a2
        img_name = co.IMG_ROBOT_BAG
        a2_hold_box = idx
      elif state[0] == BoxState.WithBoth:  # with both
        img_name = co.IMG_BOTH_BOX
        a1_hold_box = idx
        a2_hold_box = idx

      if img_name is not None:
        drawing_names.append(img_name)

    if a1_hold_box < 0:
      img_name = co.IMG_WOMAN if is_movers else co.IMG_MAN
      drawing_names.append(img_name)

    if a2_hold_box < 0:
      drawing_names.append(co.IMG_ROBOT)

    if flags.partial_obs:
      a1_pos = game_env["a1_pos"]
      a2_pos = game_env["a2_pos"]
      diff = max(abs(a1_pos[0] - a2_pos[0]), abs(a1_pos[1] - a2_pos[1]))
      if diff > 1:
        new_drawing_names = []
        for obj in drawing_names:
          if obj != co.IMG_ROBOT and obj != co.IMG_ROBOT_BAG:
            new_drawing_names.append(obj)
        drawing_names = new_drawing_names

    return drawing_names

  def _get_btn_prev_next(self, prev_disable=True, next_disable=True):
    margin = 10
    font_size = 18
    width = int((co.CANVAS_WIDTH - self.GAME_RIGHT) / 4)
    height = int(width * 0.5)
    pos_next = (int(co.CANVAS_WIDTH - width * 0.5 - margin),
                int(co.CANVAS_HEIGHT * 0.5 - 0.5 * height - margin))
    pos_prev = (int(self.GAME_RIGHT + width * 0.5 + margin),
                int(co.CANVAS_HEIGHT * 0.5 - 0.5 * height - margin))
    size = (width, height)
    btn_next = co.ButtonRect(co.BTN_NEXT,
                             pos_next,
                             size,
                             font_size,
                             "Next",
                             disable=next_disable)
    btn_prev = co.ButtonRect(co.BTN_PREV,
                             pos_prev,
                             size,
                             font_size,
                             "Prev",
                             disable=prev_disable)
    return btn_prev, btn_next

  def _get_btn_start(self, start_disable=True):
    start_btn_width = int(self.GAME_WIDTH / 3)
    start_btn_height = int(self.GAME_HEIGHT / 10)
    x_cen = int(self.GAME_LEFT + self.GAME_WIDTH / 2)
    y_cen = int(self.GAME_TOP + self.GAME_HEIGHT / 2)

    start_btn_obj = co.ButtonRect(co.BTN_START, (x_cen, y_cen),
                                  (start_btn_width, start_btn_height),
                                  30,
                                  "Start",
                                  disable=start_disable)
    return start_btn_obj

  def _get_spotlight(self, x_cen, y_cen, radius):
    outer_ltwh = (0, 0, co.CANVAS_WIDTH, co.CANVAS_HEIGHT)
    return co.CircleSpotlight(self.SPOTLIGHT, outer_ltwh, (x_cen, y_cen),
                              radius)

  def _get_btn_actions(self,
                       up_disable=False,
                       down_disable=False,
                       left_disable=False,
                       right_disable=False,
                       stay_disable=False,
                       pickup_disable=False,
                       drop_disable=False,
                       select_disable=True):
    ctrl_btn_w = int(self.GAME_WIDTH / 12)
    ctrl_btn_w_half = int(self.GAME_WIDTH / 24)
    x_ctrl_cen = int(self.GAME_RIGHT + (co.CANVAS_WIDTH - self.GAME_RIGHT) / 2)
    y_ctrl_cen = int(co.CANVAS_HEIGHT * 0.65)
    x_joy_cen = int(x_ctrl_cen - ctrl_btn_w * 1.5)
    btn_stay = co.JoystickStay((x_joy_cen, y_ctrl_cen),
                               ctrl_btn_w,
                               disable=stay_disable)
    btn_up = co.JoystickUp((x_joy_cen, y_ctrl_cen - ctrl_btn_w_half),
                           ctrl_btn_w,
                           disable=up_disable)
    btn_right = co.JoystickRight((x_joy_cen + ctrl_btn_w_half, y_ctrl_cen),
                                 ctrl_btn_w,
                                 disable=right_disable)
    btn_down = co.JoystickDown((x_joy_cen, y_ctrl_cen + ctrl_btn_w_half),
                               ctrl_btn_w,
                               disable=down_disable)
    btn_left = co.JoystickLeft((x_joy_cen - ctrl_btn_w_half, y_ctrl_cen),
                               ctrl_btn_w,
                               disable=left_disable)
    font_size = 20
    btn_pickup = co.ButtonRect(co.BTN_PICK_UP,
                               (x_ctrl_cen + int(ctrl_btn_w * 1.5),
                                y_ctrl_cen - int(ctrl_btn_w * 0.6)),
                               (ctrl_btn_w * 2, ctrl_btn_w),
                               font_size,
                               "Pick Up",
                               disable=pickup_disable)
    btn_drop = co.ButtonRect(co.BTN_DROP, (x_ctrl_cen + int(ctrl_btn_w * 1.5),
                                           y_ctrl_cen + int(ctrl_btn_w * 0.6)),
                             (ctrl_btn_w * 2, ctrl_btn_w),
                             font_size,
                             "Drop",
                             disable=drop_disable)
    btn_select = co.ButtonRect(co.BTN_SELECT,
                               (x_ctrl_cen, y_ctrl_cen + ctrl_btn_w * 2),
                               (ctrl_btn_w * 4, ctrl_btn_w),
                               font_size,
                               "Select Destination",
                               disable=select_disable)
    return (btn_up, btn_down, btn_left, btn_right, btn_stay, btn_pickup,
            btn_drop, btn_select)

  def _get_instruction_objs(self, flags: GameFlags):
    margin = 10
    pos = (self.GAME_RIGHT + margin, margin)
    width = co.CANVAS_WIDTH - pos[0] - margin
    text_instr = co.TextObject(self.TEXT_INSTRUCTION, pos, width, 18,
                               self._get_instruction(flags))

    margin = 5
    pos = (self.GAME_RIGHT + margin, margin)
    size = (co.CANVAS_WIDTH - pos[0] - margin, int(self.GAME_HEIGHT * 0.5))
    rect_instr = co.Rectangle(self.RECT_INSTRUCTION, pos, size, "white")

    return text_instr, rect_instr

  def _get_score_obj(self, score, best_score):
    margin = 10
    text_score = "Time Taken: " + str(score) + "\n"
    if best_score == 9999:
      text_score += "(Your Best: - )"
    else:
      text_score += "(Your Best: " + str(best_score) + ")"
    return co.TextObject(
        self.TEXT_SCORE,
        (self.GAME_RIGHT + margin, int(co.CANVAS_HEIGHT * 0.9)),
        co.CANVAS_WIDTH - self.GAME_RIGHT - 2 * margin,
        24,
        text_score,
        text_align="right")
