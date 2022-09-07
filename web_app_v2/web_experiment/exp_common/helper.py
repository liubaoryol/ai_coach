from typing import Mapping, Any, List, Tuple, Callable, Sequence
import os
import time
import numpy as np
from ai_coach_domain.box_push import conv_box_idx_2_state, BoxState
from ai_coach_domain.rescue import Place, Route, Location, E_Type
import web_experiment.exp_common.canvas_objects as co


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


def get_btn_boxpush_actions(game_width,
                            game_right,
                            up_disable=False,
                            down_disable=False,
                            left_disable=False,
                            right_disable=False,
                            stay_disable=False,
                            pickup_disable=False,
                            drop_disable=False,
                            select_disable=True):
  ctrl_btn_w = int(game_width / 12)
  ctrl_btn_w_half = int(game_width / 24)
  x_ctrl_cen = int(game_right + (co.CANVAS_WIDTH - game_right) / 2)
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
  btn_pickup = co.ButtonRect(
      co.BTN_PICK_UP,
      (x_ctrl_cen + int(ctrl_btn_w * 1.5), y_ctrl_cen - int(ctrl_btn_w * 0.6)),
      (ctrl_btn_w * 2, ctrl_btn_w),
      font_size,
      "Pick Up",
      disable=pickup_disable)
  btn_drop = co.ButtonRect(
      co.BTN_DROP,
      (x_ctrl_cen + int(ctrl_btn_w * 1.5), y_ctrl_cen + int(ctrl_btn_w * 0.6)),
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
  return (btn_up, btn_down, btn_left, btn_right, btn_stay, btn_pickup, btn_drop,
          btn_select)


def boxpush_game_scene(
    game_env: Mapping[str, Any],
    game_lwth: Tuple[int, int, int, int],
    is_movers: bool,
    include_background: bool = True,
    cb_is_visible: Callable[[co.DrawingObject], bool] = None
) -> List[co.DrawingObject]:
  x_grid = game_env["x_grid"]
  y_grid = game_env["y_grid"]

  game_left, game_top, game_width, game_height = game_lwth

  def coord_2_canvas(coord_x, coord_y):
    x = int(game_left + coord_x / x_grid * game_width)
    y = int(game_top + coord_y / y_grid * game_height)
    return (x, y)

  def size_2_canvas(width, height):
    w = int(width / x_grid * game_width)
    h = int(height / y_grid * game_height)
    return (w, h)

  game_objs = []
  if include_background:
    for idx, coord in enumerate(game_env["boxes"]):
      game_pos = coord_2_canvas(coord[0] + 0.5, coord[1] + 0.7)
      size = size_2_canvas(0.4, 0.2)
      obj = co.Ellipse(co.BOX_ORIGIN + str(idx), game_pos, size, "grey")
      if cb_is_visible is None or cb_is_visible(obj):
        game_objs.append(obj)

    for idx, coord in enumerate(game_env["walls"]):
      wid = 1.4
      hei = 1
      left = coord[0] + 0.5 - 0.5 * wid
      top = coord[1] + 0.5 - 0.5 * hei
      angle = 0 if game_env["wall_dir"][idx] == 0 else 0.5 * np.pi
      obj = co.GameObject(co.IMG_WALL + str(idx), coord_2_canvas(left, top),
                          size_2_canvas(wid, hei), angle, co.IMG_WALL)
      if cb_is_visible is None or cb_is_visible(obj):
        game_objs.append(obj)

    for idx, coord in enumerate(game_env["goals"]):
      hei = 0.8
      wid = 0.724
      left = coord[0] + 0.5 - 0.5 * wid
      top = coord[1] + 0.5 - 0.5 * hei
      obj = co.GameObject(co.IMG_GOAL + str(idx), coord_2_canvas(left, top),
                          size_2_canvas(wid, hei), 0, co.IMG_GOAL)
      if cb_is_visible is None or cb_is_visible(obj):
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
      if cb_is_visible is None or cb_is_visible(obj):
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
    if cb_is_visible is None or cb_is_visible(obj):
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
    if cb_is_visible is None or cb_is_visible(obj):
      game_objs.append(obj)

  return game_objs


def boxpush_game_scene_names(
    game_env: Mapping[str, Any],
    is_movers: bool,
    cb_is_visible: Callable[[str], bool] = None) -> List:

  drawing_names = []
  for idx, _ in enumerate(game_env["boxes"]):
    img_name = co.BOX_ORIGIN + str(idx)
    if cb_is_visible is None or cb_is_visible(img_name):
      drawing_names.append(img_name)

  for idx, _ in enumerate(game_env["walls"]):
    img_name = co.IMG_WALL + str(idx)
    if cb_is_visible is None or cb_is_visible(img_name):
      drawing_names.append(img_name)

  for idx, _ in enumerate(game_env["goals"]):
    img_name = co.IMG_GOAL + str(idx)
    if cb_is_visible is None or cb_is_visible(img_name):
      drawing_names.append(img_name)

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
      if cb_is_visible is None or cb_is_visible(img_name):
        drawing_names.append(img_name)

  if a1_hold_box < 0:
    img_name = co.IMG_WOMAN if is_movers else co.IMG_MAN
    if cb_is_visible is None or cb_is_visible(img_name):
      drawing_names.append(img_name)

  if a2_hold_box < 0:
    img_name = co.IMG_ROBOT
    if cb_is_visible is None or cb_is_visible(img_name):
      drawing_names.append(img_name)

  return drawing_names


def location_2_coord(loc: Location, places: Sequence[Place],
                     routes: Sequence[Route]):
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


def rescue_game_scene(
    game_env: Mapping[str, Any],
    game_lwth: Tuple[int, int, int, int],
    include_background: bool = True,
    cb_is_visible: Callable[[co.DrawingObject], bool] = None
) -> List[co.DrawingObject]:
  game_left, game_top, game_width, game_height = game_lwth

  def coord_2_canvas(coord_x, coord_y):
    x = int(game_left + coord_x * game_width)
    y = int(game_top + coord_y * game_height)
    return (x, y)

  def size_2_canvas(width, height):
    w = int(width * game_width)
    h = int(height * game_height)
    return (w, h)

  places = game_env["places"]  # type: Sequence[Place]
  routes = game_env["routes"]  # type: Sequence[Route]
  game_objs = []
  font_size = 15
  if include_background:
    for idx, place in enumerate(places):
      wid = 0.08
      hei = 0.08
      game_pos = coord_2_canvas(place.coord[0] - wid / 2,
                                place.coord[1] - hei / 2)
      size = size_2_canvas(wid, hei)
      text_width = size[0] * 3
      text_pos = (int(game_pos[0] + 0.5 * size[0] - 0.5 * text_width),
                  game_pos[1] + size[1])
      obj = co.Rectangle(place.name,
                         game_pos,
                         size,
                         fill_color="yellow",
                         line_color="black",
                         fill=True,
                         border=True)
      if cb_is_visible is None or cb_is_visible(obj):
        game_objs.append(obj)
      obj = co.TextObject("text" + place.name, text_pos, text_width, font_size,
                          place.name, "center")
      if cb_is_visible is None or cb_is_visible(obj):
        game_objs.append(obj)

    for idx, route in enumerate(routes):
      pos_s = location_2_coord(Location(E_Type.Route, idx, 0), places, routes)
      pos_e = location_2_coord(Location(E_Type.Route, idx, route.length - 1),
                               places, routes)
      obj = co.LineSegment(co.IMG_ROUTE + str(idx), coord_2_canvas(*pos_s),
                           coord_2_canvas(*pos_e))
      if cb_is_visible is None or cb_is_visible(obj):
        game_objs.append(obj)

  work_locations = game_env["work_locations"]
  work_states = game_env["work_states"]

  for idx, wstate in enumerate(work_states):
    if wstate != 0:
      loc = work_locations[idx]
      pos = location_2_coord(loc, places, routes)
      radius = size_2_canvas(0.01, 0)[0]
      obj = co.Circle(co.IMG_WORK + str(idx),
                      coord_2_canvas(*pos),
                      radius,
                      "green",
                      border=True)
      if cb_is_visible is None or cb_is_visible(obj):
        game_objs.append(obj)

  a1_pos = game_env["a1_pos"]
  radius = size_2_canvas(0.01, 0)[0]

  pos_a1 = location_2_coord(a1_pos, places, routes)
  pos_a1 = (pos_a1[0] - 0.02, pos_a1[1])
  obj = co.Circle(co.IMG_POLICE_CAR,
                  coord_2_canvas(*pos_a1),
                  radius,
                  "blue",
                  border=True)
  if cb_is_visible is None or cb_is_visible(obj):
    game_objs.append(obj)

  a2_pos = game_env["a2_pos"]
  pos_a2 = location_2_coord(a2_pos, places, routes)
  pos_a2 = (pos_a2[0] + 0.02, pos_a2[1])
  obj = co.Circle(co.IMG_FIRE_ENGINE,
                  coord_2_canvas(*pos_a2),
                  radius,
                  "red",
                  border=True)
  if cb_is_visible is None or cb_is_visible(obj):
    game_objs.append(obj)

  return game_objs


def rescue_game_scene_names(
    game_env: Mapping[str, Any],
    cb_is_visible: Callable[[str], bool] = None) -> List:

  drawing_names = []

  places = game_env["places"]  # type: Sequence[Place]
  routes = game_env["routes"]  # type: Sequence[Route]
  for idx, place in enumerate(places):
    img_name = place.name
    if cb_is_visible is None or cb_is_visible(img_name):
      drawing_names.append(img_name)
    obj_name = "text" + place.name
    if cb_is_visible is None or cb_is_visible(obj_name):
      drawing_names.append(obj_name)

  for idx, route in enumerate(routes):
    img_name = co.IMG_ROUTE + str(idx)
    if cb_is_visible is None or cb_is_visible(img_name):
      drawing_names.append(img_name)

  work_states = game_env["work_states"]
  for idx, wstate in enumerate(work_states):
    if wstate != 0:
      img_name = co.IMG_WORK + str(idx)
      if cb_is_visible is None or cb_is_visible(img_name):
        drawing_names.append(img_name)

  img_name_a1 = co.IMG_POLICE_CAR
  if cb_is_visible is None or cb_is_visible(img_name_a1):
    drawing_names.append(img_name_a1)

  img_name_a2 = co.IMG_FIRE_ENGINE
  if cb_is_visible is None or cb_is_visible(img_name_a2):
    drawing_names.append(img_name_a2)

  return drawing_names
