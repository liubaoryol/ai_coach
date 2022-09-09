from typing import Mapping, Any, List, Tuple, Callable, Sequence
import os
import time
import numpy as np
from ai_coach_domain.box_push import conv_box_idx_2_state, BoxState
from ai_coach_domain.rescue import (Place, Route, Location, E_Type, PlaceName,
                                    Work)
import web_experiment.exp_common.canvas_objects as co

RESCUE_NAME_PLACE2IMG = {
    PlaceName.Fire_stateion: co.IMG_FIRE_STATION,
    PlaceName.Police_station: co.IMG_POLICE_STATION,
    PlaceName.Campsite: co.IMG_CAMPSITE,
    PlaceName.City_hall: co.IMG_CITY_HALL,
    PlaceName.Mall: co.IMG_MALL,
    PlaceName.Bridge_1: co.IMG_BRIDGE,
    PlaceName.Bridge_2: co.IMG_BRIDGE,
}


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
      wid = 1
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
  if loc.type == E_Type.Place:
    return places[loc.id].coord
  else:
    route_id = loc.id  # type: int
    route = routes[route_id]  # type: Route
    idx = loc.index
    return route.coords[idx]


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

  place_w = 0.12
  place_h = 0.12

  places = game_env["places"]  # type: Sequence[Place]
  routes = game_env["routes"]  # type: Sequence[Route]
  game_objs = []

  def add_obj(obj):
    if cb_is_visible is None or cb_is_visible(obj):
      game_objs.append(obj)

  font_size = 15
  if include_background:
    river_coords = [
        coord_2_canvas(0.68, 1),
        coord_2_canvas(0.75, 0.9),
        coord_2_canvas(0.77, 0.83),
        coord_2_canvas(0.85, 0.77),
        coord_2_canvas(1, 0.7)
    ]
    obj = co.Curve(co.IMG_BACKGROUND, river_coords, 30, "blue")
    add_obj(obj)

    for idx, route in enumerate(routes):
      list_coord = []
      list_coord.append(coord_2_canvas(*places[route.start].coord))
      for coord in route.coords:
        list_coord.append(coord_2_canvas(*coord))
      list_coord.append(coord_2_canvas(*places[route.end].coord))

      obj = co.Curve(co.IMG_ROUTE + str(idx), list_coord, 10, "grey")
      add_obj(obj)

    def add_place(place: Place, offset, scale_x, scale_y, text_offset):
      name = place.name
      building_pos = np.array(place.coord) + np.array(offset)
      canvas_pt = coord_2_canvas(*place.coord)

      wid = place_w * scale_x
      hei = place_h * scale_y
      size = size_2_canvas(wid, hei)
      game_pos = coord_2_canvas(building_pos[0] - wid / 2,
                                building_pos[1] - hei / 2)
      text_width = size[0] * 2
      text_pos = (int(game_pos[0] + 0.5 * size[0] - 0.5 * text_width),
                  int(game_pos[1] - font_size +
                      size_2_canvas(text_offset, 0)[0]))
      obj = co.Circle("ground" + name, canvas_pt,
                      size_2_canvas(0.03, 0)[0], "grey")
      add_obj(obj)
      obj = co.GameObject(name, game_pos, size, 0, RESCUE_NAME_PLACE2IMG[name])
      add_obj(obj)
      obj = co.TextObject("text" + name, text_pos, text_width, font_size, name,
                          "center")
      add_obj(obj)

    # Fire_stateion
    add_place(places[0], (0.05, -0.1), 1, 1, 0)
    # City_hall
    add_place(places[1], (-0.03, -0.05), 1, 0.8, 0)
    # Police_station
    add_place(places[2], (-0.09, 0), 1, 1, 0)
    # Campsite
    add_place(places[4], (0, -0.07), 1.3, 1.0, 0)
    # Mall
    add_place(places[5], (0, -0.05), 1.2, 1.2, 0)

  work_locations = game_env["work_locations"]  # type: Sequence[Location]
  work_states = game_env["work_states"]
  work_info = game_env["work_info"]  # type: Sequence[Work]

  pos_a1 = location_2_coord(game_env["a1_pos"], places, routes)
  pos_a2 = location_2_coord(game_env["a2_pos"], places, routes)
  wid_a = place_w * 0.7
  hei_a = place_h * 0.7
  offset_x_a1 = 0
  offset_y_a1 = 0
  offset_x_a2 = 0
  offset_y_a2 = 0
  for idx, wstate in enumerate(work_states):
    if wstate != 0:
      loc = work_locations[idx]
      pos = location_2_coord(loc, places, routes)
      if pos == pos_a1:
        offset_x_a1 = -wid_a * 0.7 / 2
        offset_y_a1 = hei_a * 0.5 / 2
      if pos == pos_a2:
        offset_x_a2 = wid_a * 0.7 / 2
        offset_y_a2 = -hei_a * 0.5 / 2

      wid = place_w * 0.5
      hei = place_h * 0.5
      offset_x = 0
      offset_y = 0
      game_pos = coord_2_canvas(pos[0] + offset_x - wid / 2,
                                pos[1] + offset_y - hei / 2)
      size = size_2_canvas(wid, hei)
      obj = co.GameObject(co.IMG_WORK + str(idx), game_pos, size, 0,
                          co.IMG_WORK)
      add_obj(obj)
    else:
      if work_locations[idx].type == E_Type.Place:
        place = places[work_locations[idx].id]
        if place.name in [PlaceName.Bridge_1, PlaceName.Bridge_2]:
          loc = work_locations[idx]
          pos = location_2_coord(loc, places, routes)
          wid = place_w
          hei = place_h
          game_pos = coord_2_canvas(pos[0] + wid * 0 / 2,
                                    pos[1] + hei * 0.01 / 2)
          size = size_2_canvas(wid, hei * 0.25)
          obj = co.GameObject(place.name, game_pos, size, 0.15 * np.pi,
                              co.IMG_BRIDGE)
          add_obj(obj)

  if pos_a1 == pos_a2 and offset_x_a1 == 0:
    offset_x_a1 = -wid_a * 0.7 / 2
    offset_y_a1 = hei_a * 0.5 / 2
    offset_x_a2 = wid_a * 0.7 / 2
    offset_y_a2 = -hei_a * 0.5 / 2

  game_pos_a1 = coord_2_canvas(pos_a1[0] + offset_x_a1 - wid_a / 2,
                               pos_a1[1] + offset_y_a1 - hei_a / 2)
  size_a1 = size_2_canvas(wid_a, hei_a)

  game_pos_a2 = coord_2_canvas(pos_a2[0] + offset_x_a2 - wid_a / 2,
                               pos_a2[1] + offset_y_a2 - hei_a / 2)
  size_a2 = size_2_canvas(wid_a, hei_a)

  obj = co.GameObject(co.IMG_POLICE_CAR, game_pos_a1, size_a1, 0,
                      co.IMG_POLICE_CAR)
  add_obj(obj)
  obj = co.GameObject(co.IMG_FIRE_ENGINE, game_pos_a2, size_a2, 0,
                      co.IMG_FIRE_ENGINE)
  add_obj(obj)

  return game_objs


def rescue_game_scene_names(
    game_env: Mapping[str, Any],
    cb_is_visible: Callable[[str], bool] = None) -> List:

  drawing_names = []

  def add_obj_name(obj_name):
    if cb_is_visible is None or cb_is_visible(obj_name):
      drawing_names.append(obj_name)

  add_obj_name(co.IMG_BACKGROUND)

  routes = game_env["routes"]  # type: Sequence[Route]
  for idx, _ in enumerate(routes):
    add_obj_name(co.IMG_ROUTE + str(idx))

  work_locations = game_env["work_locations"]  # type: Sequence[Location]
  work_states = game_env["work_states"]
  places = game_env["places"]  # type: Sequence[Place]
  for idx, wstate in enumerate(work_states):
    if wstate == 0:
      if work_locations[idx].type == E_Type.Place:
        place = places[work_locations[idx].id]
        if place.name in [PlaceName.Bridge_1, PlaceName.Bridge_2]:
          add_obj_name(place.name)

  for idx in [0, 1, 2, 4, 5]:
    add_obj_name("ground" + places[idx].name)
    add_obj_name(places[idx].name)
    add_obj_name("text" + places[idx].name)

  for idx, wstate in enumerate(work_states):
    if wstate != 0:
      add_obj_name(co.IMG_WORK + str(idx))

  add_obj_name(co.IMG_POLICE_CAR)
  add_obj_name(co.IMG_FIRE_ENGINE)

  return drawing_names
