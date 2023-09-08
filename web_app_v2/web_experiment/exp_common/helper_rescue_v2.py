from dataclasses import dataclass
from typing import Mapping, Any, List, Tuple, Callable, Sequence
import os
import time
import numpy as np
from aic_domain.rescue_v2 import (Place, Route, Location, E_Type, PlaceName,
                                  Work, is_work_done, T_Connections)
import web_experiment.exp_common.canvas_objects as co
from web_experiment.exp_common.helper import DrawInfo

RESCUE_V2_PLACE_DRAW_INFO = {
    PlaceName.Fire_stateion:
    DrawInfo(co.IMG_FIRE_STATION, (0.07, -0.06), (0.10, 0.10),
             0,
             circles=[(0.07, -0.03, 0.1), (0, 0, 0.06)]),
    PlaceName.Police_station:
    DrawInfo(co.IMG_POLICE_STATION, (-0.1, -0.03), (0.10, 0.10),
             0,
             circles=[(-0.07, -0.02, 0.1), (0, 0, 0.06)]),
    PlaceName.Campsite:
    DrawInfo(co.IMG_CAMPSITE, (-0.05, -0.05), (0.13, 0.09),
             0,
             circles=[(0.01, -0.17, 0.2), (0, 0, 0.06)]),
    PlaceName.City_hall:
    DrawInfo(co.IMG_CITY_HALL, (0, -0.08), (0.10, 0.08),
             0,
             circles=[(0, -0.03, 0.11)]),
    PlaceName.Mall:
    DrawInfo(co.IMG_MALL, (0.06, 0.05), (0.10, 0.10),
             0,
             circles=[(0.06, 0.03, 0.12), (0, 0, 0.06)]),
    PlaceName.Hospital:
    DrawInfo(co.IMG_HOSPITAL, (0.05, -0.06), (0.10, 0.09),
             0,
             circles=[(0.03, -0.06, 0.1), (0, 0, 0.06)]),
}


def location_2_coord_v2(loc: Location, places: Sequence[Place],
                        routes: Sequence[Route]):
  if loc.type == E_Type.Place:
    return places[loc.id].coord
  else:
    route_id = loc.id  # type: int
    route = routes[route_id]  # type: Route
    idx = loc.index
    return route.coords[idx]


def rescue_v2_game_scene(
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

  # place_w = 0.12
  # place_h = 0.12

  places = game_env["places"]  # type: Sequence[Place]
  routes = game_env["routes"]  # type: Sequence[Route]
  connections = game_env["connections"]  # type: Mapping[int, T_Connections]
  game_objs = []

  def add_obj(obj):
    if cb_is_visible is None or cb_is_visible(obj):
      game_objs.append(obj)

  rcnt = 0
  font_size = 15
  if include_background:
    obj = co.GameObject(co.IMG_TOWER + "0", coord_2_canvas(0.85, 0.11),
                        size_2_canvas(0.1, 0.1), 0, co.IMG_TOWER)
    add_obj(obj)
    obj = co.GameObject(co.IMG_THUNDER + "0", coord_2_canvas(0.85, 0.01),
                        size_2_canvas(0.1, 0.1), 0, co.IMG_THUNDER)
    add_obj(obj)

    obj = co.GameObject(co.IMG_TOWER + "1", coord_2_canvas(0.9, 0.2),
                        size_2_canvas(0.1, 0.1), 0, co.IMG_TOWER)
    add_obj(obj)
    obj = co.GameObject(co.IMG_THUNDER + "1", coord_2_canvas(0.9, 0.1),
                        size_2_canvas(0.1, 0.1), 0, co.IMG_THUNDER)
    add_obj(obj)

    obj = co.GameObject(co.IMG_TOWER + "2", coord_2_canvas(0.75, 0.12),
                        size_2_canvas(0.1, 0.1), 0, co.IMG_TOWER)
    add_obj(obj)
    obj = co.GameObject(co.IMG_THUNDER + "2", coord_2_canvas(0.75, 0.02),
                        size_2_canvas(0.1, 0.1), 0, co.IMG_THUNDER)
    add_obj(obj)

    for idx, route in enumerate(routes):
      list_coord = []
      list_coord.append(coord_2_canvas(*places[route.start].coord))
      for coord in route.coords:
        list_coord.append(coord_2_canvas(*coord))
      list_coord.append(coord_2_canvas(*places[route.end].coord))

      obj = co.Curve(co.IMG_ROUTE + str(rcnt), list_coord, 10, "grey")
      add_obj(obj)
      rcnt += 1

    for idx in connections:
      for connect in connections[idx]:
        if connect[0] == E_Type.Place and connect[1] > idx:
          list_coord = [
              coord_2_canvas(*places[idx].coord),
              coord_2_canvas(*places[connect[1]].coord),
          ]
          obj = co.Curve(co.IMG_ROUTE + str(rcnt), list_coord, 10, "grey")
          add_obj(obj)
          rcnt += 1

    def add_place(place: Place, offset, size, img_name):
      name = place.name
      building_pos = np.array(place.coord) + np.array(offset)
      canvas_pt = coord_2_canvas(*place.coord)

      wid = size[0]
      hei = size[1]
      size_cnvs = size_2_canvas(wid, hei)
      game_pos = coord_2_canvas(building_pos[0] - wid / 2,
                                building_pos[1] - hei / 2)
      text_width = size_cnvs[0] * 2
      text_pos = (int(game_pos[0] + 0.5 * size_cnvs[0] - 0.5 * text_width),
                  int(game_pos[1] - font_size))
      obj = co.Circle("ground" + name, canvas_pt,
                      size_2_canvas(0.03, 0)[0], "grey")
      add_obj(obj)
      obj = co.GameObject(name, game_pos, size_cnvs, 0, img_name)
      add_obj(obj)
      obj = co.TextObject("text" + name, text_pos, text_width, font_size, name,
                          "center")
      add_obj(obj)

      p_wid = 0.025
      p_hei = p_wid * 2.5
      p_size_cnvs = size_2_canvas(p_wid, p_hei)
      for pidx in range(place.helps):
        if pidx < 2:
          p_x = game_pos[0] - p_size_cnvs[0] * (pidx + 1)
          p_y = game_pos[1] + size_cnvs[1] - p_size_cnvs[1]
        else:
          p_x = game_pos[0] - p_size_cnvs[0] * (pidx - 1)
          p_y = game_pos[1] + size_cnvs[1] - 2 * p_size_cnvs[1]
        obj = co.GameObject("human" + name + str(pidx), (p_x, p_y), p_size_cnvs,
                            0, co.IMG_HUMAN)
        add_obj(obj)

    for idx in range(6):
      place_name = places[idx].name
      offset = RESCUE_V2_PLACE_DRAW_INFO[place_name].offset
      size = RESCUE_V2_PLACE_DRAW_INFO[place_name].size
      img_name = RESCUE_V2_PLACE_DRAW_INFO[place_name].img_name
      add_place(places[idx], offset, size, img_name)

  work_locations = game_env["work_locations"]  # type: Sequence[Location]
  work_states = game_env["work_states"]
  work_info = game_env["work_info"]  # type: Sequence[Work]

  pos_a1 = location_2_coord_v2(game_env["a1_pos"], places, routes)
  pos_a2 = location_2_coord_v2(game_env["a2_pos"], places, routes)
  pos_a3 = location_2_coord_v2(game_env["a3_pos"], places, routes)
  wid_a = 0.085
  hei_a = 0.085
  offset_x_a1 = 0
  offset_y_a1 = 0
  offset_x_a2 = 0
  offset_y_a2 = 0
  offset_x_a3 = 0
  offset_y_a3 = 0
  for idx, wstate in enumerate(work_states):
    if wstate != 0:
      loc = work_locations[idx]
      pos = location_2_coord_v2(loc, places, routes)
      if pos == pos_a1:
        offset_x_a1 = -wid_a * 0.7 / 2
        offset_y_a1 = -hei_a * 0.5 / 2
      if pos == pos_a2:
        offset_x_a2 = wid_a * 0.7 / 2
        offset_y_a2 = -hei_a * 0.5 / 2
      if pos == pos_a3:
        offset_x_a3 = 0
        offset_y_a3 = hei_a * 0.5 / 2

      wid = 0.06
      hei = 0.06
      offset_x = 0
      offset_y = 0
      game_pos = coord_2_canvas(pos[0] + offset_x - wid / 2,
                                pos[1] + offset_y - hei / 2)
      size_cnvs = size_2_canvas(wid, hei)
      obj = co.GameObject(co.IMG_WORK + str(idx), game_pos, size_cnvs, 0,
                          co.IMG_WORK)
      add_obj(obj)

  if pos_a1 == pos_a2 and pos_a2 == pos_a3 and offset_x_a1 == 0:
    offset_x_a1 = -wid_a * 0.7 / 2
    offset_y_a1 = -hei_a * 0.5 / 2
    offset_x_a2 = wid_a * 0.7 / 2
    offset_y_a2 = -hei_a * 0.5 / 2
    offset_x_a3 = 0
    offset_y_a3 = hei_a * 0.5 / 2

  if pos_a1 == pos_a2 and offset_x_a1 == 0:
    offset_x_a1 = -wid_a * 0.7 / 2
    offset_y_a1 = hei_a * 0.5 / 2
    offset_x_a2 = wid_a * 0.7 / 2
    offset_y_a2 = -hei_a * 0.5 / 2

  if pos_a1 == pos_a3 and offset_x_a1 == 0:
    offset_x_a1 = -wid_a * 0.7 / 2
    offset_y_a1 = hei_a * 0.5 / 2
    offset_x_a3 = wid_a * 0.7 / 2
    offset_y_a3 = -hei_a * 0.5 / 2

  if pos_a2 == pos_a3 and offset_x_a2 == 0:
    offset_x_a2 = -wid_a * 0.7 / 2
    offset_y_a2 = hei_a * 0.5 / 2
    offset_x_a3 = wid_a * 0.7 / 2
    offset_y_a3 = -hei_a * 0.5 / 2

  game_pos_a1 = coord_2_canvas(pos_a1[0] + offset_x_a1 - wid_a / 2,
                               pos_a1[1] + offset_y_a1 - hei_a / 2)
  size_a1 = size_2_canvas(wid_a, hei_a)

  game_pos_a2 = coord_2_canvas(pos_a2[0] + offset_x_a2 - wid_a / 2,
                               pos_a2[1] + offset_y_a2 - hei_a / 2)
  size_a2 = size_2_canvas(wid_a, hei_a)

  game_pos_a3 = coord_2_canvas(pos_a3[0] + offset_x_a3 - wid_a / 2,
                               pos_a3[1] + offset_y_a3 - hei_a / 2)
  size_a3 = size_2_canvas(wid_a, hei_a)

  obj = co.GameObject(co.IMG_POLICE_CAR, game_pos_a1, size_a1, 0,
                      co.IMG_POLICE_CAR)
  add_obj(obj)
  obj = co.GameObject(co.IMG_FIRE_ENGINE, game_pos_a2, size_a2, 0,
                      co.IMG_FIRE_ENGINE)
  add_obj(obj)
  obj = co.GameObject(co.IMG_AMBULANCE, game_pos_a3, size_a3, 0,
                      co.IMG_AMBULANCE)
  add_obj(obj)

  return game_objs


def rescue_v2_game_scene_names(
    game_env: Mapping[str, Any],
    cb_is_visible: Callable[[str], bool] = None) -> List:

  drawing_names = []

  def add_obj_name(obj_name):
    if cb_is_visible is None or cb_is_visible(obj_name):
      drawing_names.append(obj_name)

  obj = add_obj_name(co.IMG_TOWER + "0")
  obj = add_obj_name(co.IMG_THUNDER + "0")
  obj = add_obj_name(co.IMG_TOWER + "1")
  obj = add_obj_name(co.IMG_THUNDER + "1")
  obj = add_obj_name(co.IMG_TOWER + "2")
  obj = add_obj_name(co.IMG_THUNDER + "2")

  rcnt = 0
  routes = game_env["routes"]  # type: Sequence[Route]
  connections = game_env["connections"]  # type: Mapping[int, T_Connections]
  for _ in routes:
    add_obj_name(co.IMG_ROUTE + str(rcnt))
    rcnt += 1

  for idx in connections:
    for connect in connections[idx]:
      if connect[0] == E_Type.Place and connect[1] > idx:
        add_obj_name(co.IMG_ROUTE + str(rcnt))
        rcnt += 1

  work_locations = game_env["work_locations"]  # type: Sequence[Location]
  work_states = game_env["work_states"]
  places = game_env["places"]  # type: Sequence[Place]

  for idx in range(6):
    add_obj_name("ground" + places[idx].name)
    add_obj_name(places[idx].name)
    add_obj_name("text" + places[idx].name)

  for idx, wstate in enumerate(work_states):
    if wstate != 0:
      add_obj_name(co.IMG_WORK + str(idx))

  work_info = game_env["work_info"]  # type: Sequence[Work]
  for idx, _ in enumerate(work_states):
    if not is_work_done(idx, work_states, work_info[idx].coupled_works):
      place = places[work_info[idx].rescue_place]
      num_help = place.helps
      for pidx in range(num_help):
        add_obj_name("human" + place.name + str(pidx))

  add_obj_name(co.IMG_POLICE_CAR)
  add_obj_name(co.IMG_FIRE_ENGINE)
  add_obj_name(co.IMG_AMBULANCE)

  return drawing_names
