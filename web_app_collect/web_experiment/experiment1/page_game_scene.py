from typing import Mapping, Any, List, Tuple, Callable
import numpy as np
from ai_coach_domain.box_push import conv_box_idx_2_state, BoxState
import web_experiment.experiment1.canvas_objects as co


def game_scene(
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


def game_scene_names(game_env: Mapping[str, Any],
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
