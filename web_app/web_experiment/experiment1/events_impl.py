from typing import Mapping, Hashable, Callable
import os
import time
import json
import logging
import copy
import numpy as np
from flask import (session, request, copy_current_request_context, current_app,
                   url_for)
from flask_socketio import emit, disconnect
from ai_coach_domain.box_push import BoxState, conv_box_idx_2_state, EventType
from ai_coach_domain.box_push.simulator import (BoxPushSimulator,
                                                BoxPushSimulator_AlwaysTogether,
                                                BoxPushSimulator_AlwaysAlone)
from ai_coach_domain.box_push.agent import (BoxPushInteractiveAgent,
                                            BoxPushAIAgent_Abstract)
from web_experiment import socketio
from web_experiment.models import db, User

ASK_LATENT = True
NOT_ASK_LATENT = False
SHOW_FAILURE = True
NOT_SHOW_FAILURE = False
TASK_A = True
TASK_B = False

IMG_ROBOT = 'robot'
IMG_WOMAN = 'woman'
IMG_MAN = 'man'
IMG_BOX = 'box'
IMG_TRASH_BAG = 'trash_bag'
IMG_WALL = 'wall'
IMG_GOAL = 'goal'
IMG_BOTH_BOX = 'both_box'
IMG_MAN_BAG = 'man_bag'
IMG_ROBOT_BAG = 'robot_bag'


def get_file_name(user_id, session_name):
  traj_dir = os.path.join(current_app.config["TRAJECTORY_PATH"], user_id)
  # save somewhere
  if not os.path.exists(traj_dir):
    os.makedirs(traj_dir)

  sec, msec = divmod(time.time() * 1000, 1000)
  time_stamp = '%s.%03d' % (time.strftime('%Y-%m-%d_%H_%M_%S',
                                          time.gmtime(sec)), msec)
  file_name = session_name + '_' + str(user_id) + '_' + time_stamp + '.txt'
  return os.path.join(traj_dir, file_name)


def initial_canvas(session_name, is_a):
  cur_user = session.get('user_id')

  user = User.query.filter_by(userid=cur_user).first()
  done = getattr(user, session_name)
  best_score = get_best_score(cur_user, is_a)

  dict_game_info = {
      'score': 0,
      'best_score': best_score,
      'done': done,
      'user_id': cur_user
  }

  # yapf: disable
  imgs = [
      {'name': IMG_ROBOT, 'src': url_for('exp1.static', filename='robot.svg')},
      {'name': IMG_WOMAN, 'src': url_for('exp1.static', filename='woman.svg')},
      {'name': IMG_MAN, 'src': url_for('exp1.static', filename='man.svg')},
      {'name': IMG_BOX, 'src': url_for('exp1.static', filename='box.svg')},
      {'name': IMG_TRASH_BAG, 'src': url_for('exp1.static', filename='trash_bag.svg')},  # noqa: E501
      {'name': IMG_WALL, 'src': url_for('exp1.static', filename='wall.svg')},
      {'name': IMG_GOAL, 'src': url_for('exp1.static', filename='goal.svg')},
      {'name': IMG_BOTH_BOX, 'src': url_for('exp1.static', filename='both_box.svg')},  # noqa: E501
      {'name': IMG_MAN_BAG, 'src': url_for('exp1.static', filename='man_bag.svg')},  # noqa: E501
      {'name': IMG_ROBOT_BAG, 'src': url_for('exp1.static', filename='robot_bag.svg')},  # noqa: E501
  ]
  # yapf: enable

  env_dict = {'game_info': dict_game_info, 'imgs': imgs}

  env_json = json.dumps(env_dict)
  emit('init_canvas', env_json)


def test_message(message):
  # print(message['data'])
  session['receive_count'] = session.get('receive_count', 0) + 1
  emit('my_response', {
      'data': message['data'],
      'count': session['receive_count']
  })


def disconnect_request():
  @copy_current_request_context
  def can_disconnect():
    disconnect()

  session['receive_count'] = session.get('receive_count', 0) + 1
  # for this emit we use a callback function
  # when the callback function is invoked we know that the message has been
  # received and it is safe to disconnect
  emit('my_response', {
      'data': 'Exp1 disconnected!',
      'count': session['receive_count']
  },
       callback=can_disconnect)


def ping_pong():
  emit('my_pong')


def test_disconnect(id_2_game: Mapping[Hashable, BoxPushSimulator]):
  env_id = request.sid
  # finish current game
  if env_id in id_2_game:
    del id_2_game[env_id]
  # print('Exp1 client disconnected', env_id)


# socketio methods
def update_html_canvas(objs, room_id, name_space):
  objs_json = json.dumps(objs)
  str_emit = 'draw_canvas'
  socketio.emit(str_emit, objs_json, room=room_id, namespace=name_space)


def get_best_score(user_id, is_a):
  user = User.query.filter_by(userid=user_id).first()
  if is_a:
    return user.best_a
  else:
    return user.best_b


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


def are_agent_states_changed(dict_env_prev, game: BoxPushSimulator):
  num_drops = len(dict_env_prev["drops"])
  num_goals = len(dict_env_prev["goals"])

  a1_pos_changed = False
  a2_pos_changed = False
  if dict_env_prev["a1_pos"] != game.a1_pos:
    a1_pos_changed = True

  if dict_env_prev["a2_pos"] != game.a2_pos:
    a2_pos_changed = True

  a1_box_prev, a2_box_prev = get_holding_box_idx(dict_env_prev["box_states"],
                                                 num_drops, num_goals)
  a1_box, a2_box = get_holding_box_idx(game.box_states, num_drops, num_goals)

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


def action_event(msg,
                 id_2_game,
                 cb_on_hold_change,
                 cb_game_finished,
                 name_space,
                 prompt_on_change,
                 auto_prompt,
                 prompt_freq,
                 map_info,
                 session_name,
                 is_a,
                 cb_go_to_next=None):
  env_id = request.sid
  action_name = msg["data"]
  align_a2_action = "aligned" in msg

  action = None
  if action_name == "Left":
    action = EventType.LEFT
  elif action_name == "Right":
    action = EventType.RIGHT
  elif action_name == "Up":
    action = EventType.UP
  elif action_name == "Down":
    action = EventType.DOWN
  elif action_name == "Pick Up":
    action = EventType.HOLD
  elif action_name == "Drop":
    action = EventType.UNHOLD
  elif action_name == "Stay":
    action = EventType.STAY

  if action is None:
    return

  game = id_2_game[env_id]  # type: BoxPushSimulator
  if game.is_finished():
    return

  dict_env_prev = copy.deepcopy(game.get_env_info())

  game.event_input(BoxPushSimulator.AGENT1, action, None)
  if align_a2_action:
    game.event_input(BoxPushSimulator.AGENT2, action, None)

  map_agent2action = game.get_joint_action()
  game.take_a_step(map_agent2action)

  if not game.is_finished():
    (a1_pos_changed, a2_pos_changed, a1_hold_changed, a2_hold_changed, a1_box,
     a2_box) = are_agent_states_changed(dict_env_prev, game)
    unchanged_agents = []
    if not a1_pos_changed and not a1_hold_changed:
      unchanged_agents.append(0)
    if not a2_pos_changed and not a2_hold_changed:
      unchanged_agents.append(1)

    draw_overlay = False
    if prompt_on_change and (a1_hold_changed or a2_hold_changed):
      draw_overlay = True

    if auto_prompt:
      session['action_count'] = session.get('action_count', 0) + 1
      if session['action_count'] >= prompt_freq:
        draw_overlay = True

    if cb_on_hold_change:
      cb_on_hold_change(game, a1_hold_changed, a2_hold_changed, a1_box, a2_box)

    game_env = game.get_env_info()
    dict_update = get_game_drawing_obj(game_env, draw_overlay, unchanged_agents,
                                       is_a)
    if cb_go_to_next and cb_go_to_next(msg, game_env):
      control_signals = {"next_page": None}
      dict_update["control_signals"] = control_signals
    update_html_canvas(dict_update, env_id, name_space)
  else:
    if cb_game_finished:
      cb_game_finished(game, msg["user_id"], env_id, session_name, map_info,
                       name_space, is_a)


def get_game_drawing_obj(game_env, draw_overlay, unchanged_agents, is_a):
  dict_game_objs, drawing_order = get_drawing_objs(game_env, is_a)
  dict_game_overlays = get_game_overlays(game_env, draw_overlay,
                                         not draw_overlay)
  list_animations = get_animations(game_env, unchanged_agents, is_a)
  dict_game_info = {
      "select_destination": draw_overlay,
      "score": game_env["current_step"],
      "disabled_actions": get_disabled_actions(game_env),
      "drawing_order": drawing_order
  }

  dict_update = {
      "game_objects": dict_game_objs,
      "overlays": dict_game_overlays,
      "game_info": dict_game_info,
      "animations": list_animations
  }

  return dict_update


def get_disabled_actions(game_env):
  a1_latent = game_env["a1_latent"]
  if a1_latent is None:
    return ["Drop", "Pick Up"]

  disabled_actions = []
  drop_ok = False
  pickup_ok = False
  num_drops = len(game_env["drops"])
  num_goals = len(game_env["goals"])
  a1_pos = game_env["a1_pos"]
  a1_box, _ = get_holding_box_idx(game_env["box_states"], num_drops, num_goals)
  if a1_box >= 0:  # set drop action status
    if a1_latent[0] == 'origin' and a1_pos == game_env["boxes"][a1_box]:
      drop_ok = True
    else:
      for idx, coord in enumerate(game_env["goals"]):
        if a1_latent[0] == 'goal' and a1_latent[1] == idx and a1_pos == coord:
          drop_ok = True
          break
  else:  # set pickup action status
    for idx, bidx in enumerate(game_env["box_states"]):
      state = conv_box_idx_2_state(bidx, num_drops, num_goals)
      coord = None
      if state[0] == BoxState.Original:
        coord = game_env["boxes"][idx]
      elif state[0] == BoxState.WithAgent2:
        coord = game_env["a2_pos"]

      if coord is not None:
        if a1_latent[0] == 'pickup' and a1_latent[1] == idx and a1_pos == coord:
          pickup_ok = True
          break

  if not drop_ok:
    disabled_actions.append("Drop")
  if not pickup_ok:
    disabled_actions.append("Pick Up")

  return disabled_actions


def get_animations(game_env, unchanged_agents, is_a):
  num_drops = len(game_env["drops"])
  num_goals = len(game_env["goals"])

  list_animations = []

  for agent_idx in unchanged_agents:
    a1_box, a2_box = get_holding_box_idx(game_env["box_states"], num_drops,
                                         num_goals)
    obj_name = ""
    if agent_idx == 0:
      if a1_box < 0:
        obj_name = IMG_WOMAN if is_a else IMG_MAN
      elif a1_box == a2_box:
        obj_name = IMG_BOTH_BOX
      else:
        obj_name = IMG_MAN_BAG
    else:
      if a2_box < 0:
        obj_name = IMG_ROBOT
      elif a1_box == a2_box:
        obj_name = IMG_BOTH_BOX
      else:
        obj_name = IMG_ROBOT_BAG

    obj = {'type': 'vibrate', 'obj_name': obj_name}
    if obj not in list_animations:
      list_animations.append(obj)

  return list_animations


def get_drawing_objs(game_env, is_a):
  x_grid = game_env["x_grid"]
  y_grid = game_env["y_grid"]

  drawing_objs = []
  drawing_order = []
  for idx, coord in enumerate(game_env["boxes"]):
    wid = 0.8
    hei = 0.4
    left = coord[0] + 0.5 - 0.5 * wid
    top = coord[1] + 0.7 - 0.5 * hei
    obj = {
        'name': 'box_origin' + str(idx),
        'pos': [left / x_grid, top / y_grid],
        'size': [wid / x_grid, hei / y_grid],
        'img_name': 'ellipse',
        'color': 'grey'
    }
    drawing_objs.append(obj)
    drawing_order.append(obj['name'])

  for idx, coord in enumerate(game_env["walls"]):
    wid = 1.4
    hei = 1.4
    left = coord[0] + 0.5 - 0.5 * wid
    top = coord[1] + 0.5 - 0.5 * hei
    angle = 0 if game_env["wall_dir"][idx] == 0 else 0.5 * np.pi
    obj = {
        'name': IMG_WALL + str(idx),
        'pos': [left / x_grid, top / y_grid],
        'size': [wid / x_grid, hei / y_grid],
        'angle': angle,
        'img_name': IMG_WALL
    }
    drawing_objs.append(obj)
    drawing_order.append(obj['name'])

  for idx, coord in enumerate(game_env["goals"]):
    hei = 0.8
    wid = 0.724
    left = coord[0] + 0.5 - 0.5 * wid
    top = coord[1] + 0.5 - 0.5 * hei
    obj = {
        'name': IMG_GOAL + str(idx),
        'pos': [left / x_grid, top / y_grid],
        'size': [wid / x_grid, hei / y_grid],
        'angle': 0,
        'img_name': IMG_GOAL
    }
    drawing_objs.append(obj)
    drawing_order.append(obj['name'])

  num_drops = len(game_env["drops"])
  num_goals = len(game_env["goals"])
  a1_hold_box = False
  a2_hold_box = False
  for idx, bidx in enumerate(game_env["box_states"]):
    state = conv_box_idx_2_state(bidx, num_drops, num_goals)
    obj = None
    if state[0] == BoxState.Original:
      coord = game_env["boxes"][idx]
      wid = 0.541
      hei = 0.60
      left = coord[0] + 0.5 - 0.5 * wid
      top = coord[1] + 0.5 - 0.5 * hei
      img_name = IMG_BOX if is_a else IMG_TRASH_BAG
      obj = {
          'name': img_name + str(idx),
          'pos': [left / x_grid, top / y_grid],
          'size': [wid / x_grid, hei / y_grid],
          'angle': 0,
          'img_name': img_name
      }
    elif state[0] == BoxState.WithAgent1:  # with a1
      coord = game_env["a1_pos"]
      offset = 0
      if game_env["a2_pos"] == coord:
        offset = -0.2
      hei = 1
      wid = 0.385
      left = coord[0] + 0.5 + offset - 0.5 * wid
      top = coord[1] + 0.5 - 0.5 * hei
      obj = {
          'name': IMG_MAN_BAG,
          'pos': [left / x_grid, top / y_grid],
          'size': [wid / x_grid, hei / y_grid],
          'angle': 0,
          'img_name': IMG_MAN_BAG
      }
      a1_hold_box = True
    elif state[0] == BoxState.WithAgent2:  # with a2
      coord = game_env["a2_pos"]
      offset = 0
      if game_env["a1_pos"] == coord:
        offset = -0.2
      hei = 0.8
      wid = 0.476
      left = coord[0] + 0.5 + offset - 0.5 * wid
      top = coord[1] + 0.5 - 0.5 * hei
      obj = {
          'name': IMG_ROBOT_BAG,
          'pos': [left / x_grid, top / y_grid],
          'size': [wid / x_grid, hei / y_grid],
          'angle': 0,
          'img_name': IMG_ROBOT_BAG
      }
      a2_hold_box = True
    elif state[0] == BoxState.WithBoth:  # with both
      coord = game_env["a1_pos"]
      hei = 1
      wid = 0.712
      left = coord[0] + 0.5 - 0.5 * wid
      top = coord[1] + 0.5 - 0.5 * hei
      obj = {
          'name': IMG_BOTH_BOX,
          'pos': [left / x_grid, top / y_grid],
          'size': [wid / x_grid, hei / y_grid],
          'angle': 0,
          'img_name': IMG_BOTH_BOX
      }
      a1_hold_box = True
      a2_hold_box = True

    if obj is not None:
      drawing_objs.append(obj)
      drawing_order.append(obj['name'])

  if not a1_hold_box:
    coord = game_env["a1_pos"]
    offset = 0
    if coord == game_env["a2_pos"]:
      offset = 0.2 if a2_hold_box else -0.2
    hei = 1
    wid = 0.23
    left = coord[0] + 0.5 + offset - 0.5 * wid
    top = coord[1] + 0.5 - 0.5 * hei
    img_name = IMG_WOMAN if is_a else IMG_MAN
    obj = {
        'name': img_name,
        'pos': [left / x_grid, top / y_grid],
        'size': [wid / x_grid, hei / y_grid],
        'angle': 0,
        'img_name': img_name
    }
    drawing_objs.append(obj)
    drawing_order.append(obj['name'])

  if not a2_hold_box:
    coord = game_env["a2_pos"]
    offset = 0
    if coord == game_env["a1_pos"]:
      offset = 0.2
    hei = 0.8
    wid = 0.422
    left = coord[0] + 0.5 + offset - 0.5 * wid
    top = coord[1] + 0.5 - 0.5 * hei
    obj = {
        'name': IMG_ROBOT,
        'pos': [left / x_grid, top / y_grid],
        'size': [wid / x_grid, hei / y_grid],
        'angle': 0,
        'img_name': IMG_ROBOT
    }
    drawing_objs.append(obj)
    drawing_order.append(obj['name'])

  return drawing_objs, drawing_order


def get_game_overlays(game_env, draw_overlay, show_current_latent):
  x_grid = game_env["x_grid"]
  y_grid = game_env["y_grid"]

  num_drops = len(game_env["drops"])
  num_goals = len(game_env["goals"])
  a1_box, _ = get_holding_box_idx(game_env["box_states"], num_drops, num_goals)

  drawing_overlays = []
  if draw_overlay:
    radius = 0.45
    if a1_box >= 0:

      coord = game_env["boxes"][a1_box]
      x_cen = coord[0] + 0.5
      y_cen = coord[1] + 0.5
      obj = {
          'type': 'selecting',
          'pos': [x_cen / x_grid, y_cen / y_grid],
          'radius': radius / x_grid,
          'id': ["origin", 0],
          'idx': len(drawing_overlays)
      }
      drawing_overlays.append(obj)

      for idx, coord in enumerate(game_env["goals"]):
        x_cen = coord[0] + 0.5
        y_cen = coord[1] + 0.5
        obj = {
            'type': 'selecting',
            'pos': [x_cen / x_grid, y_cen / y_grid],
            'radius': radius / x_grid,
            'id': ["goal", idx],
            'idx': len(drawing_overlays)
        }
        drawing_overlays.append(obj)
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
          obj = {
              'type': 'selecting',
              'pos': [x_cen / x_grid, y_cen / y_grid],
              'radius': radius / x_grid,
              'id': ["pickup", idx],
              'idx': len(drawing_overlays)
          }
          drawing_overlays.append(obj)

  if show_current_latent:
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
        radius = 0.46
        x_cen = coord[0] + 0.5
        y_cen = coord[1] + 0.5
        obj = {
            'type': 'static',
            'pos': [x_cen / x_grid, y_cen / y_grid],
            'radius': radius / x_grid
        }
        drawing_overlays.append(obj)

  return drawing_overlays


def game_end(game: BoxPushSimulator, user_id, env_id, session_name, map_info,
             name_space, is_a):

  # save trajectory
  file_name = get_file_name(user_id, session_name)
  header = game.__class__.__name__ + "-" + session_name + "\n"
  header += "User ID: %s\n" % (str(user_id), )
  header += str(map_info)
  game.save_history(file_name, header)

  # update score
  score = game.current_step
  user = User.query.filter_by(userid=user_id).first()
  if user is not None:
    best_score = get_best_score(user_id, is_a)
    new_best = False
    # the smaller the better
    if best_score > score:
      new_best = True
      if is_a:
        user.best_a = score
      else:
        user.best_b = score

    db.session.commit()

  control_signals = {"next_page": None}
  game_info = {"score": score}
  if new_best:
    game_info["best_score"] = score
  canvas_update = {"control_signals": control_signals, "game_info": game_info}
  update_html_canvas(canvas_update, env_id, name_space)


def run_task_game(msg: Mapping, id_2_game: Mapping,
                  teammate_agent: BoxPushAIAgent_Abstract,
                  cb_set_init_latent: Callable[[BoxPushSimulator], None],
                  game_map, ask_latent, name_space, is_a):
  env_id = request.sid

  # run a game
  if env_id not in id_2_game:
    if is_a:
      id_2_game[env_id] = BoxPushSimulator_AlwaysTogether(env_id)
    else:
      id_2_game[env_id] = BoxPushSimulator_AlwaysAlone(env_id)

  game = id_2_game[env_id]  # type: BoxPushSimulator
  game.init_game(**game_map)

  agent1 = BoxPushInteractiveAgent()
  agent2 = teammate_agent
  game.set_autonomous_agent(agent1, agent2)

  if cb_set_init_latent:
    cb_set_init_latent(game)

  game_env = game.get_env_info()
  dict_update = get_game_drawing_obj(game_env, ask_latent, [], is_a)
  session["action_count"] = 0
  update_html_canvas(dict_update, env_id, name_space)


def setting_event(msg, id_2_game, name_space, cb_go_to_next=None):
  env_id = request.sid
  event_name = msg["data"]

  game = id_2_game[env_id]  # type: BoxPushSimulator
  if game.is_finished():
    return

  if event_name == "Select Destination":
    game_env = game.get_env_info()
    SELECTION = True
    NO_CUR_LATENT = not SELECTION
    dict_game_overlays = get_game_overlays(game_env, SELECTION, NO_CUR_LATENT)
    dict_game_info = {
        "select_destination": SELECTION,
        "disabled_actions": get_disabled_actions(game_env)
    }

    dict_update = {"overlays": dict_game_overlays, "game_info": dict_game_info}
    if cb_go_to_next and cb_go_to_next(msg):
      control_signals = {"next_page": None}
      dict_update["control_signals"] = control_signals

    update_html_canvas(dict_update, env_id, name_space)
  elif event_name == "Set Latent":
    latent = msg["id"]

    game.event_input(BoxPushSimulator.AGENT1, EventType.SET_LATENT,
                     tuple(latent))

    game_env = game.get_env_info()
    NO_OVERLAY = False
    SHOW_CUR_LATENT = not NO_OVERLAY
    dict_game_overlays = get_game_overlays(game_env, NO_OVERLAY,
                                           SHOW_CUR_LATENT)
    dict_game_info = {
        "select_destination": NO_OVERLAY,
        "disabled_actions": get_disabled_actions(game_env)
    }

    dict_update = {"overlays": dict_game_overlays, "game_info": dict_game_info}

    if cb_go_to_next and cb_go_to_next(msg):
      control_signals = {"next_page": None}
      dict_update["control_signals"] = control_signals
    session['action_count'] = 0
    update_html_canvas(dict_update, env_id, name_space)


def done_task(msg, session_name):
  user_id = msg["user_id"]
  user = User.query.filter_by(userid=user_id).first()
  if user is not None:
    setattr(user, session_name, True)
    db.session.commit()

  dict_game_info = {'done': True}
  env_dict = {'game_info': dict_game_info}

  env_json = json.dumps(env_dict)
  emit('task_end', env_json)
  logging.info("User %s completed %s" % (user_id, session_name))
