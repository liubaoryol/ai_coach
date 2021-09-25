from typing import Mapping, Hashable
import os
import time
import json
import logging
import random
from flask import session, request, copy_current_request_context, current_app
from flask_socketio import emit, disconnect
from ai_coach_domain.box_push import BoxState, conv_box_idx_2_state, EventType
from ai_coach_domain.box_push.box_push_simulator import BoxPushSimulator
from web_experiment import socketio
from web_experiment.models import db, User

ASK_LATENT = True
NOT_ASK_LATENT = False
SHOW_FAILURE = True
NOT_SHOW_FAILURE = False


def get_file_name(user_id, session_name):
  traj_dir = current_app.config["TRAJECTORY_PATH"]
  # save somewhere
  if not os.path.exists(traj_dir):
    os.makedirs(traj_dir)

  sec, msec = divmod(time.time() * 1000, 1000)
  time_stamp = '%s.%03d' % (time.strftime('%Y-%m-%d_%H_%M_%S',
                                          time.gmtime(sec)), msec)
  file_name = session_name + '_' + str(user_id) + '_' + time_stamp + '.txt'
  return os.path.join(traj_dir, file_name)


def initial_canvas(grid_x, grid_y):
  env_dict = {'grid_x': grid_x, 'grid_y': grid_y}

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
def update_html_canvas(objs, room_id, ask_latent, name_space):
  objs["ask_latent"] = ask_latent
  objs_json = json.dumps(objs)
  str_emit = 'draw_canvas'
  socketio.emit(str_emit, objs_json, room=room_id, namespace=name_space)


def on_game_end(room_id, name_space, user_id, session_name, score, is_a):
  logging.info("User %s completed %s" % (user_id, session_name))

  socketio.emit('game_end', room=room_id, namespace=name_space)
  user = User.query.filter_by(userid=user_id).first()

  if user is not None:
    best_score = get_best_score(user_id, is_a)
    if best_score > score:
      if is_a:
        user.best_a = score
      else:
        user.best_b = score

    setattr(user, session_name, True)
    db.session.commit()


def get_best_score(user_id, is_a):
  user = User.query.filter_by(userid=user_id).first()
  if is_a:
    return user.best_a
  else:
    return user.best_b


def are_agent_states_changed(dict_env_prev, game: BoxPushSimulator):
  KEY_A1_POS = "a1_pos"
  KEY_A2_POS = "a2_pos"
  KEY_BOX_STATES = "box_states"
  num_drops = len(dict_env_prev["drops"])
  num_goals = len(dict_env_prev["goals"])

  a1_pos_changed = False
  a2_pos_changed = False
  if dict_env_prev[KEY_A1_POS] != game.a1_pos:
    a1_pos_changed = True

  if dict_env_prev[KEY_A2_POS] != game.a2_pos:
    a2_pos_changed = True

  box_states_prev = dict_env_prev[KEY_BOX_STATES]
  a1_box_prev = -1
  a2_box_prev = -1
  for idx in range(len(box_states_prev)):
    state = conv_box_idx_2_state(box_states_prev[idx], num_drops, num_goals)
    if state[0] == BoxState.WithAgent1:  # with a1
      a1_box_prev = idx
    elif state[0] == BoxState.WithAgent2:  # with a2
      a2_box_prev = idx
    elif state[0] == BoxState.WithBoth:  # with both
      a1_box_prev = idx
      a2_box_prev = idx

  box_states = game.box_states
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


def change_a2_latent_based_on_a1(game: BoxPushSimulator):
  a2_latent = game.a2_latent
  if a2_latent[0] != "pickup":
    return

  closest_idx = None
  dist = 100000

  a1_pos = game.a1_pos
  for idx, bidx in enumerate(game.box_states):
    bstate = conv_box_idx_2_state(bidx, len(game.drops), len(game.goals))
    if bstate[0] == BoxState.Original:
      box_pos = game.boxes[idx]
      dist_cur = abs(a1_pos[0] - box_pos[0]) + abs(a1_pos[1] - box_pos[1])
      if dist > dist_cur:
        dist = dist_cur
        closest_idx = idx

  if closest_idx is not None and dist < 2 and a2_latent[1] != closest_idx:
    prop = 0.1
    if prop > random.uniform(0, 1):
      game.event_input(BoxPushSimulator.AGENT2, EventType.SET_LATENT,
                       ("pickup", closest_idx))
