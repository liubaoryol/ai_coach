from typing import Mapping, Hashable
import json
from flask import session, request, copy_current_request_context
from flask_socketio import emit, disconnect
from ai_coach_domain.box_push import (BoxPushSimulator, BoxState,
                                      conv_box_idx_2_state)
from web_experiment import socketio

ASK_LATENT = True
NOT_ASK_LATENT = False
SHOW_FAILURE = True
NOT_SHOW_FAILURE = False


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
  print('Exp1 client disconnected', env_id)


# socketio methods
def update_html_canvas(objs, room_id, ask_latent, name_space):
  objs["ask_latent"] = ask_latent
  objs_json = json.dumps(objs)
  str_emit = 'draw_canvas'
  socketio.emit(str_emit, objs_json, room=room_id, namespace=name_space)


def on_game_end(room_id, name_space):
  socketio.emit('game_end', room=room_id, namespace=name_space)


def are_agent_states_changed(dict_env_prev, dict_updated):
  KEY_A1_POS = "a1_pos"
  KEY_A2_POS = "a2_pos"
  KEY_BOX_STATES = "box_states"
  num_drops = len(dict_env_prev["drops"])
  num_goals = len(dict_env_prev["goals"])

  a1_pos_changed = False
  a2_pos_changed = False
  if KEY_A1_POS in dict_updated:
    if dict_env_prev[KEY_A1_POS] != dict_updated[KEY_A1_POS]:
      a1_pos_changed = True

  if KEY_A2_POS in dict_updated:
    if dict_env_prev[KEY_A2_POS] != dict_updated[KEY_A2_POS]:
      a2_pos_changed = True

  a1_box_changed = False
  a2_box_changed = False
  if KEY_BOX_STATES in dict_updated:
    box_states_prev = dict_env_prev[KEY_BOX_STATES]
    a1_box_prev = False
    a2_box_prev = False
    for idx in range(len(box_states_prev)):
      state = conv_box_idx_2_state(box_states_prev[idx], num_drops, num_goals)
      if state[0] == BoxState.WithAgent1:  # with a1
        a1_box_prev = True
      elif state[0] == BoxState.WithAgent2:  # with a2
        a2_box_prev = True
      elif state[0] == BoxState.WithBoth:  # with both
        a1_box_prev = True
        a2_box_prev = True

    box_states = dict_updated[KEY_BOX_STATES]
    a1_box = False
    a2_box = False
    for idx in range(len(box_states)):
      state = conv_box_idx_2_state(box_states[idx], num_drops, num_goals)
      if state[0] == BoxState.WithAgent1:  # with a1
        a1_box = True
      elif state[0] == BoxState.WithAgent2:  # with a2
        a2_box = True
      elif state[0] == BoxState.WithBoth:  # with both
        a1_box = True
        a2_box = True

    if a1_box_prev != a1_box:
      a1_box_changed = True

    if a2_box_prev != a2_box:
      a2_box_changed = True

  return a1_pos_changed, a2_pos_changed, a1_box_changed, a2_box_changed
