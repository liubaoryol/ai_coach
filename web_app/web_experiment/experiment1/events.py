from typing import Mapping, Hashable
import json
from flask import session, request, copy_current_request_context
from flask_socketio import emit, disconnect
from web_experiment import socketio
from ai_coach_domain.box_push import BoxPushSimulator, EventType

g_id_2_game = {}  # type: Mapping[Hashable, BoxPushSimulator]
EXP1_NAMESPACE = '/experiment1'

ASK_LATENT = True
NOT_ASK_LATENT = False
SHOW_FAILURE = True
NOT_SHOW_FAILURE = False


@socketio.on('connect', namespace=EXP1_NAMESPACE)
def initial_canvas():
  GRID_X = BoxPushSimulator.X_GRID
  GRID_Y = BoxPushSimulator.Y_GRID
  env_dict = {'grid_x': GRID_X, 'grid_y': GRID_Y}

  env_json = json.dumps(env_dict)
  emit('init_canvas', env_json)


@socketio.on('my_echo', namespace=EXP1_NAMESPACE)
def test_message(message):
  # print(message['data'])
  session['receive_count'] = session.get('receive_count', 0) + 1
  emit('my_response', {
      'data': message['data'],
      'count': session['receive_count']
  })


@socketio.on('disconnect_request', namespace=EXP1_NAMESPACE)
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


@socketio.on('my_ping', namespace=EXP1_NAMESPACE)
def ping_pong():
  emit('my_pong')


@socketio.on('disconnect', namespace=EXP1_NAMESPACE)
def test_disconnect():
  env_id = request.sid
  # finish current game
  if env_id in g_id_2_game:
    del g_id_2_game[env_id]
  print('Exp1 client disconnected', env_id)


# socketio methods
def update_html_canvas(objs, room_id, ask_latent, show_failure):
  objs["ask_latent"] = ask_latent
  objs["show_failure"] = show_failure
  objs_json = json.dumps(objs)
  str_emit = 'draw_canvas'
  socketio.emit(str_emit, objs_json, room=room_id, namespace=EXP1_NAMESPACE)


def on_game_end(room_id):
  socketio.emit('game_end', room=room_id, namespace=EXP1_NAMESPACE)


@socketio.on('run_game', namespace=EXP1_NAMESPACE)
def run_game(msg):
  env_id = request.sid

  # run a game
  global g_id_2_game
  if env_id not in g_id_2_game:
    g_id_2_game[env_id] = BoxPushSimulator(env_id)

  game = g_id_2_game[env_id]
  dict_update = game.get_env_info()
  if dict_update is not None:
    session['action_count'] = 0
    update_html_canvas(dict_update, env_id, ASK_LATENT, NOT_SHOW_FAILURE)


@socketio.on('action_event', namespace=EXP1_NAMESPACE)
def on_key_down(msg):
  env_id = request.sid

  action = None
  action_name = msg["data"]
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

  if action:
    session['action_count'] = session.get('action_count', 0) + 1
    ASK_LATENT_FREQUENCY = 5

    global g_id_2_game
    game = g_id_2_game[env_id]
    game.event_input(BoxPushSimulator.AGENT1, action, None)
    map_agent2action = game.get_action()
    game.take_a_step(map_agent2action)

    if not game.is_finished():
      dict_update = game.get_changed_objects()
      if dict_update is None:
        dict_update = {}

      draw_overlay = (True if session['action_count'] >= ASK_LATENT_FREQUENCY
                      else False)
      SHOW_FAILURE = True
      update_html_canvas(dict_update, env_id, draw_overlay, SHOW_FAILURE)
    else:
      game.reset_game()
      on_game_end(env_id)


@socketio.on('set_latent', namespace=EXP1_NAMESPACE)
def set_latent(msg):
  env_id = request.sid
  latent = msg["data"]

  global g_id_2_game
  game = g_id_2_game[env_id]
  game.event_input(BoxPushSimulator.AGENT1, EventType.SET_LATENT, latent)

  dict_update = game.get_changed_objects()
  session['action_count'] = 0
  update_html_canvas(dict_update, env_id, NOT_ASK_LATENT, NOT_SHOW_FAILURE)
