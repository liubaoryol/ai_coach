import json
from flask import session, request, copy_current_request_context
from flask_socketio import emit, disconnect
from web_experiment import socketio
from domains.box_push.box_push_simulator import BoxPushSimulator
# import game

g_map_id_2_game = {}


@socketio.on('connect', namespace='/experiment1')
def initial_canvas():
  GRID_X = BoxPushSimulator.X_GRID
  GRID_Y = BoxPushSimulator.Y_GRID
  goals = [BoxPushSimulator.GOAL]
  env_dict = {'grid_x': GRID_X, 'grid_y': GRID_Y, 'goals': []}

  def coord2idx(coord):
    return coord[1] * GRID_X + coord[0]

  for pos in goals:
    env_dict['goals'].append(coord2idx(pos))

  env_json = json.dumps(env_dict)
  emit('init_canvas', env_json)


@socketio.on('my_echo', namespace='/experiment1')
def test_message(message):
  print(message['data'])
  session['receive_count'] = session.get('receive_count', 0) + 1
  emit('my_response', {
      'data': message['data'],
      'count': session['receive_count']
  })


@socketio.on('disconnect_request', namespace='/experiment1')
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


@socketio.on('my_ping', namespace='/experiment1')
def ping_pong():
  emit('my_pong')


@socketio.on('disconnect', namespace='/experiment1')
def test_disconnect():
  print('Exp1 client disconnected', request.sid)
  # finish current game


# socketio methods
def update_html_canvas(objs, room_id):
  objs_json = json.dumps(objs)
  socketio.emit('draw_canvas', objs_json, room=room_id)


def on_game_end(room_id):
  socketio.emit('game_end', room=room_id)


@socketio.on('run_experiment', namespace='/experiment1')
def run_experiment(msg):
  env_id = request.sid

  # run a game
  global g_map_id_2_game
  game = g_map_id_2_game[env_id] = BoxPushSimulator(env_id)

  dict_update = game.get_changed_objects()
  if dict_update is not None:
    update_html_canvas(dict_update, env_id)


@socketio.on('keydown_event', namespace='/experiment1')
def on_key_down(msg):
  env_id = request.sid

  action = None

  key_code = msg["data"]
  if key_code == "ArrowLeft":  # Left
    action = "Left"
  elif key_code == "ArrowRight":  # Right
    action = "Right"
  elif key_code == "ArrowUp":  # Up
    action = "Up"
  elif key_code == "ArrowDown":  # Down
    action = "Down"
  elif key_code == "p":  # p
    action = "p"
  elif key_code == "o":  # o
    action = "o"

  action
  list_update = None  # TODO: get a function return
  if list_update is not None:
    update_html_canvas(list_update, env_id)
