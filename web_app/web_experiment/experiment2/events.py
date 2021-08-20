import json
from flask import session, request, copy_current_request_context
from flask_socketio import emit, disconnect
from web_experiment import socketio


@socketio.on('connect', namespace='/experiment2')
def initial_canvas():
  GRID_X = 5
  GRID_Y = 5
  goals = [(1, 1), (3, 3)]
  env_dict = {'grid_x': GRID_X, 'grid_y': GRID_Y, 'goals': []}

  def coord2idx(coord):
    return coord[1] * GRID_X + coord[0]

  for pos in goals:
    env_dict['goals'].append(coord2idx(pos))

  env_json = json.dumps(env_dict)
  emit('init_canvas', env_json)


@socketio.on('my_echo', namespace='/experiment2')
def test_message(message):
  print(message['data'])
  session['receive_count'] = session.get('receive_count', 0) + 1
  emit('my_response', {
      'data': message['data'],
      'count': session['receive_count']
  })


@socketio.on('disconnect_request', namespace='/experiment2')
def disconnect_request():
  @copy_current_request_context
  def can_disconnect():
    disconnect()

  session['receive_count'] = session.get('receive_count', 0) + 1
  # for this emit we use a callback function
  # when the callback function is invoked we know that the message has been
  # received and it is safe to disconnect
  emit('my_response', {
      'data': 'Exp2 disconnected!',
      'count': session['receive_count']
  },
       callback=can_disconnect)


@socketio.on('my_ping', namespace='/experiment2')
def ping_pong():
  emit('my_pong')


@socketio.on('disconnect', namespace='/experiment2')
def test_disconnect():
  print('Exp2 client disconnected', request.sid)
  # finish current game
