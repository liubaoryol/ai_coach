from typing import Mapping, Hashable
import json
from flask import session, request, copy_current_request_context
from flask_socketio import emit, disconnect
from ai_coach_domain.box_push import BoxPushSimulator, EventType
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
def update_html_canvas(objs, room_id, ask_latent, show_failure, name_space):
  objs["ask_latent"] = ask_latent
  objs["show_failure"] = show_failure
  objs_json = json.dumps(objs)
  str_emit = 'draw_canvas'
  socketio.emit(str_emit, objs_json, room=room_id, namespace=name_space)


def on_game_end(room_id, name_space):
  socketio.emit('game_end', room=room_id, namespace=name_space)


def run_game(msg,
             name_space,
             id_2_game: Mapping[Hashable, BoxPushSimulator],
             grid_x,
             grid_y,
             game_map={}):
  env_id = request.sid

  # run a game
  if env_id not in id_2_game:
    id_2_game[env_id] = BoxPushSimulator(env_id)

  game = id_2_game[env_id]
  if len(game_map) != 0:
    game.init_game(grid_x,
                   grid_y,
                   boxes=game_map["boxes"],
                   goals=game_map["goals"],
                   walls=game_map["walls"],
                   wall_dir=game_map["wall_dir"],
                   drops=game_map["drops"])
  else:
    game.init_game_with_test_map(grid_x, grid_y)
  dict_update = game.get_env_info()
  if dict_update is not None:
    session['action_count'] = 0
    update_html_canvas(dict_update, env_id, ASK_LATENT, NOT_SHOW_FAILURE,
                       name_space)


def on_key_down(msg, name_space, id_2_game: Mapping[Hashable,
                                                    BoxPushSimulator]):
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
    game = id_2_game[env_id]
    game.event_input(BoxPushSimulator.AGENT1, action, None)
    map_agent2action = game.get_action()
    game.take_a_step(map_agent2action)

    if not game.is_finished():
      dict_update = game.get_changed_objects()
      if dict_update is None:
        dict_update = {}

      session['action_count'] = session.get('action_count', 0) + 1
      ASK_LATENT_FREQUENCY = 5

      draw_overlay = (True if session['action_count'] >= ASK_LATENT_FREQUENCY
                      else False)
      update_html_canvas(dict_update, env_id, draw_overlay, SHOW_FAILURE,
                         name_space)
    else:
      game.reset_game()
      on_game_end(env_id, name_space)


def set_latent(msg, name_space, id_2_game: Mapping[Hashable, BoxPushSimulator]):
  env_id = request.sid
  latent = msg["data"]

  game = id_2_game[env_id]
  game.event_input(BoxPushSimulator.AGENT1, EventType.SET_LATENT, latent)

  dict_update = game.get_changed_objects()
  session['action_count'] = 0
  update_html_canvas(dict_update, env_id, NOT_ASK_LATENT, NOT_SHOW_FAILURE,
                     name_space)
