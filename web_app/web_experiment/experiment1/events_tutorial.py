from typing import Mapping, Hashable
from flask import request
from ai_coach_domain.box_push import (BoxPushSimulator, EventType, BoxState,
                                      conv_box_state_2_idx)
from web_experiment import socketio
import web_experiment.experiment1.events_impl as event_impl

g_id_2_game = {}  # type: Mapping[Hashable, BoxPushSimulator]
EXP1_TUT_NAMESPACE = '/exp1_tutorial'
GRID_X = 6
GRID_Y = 6


@socketio.on('connect', namespace=EXP1_TUT_NAMESPACE)
def initial_canvas():
  event_impl.initial_canvas(GRID_X, GRID_Y)


@socketio.on('my_echo', namespace=EXP1_TUT_NAMESPACE)
def test_message(message):
  event_impl.test_message(message)


@socketio.on('disconnect_request', namespace=EXP1_TUT_NAMESPACE)
def disconnect_request():
  event_impl.disconnect_request()


@socketio.on('my_ping', namespace=EXP1_TUT_NAMESPACE)
def ping_pong():
  event_impl.ping_pong()


@socketio.on('disconnect', namespace=EXP1_TUT_NAMESPACE)
def test_disconnect():
  event_impl.test_disconnect(g_id_2_game)


@socketio.on('run_game', namespace=EXP1_TUT_NAMESPACE)
def run_game(msg):
  game_map = {
      "boxes": [(0, 1), (3, 1)],
      "goals": [(GRID_X - 1, GRID_Y - 1)],
      "walls": [(GRID_X - 2, GRID_Y - i - 1) for i in range(3)],
      "wall_dir": [0 for dummy_i in range(3)],
      "drops": []
  }
  env_id = request.sid

  # run a game
  if env_id not in g_id_2_game:
    g_id_2_game[env_id] = BoxPushSimulator(env_id)

  game = g_id_2_game[env_id]
  game.init_game(GRID_X,
                 GRID_Y,
                 boxes=game_map["boxes"],
                 goals=game_map["goals"],
                 walls=game_map["walls"],
                 wall_dir=game_map["wall_dir"],
                 drops=game_map["drops"])

  game.event_input(BoxPushSimulator.AGENT1, EventType.SET_LATENT, ("box", 0))
  dict_update = game.get_env_info()

  if dict_update is not None:
    event_impl.update_html_canvas(dict_update, env_id,
                                  event_impl.NOT_ASK_LATENT,
                                  event_impl.NOT_SHOW_FAILURE,
                                  EXP1_TUT_NAMESPACE)


@socketio.on('action_event', namespace=EXP1_TUT_NAMESPACE)
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
    game = g_id_2_game[env_id]
    game.event_input(BoxPushSimulator.AGENT1, action, None)
    map_agent2action = game.get_action()
    game.take_a_step(map_agent2action)

    if game.is_finished():
      game.reset_game()

    dict_update = game.get_changed_objects()
    if dict_update is None:
      dict_update = {}

    event_impl.update_html_canvas(dict_update, env_id,
                                  event_impl.NOT_ASK_LATENT,
                                  event_impl.SHOW_FAILURE, EXP1_TUT_NAMESPACE)


@socketio.on('set_latent', namespace=EXP1_TUT_NAMESPACE)
def set_latent(msg):
  env_id = request.sid
  latent = msg["data"]

  game = g_id_2_game[env_id]
  game.event_input(BoxPushSimulator.AGENT1, EventType.SET_LATENT, latent)

  dict_update = game.get_changed_objects()
  event_impl.update_html_canvas(dict_update, env_id, event_impl.NOT_ASK_LATENT,
                                event_impl.NOT_SHOW_FAILURE, EXP1_TUT_NAMESPACE)


@socketio.on('help_teammate', namespace=EXP1_TUT_NAMESPACE)
def help_teammate(msg):
  game_map = {
      "boxes": [(0, 1), (3, 1)],
      "goals": [(GRID_X - 1, GRID_Y - 1)],
      "walls": [(GRID_X - 2, GRID_Y - i - 1) for i in range(3)],
      "wall_dir": [0 for dummy_i in range(3)],
      "drops": []
  }
  env_id = request.sid

  # run a game
  if env_id not in g_id_2_game:
    g_id_2_game[env_id] = BoxPushSimulator(env_id)

  game = g_id_2_game[env_id]
  game.init_game(GRID_X,
                 GRID_Y,
                 boxes=game_map["boxes"],
                 goals=game_map["goals"],
                 walls=game_map["walls"],
                 wall_dir=game_map["wall_dir"],
                 drops=game_map["drops"])

  # make scenario
  game.box_states[0] = conv_box_state_2_idx((BoxState.WithAgent2, None),
                                            len(game.drops))
  game.event_input(BoxPushSimulator.AGENT1, EventType.SET_LATENT, ("box", 0))
  dict_update = game.get_env_info()

  if dict_update is not None:
    event_impl.update_html_canvas(dict_update, env_id,
                                  event_impl.NOT_ASK_LATENT,
                                  event_impl.NOT_SHOW_FAILURE,
                                  EXP1_TUT_NAMESPACE)
