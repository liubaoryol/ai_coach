from typing import Mapping, Hashable
import copy
from flask import session, request
from ai_coach_domain.box_push import BoxPushSimulator, EventType
from web_experiment import socketio
import web_experiment.experiment1.events_impl as event_impl

g_id_2_game = {}  # type: Mapping[Hashable, BoxPushSimulator]
EXP1_NAMESPACE = '/experiment1'
GRID_X = 10
GRID_Y = 10


@socketio.on('connect', namespace=EXP1_NAMESPACE)
def initial_canvas():
  event_impl.initial_canvas(GRID_X, GRID_Y)


@socketio.on('my_echo', namespace=EXP1_NAMESPACE)
def test_message(message):
  event_impl.test_message(message)


@socketio.on('disconnect_request', namespace=EXP1_NAMESPACE)
def disconnect_request():
  event_impl.disconnect_request()


@socketio.on('my_ping', namespace=EXP1_NAMESPACE)
def ping_pong():
  event_impl.ping_pong()


@socketio.on('disconnect', namespace=EXP1_NAMESPACE)
def test_disconnect():
  event_impl.test_disconnect(g_id_2_game)


@socketio.on('run_game', namespace=EXP1_NAMESPACE)
def run_game(msg):
  env_id = request.sid

  # run a game
  if env_id not in g_id_2_game:
    g_id_2_game[env_id] = BoxPushSimulator(env_id)

  game = g_id_2_game[env_id]
  game.init_game_with_test_map(GRID_X, GRID_Y)
  dict_update = game.get_env_info()
  if dict_update is not None:
    session['action_count'] = 0
    event_impl.update_html_canvas(dict_update, env_id, event_impl.ASK_LATENT,
                                  EXP1_NAMESPACE)


@socketio.on('action_event', namespace=EXP1_NAMESPACE)
def action_event(msg):
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
    dict_env_prev = copy.deepcopy(game.get_env_info())

    game.event_input(BoxPushSimulator.AGENT1, action, None)
    map_agent2action = game.get_action()
    game.take_a_step(map_agent2action)

    if not game.is_finished():
      dict_update = game.get_changed_objects()
      if dict_update is None:
        dict_update = {}

      a1_pos_changed, a2_pos_changed, a1_hold_changed, a2_hold_changed = (
          event_impl.are_agent_states_changed(dict_env_prev, dict_update))
      unchanged_agents = []
      if not a1_pos_changed and not a1_hold_changed:
        unchanged_agents.append(0)
      if not a2_pos_changed and not a2_hold_changed:
        unchanged_agents.append(1)

      dict_update["unchanged_agents"] = unchanged_agents

      session['action_count'] = session.get('action_count', 0) + 1
      ASK_LATENT_FREQUENCY = 5

      draw_overlay = (True if session['action_count'] >= ASK_LATENT_FREQUENCY
                      else False)

      if a1_hold_changed:
        draw_overlay = True

      event_impl.update_html_canvas(dict_update, env_id, draw_overlay,
                                    EXP1_NAMESPACE)
    else:
      game.reset_game()
      event_impl.on_game_end(env_id, EXP1_NAMESPACE)


@socketio.on('set_latent', namespace=EXP1_NAMESPACE)
def set_latent(msg):
  env_id = request.sid
  latent = msg["data"]

  game = g_id_2_game[env_id]
  game.event_input(BoxPushSimulator.AGENT1, EventType.SET_LATENT, latent)

  dict_update = game.get_changed_objects()
  session['action_count'] = 0
  event_impl.update_html_canvas(dict_update, env_id, event_impl.NOT_ASK_LATENT,
                                EXP1_NAMESPACE)
