from typing import Mapping, Hashable
import copy
from flask import request, session
from ai_coach_domain.box_push import (EventType, BoxState, conv_box_state_2_idx)
from ai_coach_domain.box_push import BoxPushSimulator_AlwaysAlone
from ai_coach_domain.box_push.box_push_maps import TUTORIAL_MAP
from ai_coach_domain.box_push.box_push_policy import get_simple_action
from web_experiment import socketio
import web_experiment.experiment1.events_impl as event_impl

g_id_2_game = {}  # type: Mapping[Hashable, BoxPushSimulator_AlwaysAlone]
EXP1_TUT_NAMESPACE = '/exp1_tutorial2'
GRID_X = TUTORIAL_MAP["x_grid"]
GRID_Y = TUTORIAL_MAP["y_grid"]

AGENT1 = BoxPushSimulator_AlwaysAlone.AGENT1
AGENT2 = BoxPushSimulator_AlwaysAlone.AGENT2
GAME_MAP = TUTORIAL_MAP


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
  env_id = request.sid

  # run a game
  if env_id not in g_id_2_game:
    g_id_2_game[env_id] = BoxPushSimulator_AlwaysAlone(env_id)

  game = g_id_2_game[env_id]
  game.init_game(**GAME_MAP)
  game.set_autonomous_agent()

  game.event_input(AGENT1, EventType.SET_LATENT, ("pickup", 2))
  dict_update = game.get_env_info()
  dict_update["wall_dir"] = GAME_MAP["wall_dir"]
  if dict_update is not None:
    session['action_count'] = 0
    event_impl.update_html_canvas(dict_update, env_id,
                                  event_impl.NOT_ASK_LATENT, EXP1_TUT_NAMESPACE)


@socketio.on('box_pickup_scenario', namespace=EXP1_TUT_NAMESPACE)
def box_pickup_scenario(msg):
  env_id = request.sid

  # run a game
  if env_id not in g_id_2_game:
    g_id_2_game[env_id] = BoxPushSimulator_AlwaysAlone(env_id)

  game = g_id_2_game[env_id]
  game.init_game(**GAME_MAP)
  game.set_autonomous_agent(
      cb_get_A2_action=lambda **kwargs: get_simple_action(AGENT2, **kwargs))

  ask_latent = False
  if msg['data'] == 'ask_latent':
    ask_latent = True
    game.event_input(AGENT2, EventType.SET_LATENT, ("goal", 0))
  else:
    game.event_input(AGENT1, EventType.SET_LATENT, ("pickup", 2))
    game.event_input(AGENT2, EventType.SET_LATENT, ("pickup", 2))

  dict_update = game.get_env_info()
  dict_update["wall_dir"] = GAME_MAP["wall_dir"]
  if dict_update is not None:
    session['action_count'] = 0

    event_impl.update_html_canvas(dict_update, env_id, ask_latent,
                                  EXP1_TUT_NAMESPACE)


@socketio.on('action_event', namespace=EXP1_TUT_NAMESPACE)
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

    game.event_input(AGENT1, action, None)

    map_agent2action = game.get_joint_action()
    game.take_a_step(map_agent2action)
    session['action_count'] = session.get('action_count', 0) + 1

    if not game.is_finished():
      (a1_pos_changed, a2_pos_changed, a1_hold_changed, a2_hold_changed, a1_box,
       _) = event_impl.are_agent_states_changed(dict_env_prev, game)
      unchanged_agents = []
      if not a1_pos_changed and not a1_hold_changed:
        unchanged_agents.append(0)
      if not a2_pos_changed and not a2_hold_changed:
        unchanged_agents.append(1)

      dict_update = game.get_changed_objects()
      if dict_update is None:
        dict_update = {}

      dict_update["unchanged_agents"] = unchanged_agents
      if a1_hold_changed:
        game.event_input(AGENT1, EventType.SET_LATENT, None)
        dict_update["a1_latent"] = None

      event_impl.update_html_canvas(dict_update, env_id,
                                    event_impl.NOT_ASK_LATENT,
                                    EXP1_TUT_NAMESPACE)
    else:
      game.reset_game()
      box_pickup_scenario({'data': ''})


@socketio.on('set_latent', namespace=EXP1_TUT_NAMESPACE)
def set_latent(msg):
  env_id = request.sid
  latent = msg["data"]

  game = g_id_2_game[env_id]
  game.event_input(AGENT1, EventType.SET_LATENT, tuple(latent))
  game.event_input(AGENT2, EventType.SET_LATENT, tuple(latent))

  dict_update = game.get_changed_objects()
  event_impl.update_html_canvas(dict_update, env_id, event_impl.NOT_ASK_LATENT,
                                EXP1_TUT_NAMESPACE)


@socketio.on('trapped_scenario', namespace=EXP1_TUT_NAMESPACE)
def trapped_scenario(msg):
  env_id = request.sid

  # run a game
  if env_id not in g_id_2_game:
    g_id_2_game[env_id] = BoxPushSimulator_AlwaysAlone(env_id)

  game = g_id_2_game[env_id]
  game.init_game(**GAME_MAP)
  game.set_autonomous_agent()

  # make scenario
  bidx1 = 1
  game.a1_pos = game.boxes[bidx1]
  game.box_states[bidx1] = conv_box_state_2_idx((BoxState.WithAgent1, None),
                                                len(game.drops))

  bidx2 = 0
  game.box_states[bidx2] = conv_box_state_2_idx((BoxState.WithAgent2, None),
                                                len(game.drops))
  game.a2_pos = game.boxes[bidx2]

  game.event_input(AGENT1, EventType.SET_LATENT, ("goal", 0))
  dict_update = game.get_env_info()

  if dict_update is not None:
    event_impl.update_html_canvas(dict_update, env_id,
                                  event_impl.NOT_ASK_LATENT, EXP1_TUT_NAMESPACE)
