from typing import Mapping, Hashable
import random
import copy
import logging
from flask import session, request
from ai_coach_domain.box_push import EventType
from ai_coach_domain.box_push import BoxPushSimulator_AlwaysTogether
from ai_coach_domain.box_push.box_push_maps import EXP1_MAP
from ai_coach_domain.box_push.box_push_policy import get_exp1_action
from ai_coach_domain.box_push.box_push_team_mdp import (
    BoxPushTeamMDP_AlwaysTogether)
from web_experiment import socketio
import web_experiment.experiment1.events_impl as event_impl

g_id_2_game = {}  # type: Mapping[Hashable, BoxPushSimulator_AlwaysTogether]
EXP1_NAMESPACE = '/exp1_both_user_random_2'
GRID_X = EXP1_MAP["x_grid"]
GRID_Y = EXP1_MAP["y_grid"]
EXP1_MDP = BoxPushTeamMDP_AlwaysTogether(**EXP1_MAP)

AGENT1 = BoxPushSimulator_AlwaysTogether.AGENT1
AGENT2 = BoxPushSimulator_AlwaysTogether.AGENT2


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
    g_id_2_game[env_id] = BoxPushSimulator_AlwaysTogether(env_id)

  game = g_id_2_game[env_id]
  game.init_game(**EXP1_MAP)
  temperature = 0.3
  game.set_autonomous_agent(cb_get_A2_action=lambda **kwargs: get_exp1_action(
      EXP1_MDP, AGENT2, temperature, **kwargs))

  valid_boxes = event_impl.get_valid_box_to_pickup(game)
  if len(valid_boxes) > 0:
    box_idx = random.choice(valid_boxes)
    game.event_input(AGENT2, EventType.SET_LATENT, ("pickup", box_idx))

  dict_update = game.get_env_info()
  dict_update["wall_dir"] = EXP1_MAP["wall_dir"]
  dict_update["best_score"] = event_impl.get_best_score(msg["user_id"], True)
  if dict_update is not None:
    session["action_count"] = 0
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

    game.event_input(AGENT1, action, None)
    map_agent2action = game.get_joint_action()
    game.take_a_step(map_agent2action)

    if not game.is_finished():
      (a1_pos_changed, a2_pos_changed, a1_hold_changed, a2_hold_changed, _,
       a2_box) = event_impl.are_agent_states_changed(dict_env_prev, game)
      unchanged_agents = []
      if not a1_pos_changed and not a1_hold_changed:
        unchanged_agents.append(0)
      if not a2_pos_changed and not a2_hold_changed:
        unchanged_agents.append(1)

      draw_overlay = False
      if a1_hold_changed:
        draw_overlay = True

      if a2_hold_changed:
        if a2_box >= 0:
          game.event_input(AGENT2, EventType.SET_LATENT, ("goal", 0))
        else:
          valid_boxes = event_impl.get_valid_box_to_pickup(game)
          if len(valid_boxes) > 0:
            box_idx = random.choice(valid_boxes)
            game.event_input(AGENT2, EventType.SET_LATENT, ("pickup", box_idx))
      else:
        if a2_box < 0:
          event_impl.change_a2_latent_based_on_a1(game)

      dict_update = game.get_changed_objects()
      if dict_update is None:
        dict_update = {}

      dict_update["unchanged_agents"] = unchanged_agents

      ASK_LATENT_FREQUENCY = 5
      session['action_count'] = session.get('action_count', 0) + 1
      if session['action_count'] >= ASK_LATENT_FREQUENCY:
        draw_overlay = True

      event_impl.update_html_canvas(dict_update, env_id, draw_overlay,
                                    EXP1_NAMESPACE)
    else:
      session_name = "session_a4"
      cur_user = msg["user_id"]
      file_name = event_impl.get_file_name(cur_user, session_name)
      header = "BoxPushSimulator_AlwaysTogether\n"
      header += "User ID: %s\n" % (str(cur_user), )
      header += str(EXP1_MAP)
      game.save_history(file_name, header)

      event_impl.on_game_end(env_id, EXP1_NAMESPACE, cur_user, session_name,
                             game.current_step, True)

      game.reset_game()
      logging.info("User %s completed %s" % (cur_user, session_name))


@socketio.on('set_latent', namespace=EXP1_NAMESPACE)
def set_latent(msg):
  env_id = request.sid
  latent = msg["data"]

  game = g_id_2_game[env_id]
  game.event_input(AGENT1, EventType.SET_LATENT, tuple(latent))

  dict_update = game.get_changed_objects()
  session['action_count'] = 0
  event_impl.update_html_canvas(dict_update, env_id, event_impl.NOT_ASK_LATENT,
                                EXP1_NAMESPACE)
