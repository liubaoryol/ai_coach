from typing import Mapping, Hashable
import random
import copy
from flask import request
from ai_coach_domain.box_push import EventType
from ai_coach_domain.box_push import BoxPushSimulator_AlwaysAlone
from ai_coach_domain.box_push.box_push_maps import EXP1_MAP
from ai_coach_domain.box_push.box_push_policy import get_indv_action
from ai_coach_domain.box_push.box_push_agent_mdp import (
    BoxPushAgentMDP_AlwaysAlone)
from web_experiment import socketio
import web_experiment.experiment1.events_impl as event_impl

g_id_2_game = {}  # type: Mapping[Hashable, BoxPushSimulator_AlwaysAlone]
EXP1_NAMESPACE = '/exp1_indv_tell_align'
GRID_X = EXP1_MAP["x_grid"]
GRID_Y = EXP1_MAP["y_grid"]
EXP1_MDP = BoxPushAgentMDP_AlwaysAlone(**EXP1_MAP)

AGENT1 = BoxPushSimulator_AlwaysAlone.AGENT1
AGENT2 = BoxPushSimulator_AlwaysAlone.AGENT2


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
    g_id_2_game[env_id] = BoxPushSimulator_AlwaysAlone(env_id)

  game = g_id_2_game[env_id]
  game.init_game(**EXP1_MAP)
  temperature = 0.3
  game.set_autonomous_agent(cb_get_A2_action=lambda **kwargs: get_indv_action(
      EXP1_MDP, AGENT2, temperature, **kwargs))

  valid_boxes = event_impl.get_valid_box_to_pickup(game)
  if len(valid_boxes) > 0:
    box_idx = random.choice(valid_boxes)
    game.event_input(AGENT1, EventType.SET_LATENT, ("pickup", box_idx))
    game.event_input(AGENT2, EventType.SET_LATENT, ("pickup", box_idx))

  dict_update = game.get_env_info()
  dict_update["wall_dir"] = EXP1_MAP["wall_dir"]
  if dict_update is not None:
    event_impl.update_html_canvas(dict_update, env_id,
                                  event_impl.NOT_ASK_LATENT, EXP1_NAMESPACE)


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
      (a1_pos_changed, a2_pos_changed, a1_hold_changed, a2_hold_changed, a1_box,
       a2_box) = event_impl.are_agent_states_changed(dict_env_prev, game)
      unchanged_agents = []
      if not a1_pos_changed and not a1_hold_changed:
        unchanged_agents.append(0)
      if not a2_pos_changed and not a2_hold_changed:
        unchanged_agents.append(1)

      a1_latent_prev = game.a1_latent
      a2_latent_prev = game.a2_latent

      a1_pickup = a1_hold_changed and (a1_box >= 0)
      a1_drop = a1_hold_changed and not (a1_box >= 0)
      a2_pickup = a2_hold_changed and (a2_box >= 0)
      a2_drop = a2_hold_changed and not (a2_box >= 0)

      if a1_pickup and a2_pickup:
        game.event_input(AGENT1, EventType.SET_LATENT, ("goal", 0))
        game.event_input(AGENT2, EventType.SET_LATENT, ("goal", 0))
      elif a1_pickup and a2_drop:
        game.event_input(AGENT1, EventType.SET_LATENT, ("goal", 0))

        valid_boxes = event_impl.get_valid_box_to_pickup(game)
        if len(valid_boxes) > 0:
          box_idx = random.choice(valid_boxes)
          game.event_input(AGENT2, EventType.SET_LATENT, ("pickup", box_idx))
      elif a1_drop and a2_pickup:
        valid_boxes = event_impl.get_valid_box_to_pickup(game)
        if len(valid_boxes) > 0:
          box_idx = random.choice(valid_boxes)
          game.event_input(AGENT1, EventType.SET_LATENT, ("pickup", box_idx))
        game.event_input(AGENT2, EventType.SET_LATENT, ("goal", 0))
      elif a1_drop and a2_drop:
        valid_boxes = event_impl.get_valid_box_to_pickup(game)
        if len(valid_boxes) > 0:
          box_idx = random.choice(valid_boxes)
          game.event_input(AGENT1, EventType.SET_LATENT, ("pickup", box_idx))
          game.event_input(AGENT2, EventType.SET_LATENT, ("pickup", box_idx))
      elif a1_pickup:
        game.event_input(AGENT1, EventType.SET_LATENT, ("goal", 0))
        # a2 has no box and was targetting the same box that a1 picked up
        # --> set to another box
        if a2_box < 0 and a1_box == a2_latent_prev[1]:
          valid_boxes = event_impl.get_valid_box_to_pickup(game)
          if len(valid_boxes) > 0:
            box_idx = random.choice(valid_boxes)
            game.event_input(AGENT2, EventType.SET_LATENT, ("pickup", box_idx))
      elif a1_drop:
        # a2 has a box --> set to another box
        if a2_box >= 0:
          valid_boxes = event_impl.get_valid_box_to_pickup(game)
          if len(valid_boxes) > 0:
            box_idx = random.choice(valid_boxes)
            game.event_input(AGENT1, EventType.SET_LATENT, ("pickup", box_idx))
        # a2 has no box --> set to the same box a2 is aiming for
        else:
          game.event_input(AGENT1, EventType.SET_LATENT,
                           ("pickup", a2_latent_prev[1]))
      elif a2_pickup:
        game.event_input(AGENT2, EventType.SET_LATENT, ("goal", 0))
        # a1 has no box and was targetting the same box that a2 just picked up
        # --> set to another box
        if a1_box < 0 and a2_box == a1_latent_prev[1]:
          valid_boxes = event_impl.get_valid_box_to_pickup(game)
          if len(valid_boxes) > 0:
            box_idx = random.choice(valid_boxes)
            game.event_input(AGENT1, EventType.SET_LATENT, ("pickup", box_idx))
      elif a2_drop:
        # a1 has a box --> set to another box
        if a1_box >= 0:
          valid_boxes = event_impl.get_valid_box_to_pickup(game)
          if len(valid_boxes) > 0:
            box_idx = random.choice(valid_boxes)
            game.event_input(AGENT2, EventType.SET_LATENT, ("pickup", box_idx))
        # a1 has no box --> set to the same box a1 is aiming for
        else:
          game.event_input(AGENT2, EventType.SET_LATENT,
                           ("pickup", a1_latent_prev[1]))

      dict_update = game.get_changed_objects()
      if dict_update is None:
        dict_update = {}

      dict_update["unchanged_agents"] = unchanged_agents

      event_impl.update_html_canvas(dict_update, env_id,
                                    event_impl.NOT_ASK_LATENT, EXP1_NAMESPACE)
    else:
      game.reset_game()
      event_impl.on_game_end(env_id, EXP1_NAMESPACE)


@socketio.on('set_latent', namespace=EXP1_NAMESPACE)
def set_latent(msg):
  env_id = request.sid
  latent = msg["data"]

  game = g_id_2_game[env_id]
  game.event_input(AGENT1, EventType.SET_LATENT, tuple(latent))

  dict_update = game.get_changed_objects()
  event_impl.update_html_canvas(dict_update, env_id, event_impl.NOT_ASK_LATENT,
                                EXP1_NAMESPACE)
