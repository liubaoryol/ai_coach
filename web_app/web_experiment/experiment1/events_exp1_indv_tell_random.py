from typing import Mapping, Hashable
import random
import copy
from flask import request
from ai_coach_domain.box_push import EventType
from ai_coach_domain.box_push import (BoxPushSimulator_AloneOrTogether as
                                      BoxPushSimulator)
from ai_coach_domain.box_push.box_push_maps import EXP1_MAP
from ai_coach_domain.box_push.box_push_policy import get_indv_action
from ai_coach_domain.box_push.box_push_mdp_individual import BoxPushMDP_Indv
from web_experiment import socketio
import web_experiment.experiment1.events_impl as event_impl

g_id_2_game = {}  # type: Mapping[Hashable, BoxPushSimulator]
EXP1_NAMESPACE = '/exp1_indv_tell_random'
GRID_X = EXP1_MAP["x_grid"]
GRID_Y = EXP1_MAP["y_grid"]
EXP1_MDP = BoxPushMDP_Indv(**EXP1_MAP)


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
  game.init_game(**EXP1_MAP)
  temperature = 0.3
  game.set_autonomous_agent(cb_get_A2_action=lambda **kwargs: get_indv_action(
      EXP1_MDP, BoxPushSimulator.AGENT2, temperature, **kwargs))

  valid_boxes = event_impl.get_valid_box_to_pickup(game)
  box_idx = None
  if len(valid_boxes) > 0:
    box_idx = random.choice(valid_boxes)

  game.event_input(BoxPushSimulator.AGENT1, EventType.SET_LATENT,
                   ("pickup", box_idx))
  box_idx2 = None
  for idx in valid_boxes:
    if idx != box_idx:
      box_idx2 = idx
  game.event_input(BoxPushSimulator.AGENT2, EventType.SET_LATENT,
                   ("pickup", box_idx2))

  dict_update = game.get_env_info()
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

    game.event_input(BoxPushSimulator.AGENT1, action, None)
    map_agent2action = game.get_joint_action()
    game.take_a_step(map_agent2action)

    if not game.is_finished():
      (a1_pos_changed, a2_pos_changed, a1_hold_changed, a2_hold_changed,
       a1_hold,
       a2_hold) = event_impl.are_agent_states_changed(dict_env_prev, game)
      unchanged_agents = []
      if not a1_pos_changed and not a1_hold_changed:
        unchanged_agents.append(0)
      if not a2_pos_changed and not a2_hold_changed:
        unchanged_agents.append(1)

      a1_pickup = a1_hold_changed and a1_hold
      a1_dropped = a1_hold_changed and not a1_hold
      a2_pickup = a2_hold_changed and a2_hold
      a2_dropped = a2_hold_changed and not a2_hold

      if a1_pickup:
        game.event_input(BoxPushSimulator.AGENT1, EventType.SET_LATENT,
                         ("goal", 0))

      if a1_dropped:
        valid_boxes = event_impl.get_valid_box_to_pickup(game)
        a2_box = None
        for idx, bidx in enumerate(game.box_states):
          if bidx == 2:
            a2_box = idx
            break

        box_idx = None
        if a2_box is None:
          box_idx = random.choice(valid_boxes)

        if box_idx is None:
          for idx in valid_boxes:
            if idx != a2_box:
              box_idx = idx
              break

        game.event_input(BoxPushSimulator.AGENT1, EventType.SET_LATENT,
                         ("pickup", box_idx))

      if a2_pickup:
        game.event_input(BoxPushSimulator.AGENT2, EventType.SET_LATENT,
                         ("goal", 0))

      if a2_dropped:
        valid_boxes = event_impl.get_valid_box_to_pickup(game)
        a1_box = None
        for idx, bidx in enumerate(game.box_states):
          if bidx == 1:
            a1_box = idx
            break

        box_idx = None
        if a1_box is None:
          box_idx = random.choice(valid_boxes)

        if box_idx is None:
          for idx in valid_boxes:
            if idx != a1_box:
              box_idx = idx

        game.event_input(BoxPushSimulator.AGENT2, EventType.SET_LATENT,
                         ("pickup", box_idx))

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
  game.event_input(BoxPushSimulator.AGENT1, EventType.SET_LATENT, tuple(latent))
  game.event_input(BoxPushSimulator.AGENT2, EventType.SET_LATENT, tuple(latent))

  dict_update = game.get_changed_objects()
  event_impl.update_html_canvas(dict_update, env_id, event_impl.NOT_ASK_LATENT,
                                EXP1_NAMESPACE)
