from typing import Mapping, Hashable
import random
from ai_coach_domain.box_push import EventType
from ai_coach_domain.box_push.simulator import BoxPushSimulator_AlwaysAlone
from ai_coach_domain.box_push.maps import EXP1_MAP
from ai_coach_domain.box_push.agent_mdp import (BoxPushAgentMDP_AlwaysAlone)
from web_experiment import socketio
import web_experiment.experiment1.events_impl as event_impl

g_id_2_game = {}  # type: Mapping[Hashable, BoxPushSimulator_AlwaysAlone]
EXP1_NAMESPACE = '/exp1_indv_user_random'
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
  def set_init_latent(game: BoxPushSimulator_AlwaysAlone):
    valid_boxes = event_impl.get_valid_box_to_pickup(game)
    if len(valid_boxes) > 0:
      box_idx = random.choice(valid_boxes)
      game.event_input(AGENT2, EventType.SET_LATENT, ("pickup", box_idx))

  event_impl.run_task_B_game(msg, g_id_2_game, set_init_latent, EXP1_MDP,
                             EXP1_MAP, event_impl.ASK_LATENT, EXP1_NAMESPACE)


@socketio.on('action_event', namespace=EXP1_NAMESPACE)
def action_event(msg):
  def game_finished(game, env_id, name_space):
    session_name = "session_b3"
    cur_user = msg["user_id"]
    event_impl.task_end(env_id, game, cur_user, session_name,
                        "BoxPushSimulator_AlwaysAlone", EXP1_MAP, name_space,
                        False)

  def hold_changed(game, a1_hold_changed, a2_hold_changed, a1_box, a2_box):
    a2_latent_prev = game.a2_latent

    a1_pickup = a1_hold_changed and (a1_box >= 0)
    a2_pickup = a2_hold_changed and (a2_box >= 0)
    a2_drop = a2_hold_changed and not (a2_box >= 0)

    if a2_pickup:
      game.event_input(AGENT2, EventType.SET_LATENT, ("goal", 0))

    elif a2_drop:
      valid_boxes = event_impl.get_valid_box_to_pickup(game)
      if len(valid_boxes) > 0:
        box_idx = random.choice(valid_boxes)
        game.event_input(AGENT2, EventType.SET_LATENT, ("pickup", box_idx))
      else:
        game.event_input(AGENT2, EventType.SET_LATENT, ("pickup", a1_box))

    elif a1_pickup:
      # a2 has no box and was targetting the same box that a1 picked up
      # --> set to another box
      if a2_box < 0 and a1_box == a2_latent_prev[1]:
        valid_boxes = event_impl.get_valid_box_to_pickup(game)
        if len(valid_boxes) > 0:
          box_idx = random.choice(valid_boxes)
          game.event_input(AGENT2, EventType.SET_LATENT, ("pickup", box_idx))
        else:
          game.event_input(AGENT2, EventType.SET_LATENT, ("pickup", a1_box))

  ASK_LATENT_FREQUENCY = 5
  event_impl.action_event(msg, g_id_2_game, hold_changed, game_finished,
                          EXP1_NAMESPACE, True, True, ASK_LATENT_FREQUENCY)


@socketio.on('set_latent', namespace=EXP1_NAMESPACE)
def set_latent(msg):
  event_impl.set_latent(msg, g_id_2_game, EXP1_NAMESPACE)
