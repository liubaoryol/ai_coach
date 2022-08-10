from typing import Mapping, Hashable
import random
from ai_coach_domain.box_push import EventType
from ai_coach_domain.box_push.simulator import BoxPushSimulator_AlwaysAlone
from ai_coach_domain.box_push.maps import EXP1_MAP
from ai_coach_domain.box_push.mdp import (BoxPushAgentMDP_AlwaysAlone,
                                          BoxPushTeamMDP_AlwaysAlone)
from ai_coach_domain.box_push.mdppolicy import BoxPushPolicyIndvExp1
from ai_coach_domain.box_push.agent import BoxPushAIAgent_Host
from web_experiment import socketio
import web_experiment.experiment1.events_impl as event_impl

g_id_2_game = {}  # type: Mapping[Hashable, BoxPushSimulator_AlwaysAlone]
EXP1_NAMESPACE = '/exp1_indv_tell_align'
SESSION_NAME = "session_b1"
TASK_TYPE = event_impl.TASK_B

EXP1_MDP = BoxPushAgentMDP_AlwaysAlone(**EXP1_MAP)
EXP1_TASK_MDP = BoxPushTeamMDP_AlwaysAlone(**EXP1_MAP)

AGENT1 = BoxPushSimulator_AlwaysAlone.AGENT1
AGENT2 = BoxPushSimulator_AlwaysAlone.AGENT2

TEMPERATURE = 0.3
TEAMMATE_POLICY = BoxPushPolicyIndvExp1(EXP1_TASK_MDP, EXP1_MDP, TEMPERATURE,
                                        AGENT2)


@socketio.on('connect', namespace=EXP1_NAMESPACE)
def initial_canvas():
  event_impl.initial_canvas(SESSION_NAME, TASK_TYPE)


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
  agent2 = BoxPushAIAgent_Host(TEAMMATE_POLICY)

  def set_init_latent(game: BoxPushSimulator_AlwaysAlone):
    valid_boxes = event_impl.get_valid_box_to_pickup(game)
    if len(valid_boxes) > 0:
      box_idx = random.choice(valid_boxes)
      game.event_input(AGENT1, EventType.SET_LATENT, ("pickup", box_idx))
      game.event_input(AGENT2, EventType.SET_LATENT, ("pickup", box_idx))

  event_impl.run_task_game(msg, g_id_2_game, agent2, set_init_latent, EXP1_MAP,
                           event_impl.NOT_ASK_LATENT, EXP1_NAMESPACE, TASK_TYPE)


@socketio.on('action_event', namespace=EXP1_NAMESPACE)
def action_event(msg):
  def hold_changed(game: BoxPushSimulator_AlwaysAlone, a1_hold_changed,
                   a2_hold_changed, a1_box, a2_box):
    a1_latent_prev = game.agent_1.get_current_latent()
    a2_latent_prev = game.agent_2.get_current_latent()

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
      else:
        game.event_input(AGENT2, EventType.SET_LATENT, ("pickup", a1_box))

    elif a1_drop and a2_pickup:
      valid_boxes = event_impl.get_valid_box_to_pickup(game)
      if len(valid_boxes) > 0:
        box_idx = random.choice(valid_boxes)
        game.event_input(AGENT1, EventType.SET_LATENT, ("pickup", box_idx))
      else:
        game.event_input(AGENT1, EventType.SET_LATENT, ("pickup", a2_box))
      game.event_input(AGENT2, EventType.SET_LATENT, ("goal", 0))

    elif a1_drop and a2_drop:  # TODO: agent1 follows agent2 latent
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
        else:
          game.event_input(AGENT1, EventType.SET_LATENT, ("pickup", a2_box))
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
        else:
          game.event_input(AGENT2, EventType.SET_LATENT, ("pickup", a1_box))
      # a1 has no box --> set to the same box a1 is aiming for
      else:
        game.event_input(AGENT2, EventType.SET_LATENT,
                         ("pickup", a1_latent_prev[1]))

  ASK_LATENT_FREQUENCY = 5
  event_impl.action_event(msg, g_id_2_game, hold_changed, event_impl.game_end,
                          EXP1_NAMESPACE, False, False, ASK_LATENT_FREQUENCY,
                          EXP1_MAP, SESSION_NAME, TASK_TYPE)


@socketio.on('setting_event', namespace=EXP1_NAMESPACE)
def setting_event(msg):
  event_impl.setting_event(msg, g_id_2_game, EXP1_NAMESPACE)


@socketio.on('done_task', namespace=EXP1_NAMESPACE)
def done_task(msg):
  event_impl.done_task(msg, SESSION_NAME)
