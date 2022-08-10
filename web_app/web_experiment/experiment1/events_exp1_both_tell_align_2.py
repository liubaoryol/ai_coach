from typing import Mapping, Hashable
import random
from ai_coach_domain.box_push import EventType
from ai_coach_domain.box_push.simulator import BoxPushSimulator_AlwaysTogether
from ai_coach_domain.box_push.maps import EXP1_MAP
from ai_coach_domain.box_push.mdp import BoxPushTeamMDP_AlwaysTogether
from ai_coach_domain.box_push.mdppolicy import BoxPushPolicyTeamExp1
from ai_coach_domain.box_push.agent import BoxPushAIAgent_Host
from web_experiment import socketio
import web_experiment.experiment1.events_impl as event_impl

g_id_2_game = {}  # type: Mapping[Hashable, BoxPushSimulator_AlwaysTogether]
EXP1_NAMESPACE = '/exp1_both_tell_align_2'
SESSION_NAME = "session_a2"
TASK_TYPE = event_impl.TASK_A

EXP1_MDP = BoxPushTeamMDP_AlwaysTogether(**EXP1_MAP)
AGENT1 = BoxPushSimulator_AlwaysTogether.AGENT1
AGENT2 = BoxPushSimulator_AlwaysTogether.AGENT2

TEMPERATURE = 0.3
TEAMMATE_POLICY = BoxPushPolicyTeamExp1(EXP1_MDP, TEMPERATURE, AGENT2)


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

  def set_init_latent(game: BoxPushSimulator_AlwaysTogether):
    valid_boxes = event_impl.get_valid_box_to_pickup(game)
    if len(valid_boxes) > 0:
      box_idx = random.choice(valid_boxes)
      game.event_input(AGENT1, EventType.SET_LATENT, ("pickup", box_idx))
      game.event_input(AGENT2, EventType.SET_LATENT, ("pickup", box_idx))

  event_impl.run_task_game(msg, g_id_2_game, agent2, set_init_latent, EXP1_MAP,
                           event_impl.NOT_ASK_LATENT, EXP1_NAMESPACE, TASK_TYPE)


@socketio.on('action_event', namespace=EXP1_NAMESPACE)
def action_event(msg):
  def hold_changed(game, a1_hold_changed, a2_hold_changed, a1_box, a2_box):
    if a1_hold_changed:
      if a1_box >= 0:
        game.event_input(AGENT1, EventType.SET_LATENT, ("goal", 0))
        game.event_input(AGENT2, EventType.SET_LATENT, ("goal", 0))
      else:
        valid_boxes = event_impl.get_valid_box_to_pickup(game)
        if len(valid_boxes) > 0:
          box_idx = random.choice(valid_boxes)
          game.event_input(AGENT1, EventType.SET_LATENT, ("pickup", box_idx))
          game.event_input(AGENT2, EventType.SET_LATENT, ("pickup", box_idx))

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
