from typing import Mapping, Hashable
from ai_coach_domain.box_push.simulator import BoxPushSimulator_AlwaysAlone
from ai_coach_domain.box_push.maps import EXP1_MAP
from ai_coach_domain.box_push.mdp import BoxPushAgentMDP_AlwaysAlone
from ai_coach_domain.box_push.mdppolicy import BoxPushPolicyIndvExp1
from ai_coach_domain.box_push.agent import BoxPushAIAgent_Indv2
from web_experiment import socketio
import web_experiment.experiment1.events_impl as event_impl

g_id_2_game = {}  # type: Mapping[Hashable, BoxPushSimulator_AlwaysAlone]
EXP1_NAMESPACE = '/exp1_indv_user_random_2'
GRID_X = EXP1_MAP["x_grid"]
GRID_Y = EXP1_MAP["y_grid"]
EXP1_MDP = BoxPushAgentMDP_AlwaysAlone(**EXP1_MAP)

AGENT1 = BoxPushSimulator_AlwaysAlone.AGENT1
AGENT2 = BoxPushSimulator_AlwaysAlone.AGENT2

TEMPERATURE = 0.3
TEAMMATE_POLICY = BoxPushPolicyIndvExp1(EXP1_MDP, TEMPERATURE)


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
  agent2 = BoxPushAIAgent_Indv2(TEAMMATE_POLICY)

  event_impl.run_task_game(msg, g_id_2_game, agent2, None, EXP1_MDP, EXP1_MAP,
                           event_impl.ASK_LATENT, EXP1_NAMESPACE, False)


@socketio.on('action_event', namespace=EXP1_NAMESPACE)
def action_event(msg):
  def game_finished(game, env_id, name_space):
    session_name = "session_b4"
    cur_user = msg["user_id"]
    event_impl.task_end(env_id, game, cur_user, session_name,
                        "BoxPushSimulator_AlwaysAlone", EXP1_MAP, name_space,
                        False)

  ASK_LATENT_FREQUENCY = 5
  event_impl.action_event(msg, g_id_2_game, None, game_finished, EXP1_NAMESPACE,
                          True, True, ASK_LATENT_FREQUENCY)


@socketio.on('set_latent', namespace=EXP1_NAMESPACE)
def set_latent(msg):
  event_impl.set_latent(msg, g_id_2_game, EXP1_NAMESPACE)
