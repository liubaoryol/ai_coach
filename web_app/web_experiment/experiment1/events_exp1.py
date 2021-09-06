from typing import Mapping, Hashable
from ai_coach_domain.box_push import BoxPushSimulator
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
  event_impl.run_game(msg, EXP1_NAMESPACE, g_id_2_game, GRID_X, GRID_Y, {})


@socketio.on('action_event', namespace=EXP1_NAMESPACE)
def on_key_down(msg):
  event_impl.on_key_down(msg, EXP1_NAMESPACE, g_id_2_game)


@socketio.on('set_latent', namespace=EXP1_NAMESPACE)
def set_latent(msg):
  event_impl.set_latent(msg, EXP1_NAMESPACE, g_id_2_game)
