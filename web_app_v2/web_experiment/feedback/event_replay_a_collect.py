from web_experiment import socketio
from ai_coach_domain.box_push.maps import EXP1_MAP
from flask import (request, session)
import web_experiment.experiment1.events_impl as event_impl
from web_experiment.auth.util import update_canvas

GRID_X = EXP1_MAP["x_grid"]
GRID_Y = EXP1_MAP["y_grid"]
NAMESPACE = '/together_collect'

# TOGETHER_NAMESPACE
@socketio.on('connect', namespace=NAMESPACE)
def initial_canvas():
    event_impl.initial_canvas(GRID_X, GRID_Y)
    update_canvas(request.sid, NAMESPACE, False)

@socketio.on('next', namespace=NAMESPACE)
def next_index():
    if session['index'] < (session['max_index'] - 1):
        session['index'] += 1
        update_canvas(request.sid, NAMESPACE, False)

@socketio.on('prev', namespace=NAMESPACE)
def prev_index():
    if session['index'] > 0:
        session['index'] -= 1
        update_canvas(request.sid, NAMESPACE, False)

@socketio.on('index', namespace=NAMESPACE)
def next_index(msg):
    idx = int(msg['index'])
    if (idx <= (session['max_index'] - 1) and idx >= 0):
        session['index'] = idx
        update_canvas(request.sid, NAMESPACE, False)

@socketio.on('record_latent', namespace = NAMESPACE)
def record_namespace(msg):
    lstate = msg['latent']
    session['latent_human_recorded'][session['index']] = lstate
    print(session['latent_human_recorded'])
    