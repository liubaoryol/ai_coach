from web_experiment import socketio
from ai_coach_domain.box_push.maps import EXP1_MAP
from flask import (request, session)
import web_experiment.experiment1.events_impl as event_impl
from web_experiment.auth.util import update_canvas

GRID_X = EXP1_MAP["x_grid"]
GRID_Y = EXP1_MAP["y_grid"]
ALONE_NAMESPACE = '/alone'

# TOGETHER_NAMESPACE
@socketio.on('connect', namespace=ALONE_NAMESPACE)
def initial_canvas():
    event_impl.initial_canvas(GRID_X, GRID_Y)
    update_canvas(request.sid, ALONE_NAMESPACE, True, is_movers_domain=False)

@socketio.on('next', namespace=ALONE_NAMESPACE)
def next_index():
    if session['index'] < (session['max_index'] - 1):
        session['index'] += 1
        update_canvas(request.sid, ALONE_NAMESPACE, True, is_movers_domain=False)

@socketio.on('prev', namespace=ALONE_NAMESPACE)
def prev_index():
    if session['index'] > 0:
        session['index'] -= 1
        update_canvas(request.sid, ALONE_NAMESPACE, True, is_movers_domain=False)

@socketio.on('index', namespace=ALONE_NAMESPACE)
def next_index(msg):
    idx = int(msg['index'])
    if (idx <= (session['max_index'] - 1) and idx >= 0):
        session['index'] = idx
        update_canvas(request.sid, ALONE_NAMESPACE, True, is_movers_domain=False)