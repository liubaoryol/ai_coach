from web_experiment import socketio
from ai_coach_domain.box_push.maps import EXP1_MAP
from flask import (request, session)
import web_experiment.experiment1.events_impl as event_impl
from web_experiment.auth.util import update_canvas

GRID_X = EXP1_MAP["x_grid"]
GRID_Y = EXP1_MAP["y_grid"]
TOGETHER_NAMESPACE = '/feedback_together_true_latent'

# TOGETHER_NAMESPACE
@socketio.on('connect', namespace=TOGETHER_NAMESPACE)
def initial_canvas():
    event_impl.initial_canvas(GRID_X, GRID_Y)
    update_canvas_helper(session['groupid'])   




@socketio.on('next', namespace=TOGETHER_NAMESPACE)
def next_index():
    if session['index'] < (session['max_index'] - 1):
        session['index'] += 1
        update_canvas_helper(session['groupid'])  

@socketio.on('prev', namespace=TOGETHER_NAMESPACE)
def prev_index():
    if session['index'] > 0:
        session['index'] -= 1
        update_canvas_helper(session['groupid'])

@socketio.on('index', namespace=TOGETHER_NAMESPACE)
def next_index(msg):
    idx = int(msg['index'])
    print(session)
    if (idx <= (session['max_index'] - 1) and idx >= 0):
        session['index'] = idx
        update_canvas_helper(session['groupid'])
        

def update_canvas_helper(groupid):
    if groupid == 'C':
        update_canvas(request.sid, TOGETHER_NAMESPACE, True, latent_from = "After Game",
        make_prediction=False)    
    elif groupid == 'B':
        update_canvas(request.sid, TOGETHER_NAMESPACE, True, latent_from = "None",
        make_prediction=False)  
    