from web_experiment import socketio
from flask import (request, session)
import web_experiment.experiment1.events_impl as event_impl
from web_experiment.auth.util import update_canvas

NAMESPACE = '/alone_record'
TASK_TYPE = event_impl.TASK_B


# TOGETHER_NAMESPACE
@socketio.on('connect', namespace=NAMESPACE)
def initial_canvas():
  event_impl.initial_canvas(session['session_name'], TASK_TYPE)
  update_canvas(request.sid, NAMESPACE, False, is_movers_domain=TASK_TYPE)


@socketio.on('next', namespace=NAMESPACE)
def next_index():
  if session['index'] < (session['max_index'] - 1):
    session['index'] += 1
    update_canvas(request.sid, NAMESPACE, False, is_movers_domain=TASK_TYPE)


@socketio.on('prev', namespace=NAMESPACE)
def prev_index():
  if session['index'] > 0:
    session['index'] -= 1
    update_canvas(request.sid, NAMESPACE, False, is_movers_domain=TASK_TYPE)


@socketio.on('index', namespace=NAMESPACE)
def goto_index(msg):
  idx = int(msg['index'])
  if (idx <= (session['max_index'] - 1) and idx >= 0):
    session['index'] = idx
    update_canvas(request.sid, NAMESPACE, False, is_movers_domain=TASK_TYPE)


@socketio.on('record_latent', namespace=NAMESPACE)
def record_namespace(msg):
  lstate = msg['latent']
  session['latent_human_recorded'][session['index']] = lstate
  print(session['latent_human_recorded'])
