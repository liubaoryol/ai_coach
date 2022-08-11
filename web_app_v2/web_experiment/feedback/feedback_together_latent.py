from web_experiment import socketio
from flask import (request, session)
import web_experiment.experiment1.events_impl as event_impl
from web_experiment.auth.util import update_canvas

TOGETHER_NAMESPACE = '/feedback_together_latent'
TASK_TYPE = event_impl.TASK_A


# TOGETHER_NAMESPACE
@socketio.on('connect', namespace=TOGETHER_NAMESPACE)
def initial_canvas():
  event_impl.initial_canvas(session['session_name'], TASK_TYPE)
  update_canvas_helper(session['groupid'])


@socketio.on('next', namespace=TOGETHER_NAMESPACE)
def next_index():

  if session['index'] < (session['max_index']):
    session['index'] += 1
    update_canvas_helper(session['groupid'])


@socketio.on('prev', namespace=TOGETHER_NAMESPACE)
def prev_index():
  if session['index'] > 0:
    session['index'] -= 1
    update_canvas_helper(session['groupid'])


@socketio.on('index', namespace=TOGETHER_NAMESPACE)
def goto_index(msg):
  idx = int(msg['index'])
  if (idx <= (session['max_index']) and idx >= 0):
    session['index'] = idx
    update_canvas_helper(session['groupid'])


def update_canvas_helper(groupid):
  if groupid == 'C':
    update_canvas(request.sid, TOGETHER_NAMESPACE, True, "collected", TASK_TYPE)
  elif groupid == 'D':
    update_canvas(request.sid, TOGETHER_NAMESPACE, True, "predicted", TASK_TYPE)
