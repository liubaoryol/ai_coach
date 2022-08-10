from web_experiment import socketio
from ai_coach_domain.box_push.maps import EXP1_MAP
from flask import (request, session, flash)
import web_experiment.experiment1.events_impl as event_impl
from web_experiment.auth.util import update_canvas
import json
from web_experiment.feedback.helper import store_latent_locally

NAMESPACE = '/together_collect'
TASK_TYPE = event_impl.TASK_A


# TOGETHER_NAMESPACE
@socketio.on('connect', namespace=NAMESPACE)
def initial_canvas():
  event_impl.initial_canvas(session['session_name'], TASK_TYPE)
  update_canvas(request.sid, NAMESPACE, False, is_movers_domain=TASK_TYPE)


@socketio.on('next', namespace=NAMESPACE)
def next_index(msg):
  if session['index'] < (session['max_index'] - 1):
    record_latent(msg)
    session['index'] += 1
    update_canvas(request.sid, NAMESPACE, False, is_movers_domain=TASK_TYPE)
  # find a better way to only store once
  if session['index'] == (session['max_index'] - 1):
    objs = {}
    objs_json = json.dumps(objs)
    print(session['latent_human_recorded'])
    store_latent_locally(session['user_id'], session['session_name'],
                         'BoxPushSimulator_AlwaysTogether', EXP1_MAP,
                         session['latent_human_recorded'])

    socketio.emit('complete', objs_json, room=request.sid, namespace=NAMESPACE)


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
  record_latent(msg)


def record_latent(msg):
  lstate = msg['latent']
  session['latent_human_recorded'][session['index']] = lstate
  print(session['latent_human_recorded'])
