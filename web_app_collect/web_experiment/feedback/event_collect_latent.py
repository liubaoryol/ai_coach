import json
from flask import session
from flask_socketio import emit
from ai_coach_domain.box_push.maps import EXP1_MAP
from web_experiment import socketio
from web_experiment.auth.util import update_canvas
from web_experiment.feedback.helper import store_latent_locally
from web_experiment.feedback.define import (COLLECT_NAMESPACES,
                                            COLLECT_CANVAS_PAGELIST,
                                            COLLECT_SESSION_GAME_TYPE)


def record_latent(msg):
  lstate = msg['latent']
  session['latent_human_recorded'][session['index']] = lstate
  print(session['latent_human_recorded'])


for domain_type in COLLECT_NAMESPACES:
  name_space = '/' + COLLECT_NAMESPACES[domain_type]

  def make_init_canvas(domain_type):
    def init_canvas():
      update_canvas(COLLECT_CANVAS_PAGELIST[domain_type][0], init_imgs=True)

    return init_canvas

  def make_next_index(domain_type):
    def next_index(msg):
      if session['index'] < (session['max_index']):
        record_latent(msg)
        session['index'] += 1
        update_canvas(COLLECT_CANVAS_PAGELIST[domain_type][0], init_imgs=False)
      # find a better way to only store once
      elif session['index'] == (session['max_index']):
        update_canvas(COLLECT_CANVAS_PAGELIST[domain_type][0], init_imgs=False)
        objs = {}
        objs_json = json.dumps(objs)
        print(session['latent_human_recorded'])
        store_latent_locally(session['user_id'], session['loaded_session_name'],
                             COLLECT_SESSION_GAME_TYPE[domain_type], EXP1_MAP,
                             session['latent_human_recorded'])

        emit('complete', objs_json)

    return next_index

  def make_prev_index(domain_type):
    def prev_index():
      if session['index'] > 0:
        session['index'] -= 1
        update_canvas(COLLECT_CANVAS_PAGELIST[domain_type][0], init_imgs=False)

    return prev_index

  def make_goto_index(domain_type):
    def goto_index(msg):
      idx = int(msg['index'])
      print(idx)
      if (idx <= (session['max_index']) and idx >= 0):
        session['index'] = idx
        update_canvas(COLLECT_CANVAS_PAGELIST[domain_type][0], init_imgs=False)

    return goto_index

  def make_record_latent():
    def record_latent_wrapper(msg):
      record_latent(msg)

    return record_latent_wrapper

  socketio.on_event('connect', make_init_canvas(domain_type), name_space)
  socketio.on_event('next', make_next_index(domain_type), name_space)
  socketio.on_event('prev', make_prev_index(domain_type), name_space)
  socketio.on_event('index', make_goto_index(domain_type), name_space)
  socketio.on_event('record_latent', make_record_latent(), name_space)
