from flask import session
from flask_socketio import emit
from web_experiment import socketio
from web_experiment.review.util import update_canvas
from web_experiment.feedback.define import (COLLECT_CANVAS_PAGELIST)


def record_latent(msg):
  lstate = msg['latent']
  session['latent_human_recorded'][session['index']] = lstate
  print(session['latent_human_recorded'])


for socket_type in COLLECT_CANVAS_PAGELIST:
  name_space = '/' + socket_type

  def make_init_canvas(domain_type):
    def init_canvas():
      update_canvas(COLLECT_CANVAS_PAGELIST[domain_type][0], init_imgs=True)
      latent = session['latent_human_recorded'][session['index']]
      emit("cur_latent", {"latent": latent})

    return init_canvas

  def make_next_index(domain_type):
    def next_index(msg):
      if session['index'] < session['max_index']:
        session['index'] += 1
        update_canvas(COLLECT_CANVAS_PAGELIST[domain_type][0], init_imgs=False)
        latent = session['latent_human_recorded'][session['index']]
        emit("cur_latent", {"latent": latent})
        if session['index'] == session['max_index']:
          emit('complete')

    return next_index

  def make_prev_index(domain_type):
    def prev_index():
      if session['index'] > 0:
        session['index'] -= 1
        update_canvas(COLLECT_CANVAS_PAGELIST[domain_type][0], init_imgs=False)
        latent = session['latent_human_recorded'][session['index']]
        emit("cur_latent", {"latent": latent})

    return prev_index

  def make_goto_index(domain_type):
    def goto_index(msg):
      idx = int(msg['index'])
      if (idx <= (session['max_index']) and idx >= 0):
        session['index'] = idx
        update_canvas(COLLECT_CANVAS_PAGELIST[domain_type][0], init_imgs=False)
        latent = session['latent_human_recorded'][session['index']]
        emit("cur_latent", {"latent": latent})

    return goto_index

  def make_record_latent():
    def record_latent_wrapper(msg):
      record_latent(msg)

    return record_latent_wrapper

  socketio.on_event('connect', make_init_canvas(socket_type), name_space)
  socketio.on_event('next', make_next_index(socket_type), name_space)
  socketio.on_event('prev', make_prev_index(socket_type), name_space)
  socketio.on_event('index', make_goto_index(socket_type), name_space)
  socketio.on_event('record_latent', make_record_latent(), name_space)
