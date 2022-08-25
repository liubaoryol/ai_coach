from web_experiment import socketio
from flask import session
from web_experiment.auth.util import update_canvas
from web_experiment.replay.define import (RECORD_NAMESPACES,
                                          RECORD_CANVAS_PAGELIST)

for domain_type in RECORD_NAMESPACES:
  name_space = '/' + RECORD_NAMESPACES[domain_type]

  def make_init_canvas(domain_type):
    def init_canvas():
      update_canvas(RECORD_CANVAS_PAGELIST[domain_type][0], init_imgs=True)

    return init_canvas

  def make_next_index(domain_type):
    def next_index(msg):
      if session['index'] < (session['max_index']):
        session['index'] += 1
        update_canvas(RECORD_CANVAS_PAGELIST[domain_type][0], init_imgs=False)

    return next_index

  def make_prev_index(domain_type):
    def prev_index():
      if session['index'] > 0:
        session['index'] -= 1
        update_canvas(RECORD_CANVAS_PAGELIST[domain_type][0], init_imgs=False)

    return prev_index

  def make_goto_index(domain_type):
    def goto_index(msg):
      idx = int(msg['index'])
      if (idx <= (session['max_index']) and idx >= 0):
        session['index'] = idx
        update_canvas(RECORD_CANVAS_PAGELIST[domain_type][0], init_imgs=False)

    return goto_index

  def make_record_latent():
    def record_latent_wrapper(msg):
      lstate = msg['latent']
      session['latent_human_recorded'][session['index']] = lstate
      print(session['latent_human_recorded'])

    return record_latent_wrapper

  socketio.on_event('connect', make_init_canvas(domain_type), name_space)
  socketio.on_event('next', make_next_index(domain_type), name_space)
  socketio.on_event('prev', make_prev_index(domain_type), name_space)
  socketio.on_event('index', make_goto_index(domain_type), name_space)
  socketio.on_event('record_latent', make_record_latent(), name_space)
