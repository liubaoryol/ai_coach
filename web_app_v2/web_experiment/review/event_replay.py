from flask import session
from web_experiment import socketio
from web_experiment.define import EMode, PageKey
from web_experiment.review.define import REPLAY_CANVAS_PAGELIST, get_socket_name
from web_experiment.review.util import update_canvas, update_latent_state

for domain_type in REPLAY_CANVAS_PAGELIST:
  name_space = '/' + get_socket_name(PageKey.Replay, domain_type)

  def make_init_canvas(domain_type):
    def init_canvas():
      update_canvas(REPLAY_CANVAS_PAGELIST[domain_type][0], init_imgs=True)
      update_latent_state(domain_type, mode=EMode.Replay)

    return init_canvas

  def make_next_index(domain_type):
    def next_index():
      if session['index'] < (session['max_index']):
        session['index'] += 1
        update_canvas(REPLAY_CANVAS_PAGELIST[domain_type][0], False)
        update_latent_state(domain_type, mode=EMode.Replay)

    return next_index

  def make_prev_index(domain_type):
    def prev_index():
      if session['index'] > 0:
        session['index'] -= 1
        update_canvas(REPLAY_CANVAS_PAGELIST[domain_type][0], False)
        update_latent_state(domain_type, mode=EMode.Replay)

    return prev_index

  def make_goto_index(domain_type):
    def goto_index(msg):
      idx = int(msg['index'])
      if (idx <= (session['max_index']) and idx >= 0):
        session['index'] = idx
        update_canvas(REPLAY_CANVAS_PAGELIST[domain_type][0], False)
        update_latent_state(domain_type, mode=EMode.Replay)

    return goto_index

  socketio.on_event('connect', make_init_canvas(domain_type), name_space)
  socketio.on_event('next', make_next_index(domain_type), name_space)
  socketio.on_event('prev', make_prev_index(domain_type), name_space)
  socketio.on_event('index', make_goto_index(domain_type), name_space)
