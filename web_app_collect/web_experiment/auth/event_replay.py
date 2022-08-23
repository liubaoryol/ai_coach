from flask import session
from web_experiment.auth.util import update_canvas, update_latent_state
from web_experiment.define import EMode
import web_experiment.auth.define as dfn
from web_experiment import socketio

for session_name in dfn.REPLAY_NAMESPACES:
  name_space = '/' + dfn.REPLAY_NAMESPACES[session_name]

  def make_init_canvas(session_name):
    def init_canvas():
      update_canvas(dfn.REPLAY_CANVAS_PAGELIST[session_name][0], init_imgs=True)
      update_latent_state(domain_type=dfn.REPALY_SESSION_TYPE[session_name],
                          mode=EMode.Replay)

    return init_canvas

  def make_next_index(session_name):
    def next_index():
      if session['index'] < (session['max_index']):
        session['index'] += 1
        update_canvas(dfn.REPLAY_CANVAS_PAGELIST[session_name][0], False)
        update_latent_state(domain_type=dfn.REPALY_SESSION_TYPE[session_name],
                            mode=EMode.Replay)

    return next_index

  def make_prev_index(session_name):
    def prev_index():
      if session['index'] > 0:
        session['index'] -= 1
        update_canvas(dfn.REPLAY_CANVAS_PAGELIST[session_name][0], False)
        update_latent_state(domain_type=dfn.REPALY_SESSION_TYPE[session_name],
                            mode=EMode.Replay)

    return prev_index

  def make_goto_index(session_name):
    def goto_index(msg):
      idx = int(msg['index'])
      if (idx <= (session['max_index']) and idx >= 0):
        session['index'] = idx
        update_canvas(dfn.REPLAY_CANVAS_PAGELIST[session_name][0], False)
        update_latent_state(domain_type=dfn.REPALY_SESSION_TYPE[session_name],
                            mode=EMode.Replay)

    return goto_index

  socketio.on_event('connect', make_init_canvas(session_name), name_space)
  socketio.on_event('next', make_next_index(session_name), name_space)
  socketio.on_event('prev', make_prev_index(session_name), name_space)
  socketio.on_event('index', make_goto_index(session_name), name_space)
