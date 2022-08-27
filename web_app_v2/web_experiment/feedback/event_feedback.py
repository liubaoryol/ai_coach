from flask import session
from web_experiment import socketio
from web_experiment.auth.util import update_canvas, update_latent_state
from web_experiment.feedback.define import (FEEDBACK_NAMESPACES,
                                            FEEDBACK_CANVAS_PAGELIST)
from web_experiment.define import EMode, GroupName


def update_canvas_helper(domain_type, groupid, init_imgs=False):
  update_canvas(FEEDBACK_CANVAS_PAGELIST[domain_type][0], init_imgs)
  if groupid == GroupName.Group_C:
    update_latent_state(domain_type, EMode.Collected)
  elif groupid == GroupName.Group_D:
    update_latent_state(domain_type, EMode.Predicted)


for domain_type in FEEDBACK_NAMESPACES:
  name_space = '/' + FEEDBACK_NAMESPACES[domain_type]

  def make_init_canvas(domain_type):
    def init_canvas():
      update_canvas_helper(domain_type, session["groupid"], True)

    return init_canvas

  def make_next_index(domain_type):
    def next_index():
      if session['index'] < (session['max_index'] - 1):
        session['index'] += 1
      update_canvas_helper(domain_type, session["groupid"], False)

    return next_index

  def make_prev_index(domain_type):
    def prev_index():
      if session['index'] > 0:
        session['index'] -= 1
      update_canvas_helper(domain_type, session["groupid"], False)

    return prev_index

  def make_goto_index(domain_type):
    def goto_index(msg):
      idx = int(msg['index'])
      print(idx)
      if (idx <= (session['max_index']) and idx >= 0):
        session['index'] = idx
      update_canvas_helper(domain_type, session["groupid"], False)

    return goto_index

  socketio.on_event('connect', make_init_canvas(domain_type), name_space)
  socketio.on_event('next', make_next_index(domain_type), name_space)
  socketio.on_event('prev', make_prev_index(domain_type), name_space)
  socketio.on_event('index', make_goto_index(domain_type), name_space)
