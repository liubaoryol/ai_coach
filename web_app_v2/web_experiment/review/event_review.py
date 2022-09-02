from flask import session
from flask_socketio import emit
from web_experiment import socketio
from web_experiment.define import PageKey
from web_experiment.review.define import REVIEW_CANVAS_PAGELIST, get_socket_name
from web_experiment.review.util import update_canvas, canvas_button_clicked

for domain_type in REVIEW_CANVAS_PAGELIST:
  name_space = '/' + get_socket_name(PageKey.Review, domain_type)

  def make_init_canvas(domain_type):
    def init_canvas():
      update_canvas(REVIEW_CANVAS_PAGELIST[domain_type][0], init_imgs=True)

    return init_canvas

  def make_goto_index(domain_type):
    def goto_index(msg):
      idx = int(msg['index'])
      if (idx <= (session['max_index']) and idx >= 0):
        session['index'] = idx
        update_canvas(REVIEW_CANVAS_PAGELIST[domain_type][0], init_imgs=False)
        if session['index'] == session['max_index']:
          emit('complete')

    return goto_index

  def make_button_clicked(domain_type):
    def button_clicked(msg):
      button = msg["name"]
      page = REVIEW_CANVAS_PAGELIST[domain_type][0]
      canvas_button_clicked(button, page)

      traj_idx = session['index']
      dict_game = session["dict"][traj_idx]
      latent = dict_game["a1_latent"]
      session['latent_human_recorded'][session['index']] = (
          f"{latent[0]}, {latent[1]}")

    return button_clicked

  socketio.on_event('connect', make_init_canvas(domain_type), name_space)
  socketio.on_event('index', make_goto_index(domain_type), name_space)
  socketio.on_event('button_clicked', make_button_clicked(domain_type),
                    name_space)
