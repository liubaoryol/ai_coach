from typing import Any, Mapping
from flask import session, request
from flask_socketio import emit
from web_experiment import socketio
from web_experiment.define import PageKey, get_domain_type
from web_experiment.review.define import REVIEW_CANVAS_PAGELIST, get_socket_name
from web_experiment.review.util import (update_canvas, canvas_button_clicked,
                                        SessionData, load_trajectory,
                                        latent_state_from_traj)
from web_experiment.feedback.helper import store_latent_locally

g_id_2_session_data = {}  # type: Mapping[Any, SessionData]

for domain_type in REVIEW_CANVAS_PAGELIST:
  name_space = '/' + get_socket_name(PageKey.Review, domain_type)

  def make_init_canvas(domain_type):
    def init_canvas():
      global g_id_2_session_data
      sid = request.sid
      cur_user = session.get('user_id')
      session_name = session.get('loaded_session_name')

      trajectory = load_trajectory(session_name, cur_user)
      g_id_2_session_data[sid] = SessionData(cur_user, session_name, trajectory,
                                             0)
      max_idx = len(trajectory) - 1

      update_canvas(REVIEW_CANVAS_PAGELIST[domain_type][0],
                    g_id_2_session_data[sid],
                    init_imgs=True,
                    domain_type=domain_type)
      emit("set_max", {"max_index": max_idx})

    return init_canvas

  def make_goto_index(domain_type):
    def goto_index(msg):
      global g_id_2_session_data
      sid = request.sid
      session_data = g_id_2_session_data[sid]

      idx = int(msg['index'])
      max_index = len(session_data.trajectory) - 1
      if (idx <= max_index and idx >= 0):
        session_data.index = idx
        update_canvas(REVIEW_CANVAS_PAGELIST[domain_type][0],
                      session_data,
                      init_imgs=False)
        if session_data.index == max_index:
          emit('complete')

    return goto_index

  def make_button_clicked(domain_type):
    def button_clicked(msg):
      global g_id_2_session_data
      sid = request.sid
      session_data = g_id_2_session_data[sid]

      button = msg["name"]
      page = REVIEW_CANVAS_PAGELIST[domain_type][0]
      canvas_button_clicked(button, page, session_data)

    return button_clicked

  def make_disconnected():
    def disconnected():
      global g_id_2_session_data
      sid = request.sid

      if sid in g_id_2_session_data:
        session_data = g_id_2_session_data[sid]
        latents = latent_state_from_traj(
            session_data.trajectory, get_domain_type(session_data.session_name))

        # save latent
        store_latent_locally(session_data.user_id, session_data.session_name,
                             latents)
        # delete session_data
        del g_id_2_session_data[sid]

    return disconnected

  socketio.on_event('connect', make_init_canvas(domain_type), name_space)
  socketio.on_event('disconnect', make_disconnected(), name_space)
  socketio.on_event('index', make_goto_index(domain_type), name_space)
  socketio.on_event('button_clicked', make_button_clicked(domain_type),
                    name_space)
