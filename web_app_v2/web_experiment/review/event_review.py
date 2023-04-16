from typing import Any, Mapping
from flask import session, request
from flask_socketio import emit
from web_experiment import socketio
from web_experiment.models import ExpDataCollection
from web_experiment.define import (PageKey, get_domain_type,
                                   get_record_session_key)
from web_experiment.exp_common.page_replay import UserDataReplay
from web_experiment.review.define import REVIEW_CANVAS_PAGELIST, get_socket_name
from web_experiment.review.util import (update_canvas, canvas_button_clicked,
                                        load_trajectory, latent_state_from_traj,
                                        no_trajectory_page, get_init_user_data)
from web_experiment.feedback.helper import (store_latent_locally,
                                            store_user_fix_locally)

g_id_2_session_data = {}  # type: Mapping[Any, UserDataReplay]

for domain_type in REVIEW_CANVAS_PAGELIST:
  name_space = '/' + get_socket_name(PageKey.Review, domain_type)

  def make_init_canvas(domain_type, name_space=name_space):

    def init_canvas():
      global g_id_2_session_data
      sid = request.sid
      cur_user = session.get('user_id')
      session_name = session.get('loaded_session_name')

      trajectory = load_trajectory(session_name, cur_user)
      if trajectory is None:
        no_trajectory_page(sid, name_space, "ERROR: No trajectory")
        emit('complete')
        return

      record_session_name = get_record_session_key(session_name)
      query = ExpDataCollection.query.filter_by(subject_id=cur_user).first()
      if getattr(query, record_session_name):
        no_trajectory_page(sid, name_space, "You cannot fix this task anymore.")
        emit('complete')
        return

      user_data = get_init_user_data(cur_user, session_name, trajectory)

      g_id_2_session_data[sid] = user_data

      max_idx = len(trajectory) - 1

      update_canvas(sid,
                    name_space,
                    REVIEW_CANVAS_PAGELIST[domain_type][0],
                    g_id_2_session_data[sid],
                    init_imgs=True,
                    domain_type=domain_type)
      emit("set_max", {"max_index": max_idx})

    return init_canvas

  def make_goto_index(domain_type, name_space=name_space):

    def goto_index(msg):
      global g_id_2_session_data
      sid = request.sid
      if sid not in g_id_2_session_data:
        return

      user_data = g_id_2_session_data[sid]

      idx = int(msg['index'])
      max_index = len(user_data.data[UserDataReplay.TRAJECTORY]) - 1
      if (idx <= max_index and idx >= 0):
        user_data.data[UserDataReplay.TRAJ_IDX] = idx
        update_canvas(sid,
                      name_space,
                      REVIEW_CANVAS_PAGELIST[domain_type][0],
                      user_data,
                      init_imgs=False)
        if user_data.data[UserDataReplay.TRAJ_IDX] == max_index:
          emit('complete')

    return goto_index

  def make_button_clicked(domain_type, name_space=name_space):

    def button_clicked(msg):
      global g_id_2_session_data
      sid = request.sid
      if sid not in g_id_2_session_data:
        return

      user_data = g_id_2_session_data[sid]

      button = msg["name"]
      page = REVIEW_CANVAS_PAGELIST[domain_type][0]
      canvas_button_clicked(sid, name_space, button, page, user_data)

    return button_clicked

  def make_disconnected():

    def disconnected():
      global g_id_2_session_data
      sid = request.sid

      if sid in g_id_2_session_data:
        user_data = g_id_2_session_data[sid]
        latents = latent_state_from_traj(
            user_data.data[UserDataReplay.TRAJECTORY],
            get_domain_type(user_data.data[UserDataReplay.SESSION_NAME]))

        # save latent
        store_latent_locally(user_data.data[UserDataReplay.USER],
                             user_data.data[UserDataReplay.SESSION_NAME],
                             latents)
        store_user_fix_locally(user_data.data[UserDataReplay.USER],
                               user_data.data[UserDataReplay.SESSION_NAME],
                               user_data.data[UserDataReplay.USER_FIX])
        # delete session_data
        del g_id_2_session_data[sid]

    return disconnected

  socketio.on_event('connect', make_init_canvas(domain_type), name_space)
  socketio.on_event('disconnect', make_disconnected(), name_space)
  socketio.on_event('index', make_goto_index(domain_type), name_space)
  socketio.on_event('button_clicked', make_button_clicked(domain_type),
                    name_space)
