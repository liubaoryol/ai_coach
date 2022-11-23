from typing import Mapping, Any
from flask import session, request
from flask_socketio import emit
from web_experiment import socketio
from web_experiment.exp_common.page_replay import UserDataReplay
from web_experiment.review.util import (update_canvas, load_trajectory,
                                        get_init_user_data)
from web_experiment.feedback.define import (COLLECT_CANVAS_PAGELIST,
                                            COLLECT_SOCKET_DOMAIN)
from web_experiment.feedback.helper import store_latent_locally

g_id_2_session_data = {}  # type: Mapping[Any, UserDataReplay]


def record_latent(msg, user_data: UserDataReplay):
  lstate = msg['latent']
  user_data.data[UserDataReplay.LATENT_COLLECTED][user_data.data[
      UserDataReplay.TRAJ_IDX]] = lstate
  print(user_data.data[UserDataReplay.LATENT_COLLECTED])


for socket_type in COLLECT_CANVAS_PAGELIST:
  name_space = '/' + socket_type

  def make_init_canvas(socket_type, name_space=name_space):

    def init_canvas():
      global g_id_2_session_data
      sid = request.sid
      cur_user = session.get('user_id')
      session_name = session.get('loaded_session_name')

      trajectory = load_trajectory(session_name, cur_user)
      if trajectory is None:
        emit('complete')
        return

      user_data = get_init_user_data(cur_user, session_name, trajectory)
      g_id_2_session_data[sid] = user_data
      max_idx = len(trajectory) - 1
      user_data.data[UserDataReplay.LATENT_COLLECTED] = (["None"] *
                                                         len(trajectory))
      update_canvas(sid,
                    name_space,
                    COLLECT_CANVAS_PAGELIST[socket_type][0],
                    user_data,
                    init_imgs=True,
                    domain_type=COLLECT_SOCKET_DOMAIN[socket_type])
      latent = user_data.data[UserDataReplay.LATENT_COLLECTED][user_data.data[
          UserDataReplay.TRAJ_IDX]]
      emit("cur_latent", {"latent": latent})
      emit("set_max", {"max_index": max_idx})

    return init_canvas

  def make_next_index(socket_type, name_space=name_space):

    def next_index(msg):
      global g_id_2_session_data
      sid = request.sid
      if sid not in g_id_2_session_data:
        return

      user_data = g_id_2_session_data[sid]

      max_index = len(user_data.data[UserDataReplay.TRAJECTORY]) - 1
      if user_data.data[UserDataReplay.TRAJ_IDX] < max_index:
        user_data.data[UserDataReplay.TRAJ_IDX] += 1
        update_canvas(sid,
                      name_space,
                      COLLECT_CANVAS_PAGELIST[socket_type][0],
                      user_data,
                      init_imgs=False)
        latent = user_data.data[UserDataReplay.LATENT_COLLECTED][user_data.data[
            UserDataReplay.TRAJ_IDX]]
        emit("cur_latent", {"latent": latent})
        if user_data.data[UserDataReplay.TRAJ_IDX] == max_index:
          emit('complete')

    return next_index

  def make_prev_index(socket_type, name_space=name_space):

    def prev_index():
      global g_id_2_session_data
      sid = request.sid
      if sid not in g_id_2_session_data:
        return
      user_data = g_id_2_session_data[sid]

      if user_data.data[UserDataReplay.TRAJ_IDX] > 0:
        user_data.data[UserDataReplay.TRAJ_IDX] -= 1
        update_canvas(sid,
                      name_space,
                      COLLECT_CANVAS_PAGELIST[socket_type][0],
                      user_data,
                      init_imgs=False)
        latent = user_data.data[UserDataReplay.LATENT_COLLECTED][user_data.data[
            UserDataReplay.TRAJ_IDX]]
        emit("cur_latent", {"latent": latent})

    return prev_index

  def make_goto_index(socket_type, name_space=name_space):

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
                      COLLECT_CANVAS_PAGELIST[socket_type][0],
                      user_data,
                      init_imgs=False)
        latent = user_data.data[UserDataReplay.LATENT_COLLECTED][user_data.data[
            UserDataReplay.TRAJ_IDX]]
        emit("cur_latent", {"latent": latent})

    return goto_index

  def make_record_latent():

    def record_latent_wrapper(msg):
      global g_id_2_session_data
      sid = request.sid
      if sid not in g_id_2_session_data:
        return
      user_data = g_id_2_session_data[sid]
      record_latent(msg, user_data)

    return record_latent_wrapper

  def make_disconnected():

    def disconnected():
      global g_id_2_session_data
      sid = request.sid

      if sid in g_id_2_session_data:
        user_data = g_id_2_session_data[sid]

        # save latent
        store_latent_locally(user_data.data[UserDataReplay.USER],
                             user_data.data[UserDataReplay.SESSION_NAME],
                             user_data.data[UserDataReplay.LATENT_COLLECTED])
        # delete session_data
        del g_id_2_session_data[sid]

    return disconnected

  socketio.on_event('connect', make_init_canvas(socket_type), name_space)
  socketio.on_event('disconnect', make_disconnected(), name_space)
  socketio.on_event('next', make_next_index(socket_type), name_space)
  socketio.on_event('prev', make_prev_index(socket_type), name_space)
  socketio.on_event('index', make_goto_index(socket_type), name_space)
  socketio.on_event('record_latent', make_record_latent(), name_space)
