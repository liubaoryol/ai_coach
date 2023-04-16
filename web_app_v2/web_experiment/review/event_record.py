from typing import Mapping, Any
from flask import session, request
from flask_socketio import emit
from web_experiment import socketio
from web_experiment.define import PageKey
from web_experiment.exp_common.page_replay import UserDataReplay
from web_experiment.review.define import REPLAY_CANVAS_PAGELIST, get_socket_name
from web_experiment.review.util import (update_canvas, get_init_user_data,
                                        load_trajectory)

g_id_2_session_data = {}  # type: Mapping[Any, UserDataReplay]

for domain_type in REPLAY_CANVAS_PAGELIST:
  name_space = '/' + get_socket_name(PageKey.Record, domain_type)

  def make_init_canvas(domain_type, name_space=name_space):

    def init_canvas():
      global g_id_2_session_data
      sid = request.sid
      cur_user = session.get('user_id')
      session_name = session.get('loaded_session_name')

      trajectory = load_trajectory(session_name, cur_user)
      if trajectory is None:
        return
      user_data = get_init_user_data(cur_user, session_name, trajectory)
      g_id_2_session_data[sid] = user_data
      max_idx = len(trajectory) - 1
      user_data.data[UserDataReplay.LATENT_COLLECTED] = (["None"] *
                                                         len(trajectory))
      update_canvas(sid,
                    name_space,
                    REPLAY_CANVAS_PAGELIST[domain_type][0],
                    user_data,
                    init_imgs=True,
                    domain_type=domain_type)
      emit("set_max", {"max_index": max_idx})

    return init_canvas

  def make_next_index(domain_type, name_space=name_space):

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
                      REPLAY_CANVAS_PAGELIST[domain_type][0],
                      user_data,
                      init_imgs=False)

    return next_index

  def make_prev_index(domain_type, name_space=name_space):

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
                      REPLAY_CANVAS_PAGELIST[domain_type][0],
                      user_data,
                      init_imgs=False)

    return prev_index

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
                      REPLAY_CANVAS_PAGELIST[domain_type][0],
                      user_data,
                      init_imgs=False)

    return goto_index

  def make_record_latent():

    def record_latent_wrapper(msg):
      global g_id_2_session_data
      sid = request.sid
      if sid not in g_id_2_session_data:
        return
      user_data = g_id_2_session_data[sid]

      lstate = msg['latent']
      user_data.data[UserDataReplay.LATENT_COLLECTED][user_data.data[
          UserDataReplay.TRAJ_IDX]] = lstate
      print(user_data.data[UserDataReplay.LATENT_COLLECTED])

    return record_latent_wrapper

  def make_disconnected():

    def disconnected():
      global g_id_2_session_data
      sid = request.sid

      if sid in g_id_2_session_data:
        # delete session_data
        del g_id_2_session_data[sid]

    return disconnected

  socketio.on_event('connect', make_init_canvas(domain_type), name_space)
  socketio.on_event('disconnect', make_disconnected(), name_space)
  socketio.on_event('next', make_next_index(domain_type), name_space)
  socketio.on_event('prev', make_prev_index(domain_type), name_space)
  socketio.on_event('index', make_goto_index(domain_type), name_space)
  socketio.on_event('record_latent', make_record_latent(), name_space)
