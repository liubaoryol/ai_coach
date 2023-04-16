from typing import Mapping, Any
from flask import session, request
from flask_socketio import emit
from web_experiment import socketio
from web_experiment.exp_common.page_replay import UserDataReplay
from web_experiment.review.util import (update_canvas, update_latent_state,
                                        get_init_user_data, load_trajectory,
                                        predict_human_latent_full)
from web_experiment.feedback.define import (FEEDBACK_NAMESPACES,
                                            FEEDBACK_CANVAS_PAGELIST)
from web_experiment.feedback.helper import load_latent
from web_experiment.define import EMode, GroupName

g_id_2_session_data = {}  # type: Mapping[Any, UserDataReplay]


def update_canvas_helper(sid,
                         name_space,
                         domain_type,
                         groupid,
                         user_data: UserDataReplay,
                         init_imgs=False):
  update_canvas(sid, name_space, FEEDBACK_CANVAS_PAGELIST[domain_type][0],
                user_data, init_imgs, domain_type)
  if groupid == GroupName.Group_C:
    update_latent_state(domain_type, EMode.Collected, user_data)
  elif groupid == GroupName.Group_D:
    update_latent_state(domain_type, EMode.Predicted, user_data)


for domain_type in FEEDBACK_NAMESPACES:
  name_space = '/' + FEEDBACK_NAMESPACES[domain_type]

  def make_init_canvas(domain_type, name_space=name_space):

    def init_canvas():
      global g_id_2_session_data
      sid = request.sid
      cur_user = session.get('user_id')
      session_name = session.get('loaded_session_name')
      groupid = session.get('groupid')

      trajectory = load_trajectory(session_name, cur_user)
      if trajectory is None:
        return

      user_data = get_init_user_data(cur_user, session_name, trajectory)
      user_data.data[UserDataReplay.GROUP_ID] = groupid

      if groupid == GroupName.Group_C:
        user_data.data[UserDataReplay.LATENT_COLLECTED] = load_latent(
            cur_user, session_name)
      elif groupid == GroupName.Group_D:
        lstates_full = predict_human_latent_full(trajectory, domain_type)
        lstates_full.append("None")
        user_data.data[UserDataReplay.LATENT_PREDICTED] = lstates_full

      g_id_2_session_data[sid] = user_data
      max_idx = len(trajectory) - 1

      update_canvas_helper(sid, name_space, domain_type, groupid, user_data,
                           True)
      emit("set_max", {"max_index": max_idx})

    return init_canvas

  def make_next_index(domain_type, name_space=name_space):

    def next_index():
      global g_id_2_session_data
      sid = request.sid
      if sid not in g_id_2_session_data:
        return
      user_data = g_id_2_session_data[sid]

      max_index = len(user_data.data[UserDataReplay.TRAJECTORY]) - 1
      if user_data.data[UserDataReplay.TRAJ_IDX] < max_index:
        user_data.data[UserDataReplay.TRAJ_IDX] += 1
      update_canvas_helper(sid, name_space, domain_type,
                           user_data.data[UserDataReplay.GROUP_ID], user_data)

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
      update_canvas_helper(sid, name_space, domain_type,
                           user_data.data[UserDataReplay.GROUP_ID], user_data)

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
      update_canvas_helper(sid, name_space, domain_type,
                           user_data.data[UserDataReplay.GROUP_ID], user_data)

    return goto_index

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
