from typing import Mapping
from flask import request, session, current_app
from web_experiment import socketio
from web_experiment.models import User, ExpIntervention
from web_experiment.define import ExpType, INTERV_SESSIONS, GroupName
import web_experiment.exp_common.events_impl as event_impl
from web_experiment.exp_intervention.define import (GAMEPAGES,
                                                    get_socket_namespace,
                                                    get_socket_type)
from web_experiment.exp_common.page_exp1_base import Exp1UserData

g_id_2_user_data = {}  # type: Mapping[str, Exp1UserData]

for page_key in INTERV_SESSIONS:
  visited_namespaces = []
  for group_key in GroupName.__dict__:
    if group_key.startswith('__') or group_key.startswith('_'):
      continue

    groupid = GroupName.__dict__[group_key]
    name_space = '/' + get_socket_namespace(page_key, groupid)

    # prevent adding same handlers multiple times
    if name_space in visited_namespaces:
      continue
    visited_namespaces.append(name_space)

    def make_init_canvas(page_key=page_key, groupid=groupid):
      def initial_canvas():
        global g_id_2_user_data
        cur_user = session.get('user_id')
        user = User.query.filter_by(userid=cur_user).first()
        env_id = request.sid
        user_data = Exp1UserData(user=user)
        g_id_2_user_data[env_id] = user_data

        user_data.data[
            Exp1UserData.SAVE_PATH] = current_app.config["TRAJECTORY_PATH"]
        user_data.data[Exp1UserData.EXP_TYPE] = ExpType.Intervention

        expinfo = ExpIntervention.query.filter_by(subject_id=cur_user).first()
        user_data.data[Exp1UserData.SESSION_DONE] = getattr(expinfo, page_key)

        event_impl.initial_canvas(page_key, user_data,
                                  GAMEPAGES[get_socket_type(page_key, groupid)])

      return initial_canvas

    def make_disconnected():
      def disconnected():
        global g_id_2_user_data
        env_id = request.sid
        if env_id in g_id_2_user_data:
          del g_id_2_user_data[env_id]

      return disconnected

    def make_button_clicked(page_key=page_key, groupid=groupid):
      def button_clicked(msg):
        global g_id_2_user_data
        button = msg["name"]
        env_id = request.sid
        user_data = g_id_2_user_data[env_id]
        event_impl.button_clicked(button, page_key, user_data,
                                  GAMEPAGES[get_socket_type(page_key, groupid)])

      return button_clicked

    socketio.on_event('connect', make_init_canvas(), name_space)
    socketio.on_event('disconnect', make_disconnected(), name_space)
    socketio.on_event('button_clicked', make_button_clicked(), name_space)
