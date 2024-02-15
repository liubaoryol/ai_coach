import logging
from typing import Mapping
from flask import request, session, current_app
from web_experiment import socketio
from web_experiment.models import User, ExpIntervention
from web_experiment.define import ExpType, get_domain_type
import web_experiment.exp_common.events_impl as event_impl
from web_experiment.exp_intervention.define import GAMEPAGES, SocketType
from web_experiment.exp_common.page_base import Exp1UserData

g_id_2_user_data = {}  # type: Mapping[str, Exp1UserData]

for socket_type in SocketType:
  name_space = '/' + socket_type.name

  def make_init_canvas(socket_type):
    def initial_canvas(name_space=name_space):
      global g_id_2_user_data
      cur_user = session.get('user_id')
      user = User.query.filter_by(userid=cur_user).first()
      env_id = request.sid
      user_data = Exp1UserData(user=user)
      g_id_2_user_data[env_id] = user_data

      user_data.data[
          Exp1UserData.SAVE_PATH] = current_app.config["TRAJECTORY_PATH"]
      user_data.data[Exp1UserData.USER_LABEL_PATH] = (
          current_app.config["USER_LABEL_PATH"])
      user_data.data[Exp1UserData.EXP_TYPE] = ExpType.Intervention

      session_name = session["loaded_session_name"]
      expinfo = ExpIntervention.query.filter_by(subject_id=cur_user).first()
      user_data.data[Exp1UserData.SESSION_DONE] = getattr(expinfo, session_name)
      logging.info(f"{cur_user}({env_id}) connected to socketio {name_space}")
      event_impl.initial_canvas(env_id, name_space, session_name, user_data,
                                GAMEPAGES[socket_type],
                                get_domain_type(session_name))

    return initial_canvas

  def make_disconnected():
    def disconnected():
      global g_id_2_user_data
      env_id = request.sid
      if env_id in g_id_2_user_data:
        del g_id_2_user_data[env_id]

    return disconnected

  def make_button_clicked(socket_type, name_space=name_space):
    def button_clicked(msg):
      global g_id_2_user_data
      button = msg["name"]
      env_id = request.sid
      user_data = g_id_2_user_data[env_id]
      event_impl.button_clicked(env_id, name_space, button, user_data,
                                GAMEPAGES[socket_type])

    return button_clicked

  socketio.on_event('connect', make_init_canvas(socket_type), name_space)
  socketio.on_event('disconnect', make_disconnected(), name_space)
  socketio.on_event('button_clicked', make_button_clicked(socket_type),
                    name_space)
