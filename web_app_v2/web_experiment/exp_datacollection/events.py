from typing import Mapping
from flask import request, session, current_app
from web_experiment import socketio
from web_experiment.models import User, ExpDataCollection
from web_experiment.define import ExpType
import web_experiment.exp_common.events_impl as event_impl
from web_experiment.exp_common.page_exp1_base import Exp1UserData
from web_experiment.exp_datacollection.define import GAMEPAGES, SocketType

g_id_2_user_data = {}  # type: Mapping[str, Exp1UserData]

for socket_type in SocketType:
  name_space = '/' + socket_type.name

  def make_init_canvas(socket_type):
    def initial_canvas():
      global g_id_2_user_data
      cur_user = session.get('user_id')
      user = User.query.filter_by(userid=cur_user).first()
      env_id = request.sid
      user_data = Exp1UserData(user=user)
      g_id_2_user_data[env_id] = user_data

      user_data.data[Exp1UserData.SAVE_PATH] = (
          current_app.config["TRAJECTORY_PATH"])
      user_data.data[Exp1UserData.EXP_TYPE] = ExpType.Data_collection

      session_name = session["loaded_session_name"]
      expinfo = ExpDataCollection.query.filter_by(subject_id=cur_user).first()
      user_data.data[Exp1UserData.SESSION_DONE] = getattr(expinfo, session_name)

      event_impl.initial_canvas(session_name, user_data, GAMEPAGES[socket_type])

    return initial_canvas

  def make_disconnected():
    def disconnected():
      global g_id_2_user_data
      env_id = request.sid
      if env_id in g_id_2_user_data:
        del g_id_2_user_data[env_id]

    return disconnected

  def make_button_clicked(socket_type):
    def button_clicked(msg):
      global g_id_2_user_data
      button = msg["name"]
      env_id = request.sid
      user_data = g_id_2_user_data[env_id]
      event_impl.button_clicked(button, user_data, GAMEPAGES[socket_type])

    return button_clicked

  socketio.on_event('connect', make_init_canvas(socket_type), name_space)
  socketio.on_event('disconnect', make_disconnected(), name_space)
  socketio.on_event('button_clicked', make_button_clicked(socket_type),
                    name_space)