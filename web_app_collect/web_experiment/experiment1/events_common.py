from typing import Mapping
from flask import request, session, current_app
from web_experiment import socketio
from web_experiment.models import User
import web_experiment.experiment1.events_impl as event_impl
import web_experiment.experiment1.task_define as td
from web_experiment.experiment1.page_exp1_base import Exp1UserData

g_id_2_user_data = {}  # type: Mapping[str, Exp1UserData]

for session_name in td.EXP1_PAGENAMES:
  name_space = '/' + td.EXP1_PAGENAMES[session_name]

  def make_init_canvas(session_name):
    def initial_canvas():
      global g_id_2_user_data
      cur_user = session.get('user_id')
      user = User.query.filter_by(userid=cur_user).first()
      env_id = request.sid
      user_data = Exp1UserData(user=user)
      g_id_2_user_data[env_id] = user_data
      user_data.data[
          Exp1UserData.SAVE_PATH] = current_app.config["TRAJECTORY_PATH"]

      event_impl.initial_canvas(session_name, user_data,
                                td.EXP1_GAMEPAGES[session_name])

    return initial_canvas

  def make_disconnected(session_name):
    def disconnected():
      global g_id_2_user_data
      env_id = request.sid
      if env_id in g_id_2_user_data:
        del g_id_2_user_data[env_id]

    return disconnected

  def make_button_clicked(session_name):
    def button_clicked(msg):
      global g_id_2_user_data
      button = msg["name"]
      env_id = request.sid
      user_data = g_id_2_user_data[env_id]
      event_impl.button_clicked(button, session_name, user_data,
                                td.EXP1_GAMEPAGES[session_name])

    return button_clicked

  socketio.on_event('connect', make_init_canvas(session_name), name_space)
  socketio.on_event('disconnect', make_disconnected(session_name), name_space)
  socketio.on_event('button_clicked', make_button_clicked(session_name),
                    name_space)
