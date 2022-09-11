from typing import Mapping
from flask import request
from web_experiment import socketio
from web_experiment.define import ExpType
import web_experiment.exp_common.events_impl as event_impl
from web_experiment.exp_common.page_base import Exp1UserData
from web_experiment.demo.define import E_SessionName, GAMEPAGES

g_id_2_user_data = {}  # type: Mapping[str, Exp1UserData]

for e_session in E_SessionName:
  name_space = '/' + e_session.name

  def make_init_canvas(e_session: E_SessionName):
    def initial_canvas():
      global g_id_2_user_data
      env_id = request.sid
      user_data = Exp1UserData(user=None)
      g_id_2_user_data[env_id] = user_data

      user_data.data[Exp1UserData.EXP_TYPE] = ExpType.Data_collection
      user_data.data[Exp1UserData.SESSION_DONE] = False

      domain_type = GAMEPAGES[e_session][0]._DOMAIN_TYPE
      event_impl.initial_canvas(e_session.name, user_data, GAMEPAGES[e_session],
                                domain_type)

    return initial_canvas

  def make_disconnected():
    def disconnected():
      global g_id_2_user_data
      env_id = request.sid
      if env_id in g_id_2_user_data:
        del g_id_2_user_data[env_id]

    return disconnected

  def make_button_clicked(e_session: E_SessionName):
    def button_clicked(msg):
      global g_id_2_user_data
      button = msg["name"]
      env_id = request.sid
      user_data = g_id_2_user_data[env_id]
      event_impl.button_clicked(button, user_data, GAMEPAGES[e_session])

    return button_clicked

  socketio.on_event('connect', make_init_canvas(e_session), name_space)
  socketio.on_event('disconnect', make_disconnected(), name_space)
  socketio.on_event('button_clicked', make_button_clicked(e_session),
                    name_space)
