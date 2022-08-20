from web_experiment import socketio
import web_experiment.experiment1.events_impl as event_impl
import web_experiment.experiment1.task_define as td

for session_name in td.EXP1_PAGENAMES:
  name_space = '/' + td.EXP1_PAGENAMES[session_name]

  def make_init_canvas(session_name):
    def initial_canvas():
      event_impl.initial_canvas(session_name)

    return initial_canvas

  def make_disconnected(session_name):
    def disconnected():
      event_impl.disconnected(session_name)

    return disconnected

  def make_button_clicked(session_name):
    def button_clicked(msg):
      button = msg["name"]
      event_impl.button_clicked(button, session_name)

    return button_clicked

  socketio.on_event('connect', make_init_canvas(session_name), name_space)
  socketio.on_event('disconnect', make_disconnected(session_name), name_space)
  socketio.on_event('button_clicked', make_button_clicked(session_name),
                    name_space)
