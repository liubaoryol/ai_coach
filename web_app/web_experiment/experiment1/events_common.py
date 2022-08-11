from web_experiment import socketio
import web_experiment.experiment1.events_impl as event_impl
import web_experiment.experiment1.task_data as td

for session_name in td.EXP1_PAGENAMES:
  name_space = '/' + td.EXP1_PAGENAMES[session_name]

  def make_init_canvas(session_name):
    def initial_canvas():
      event_impl.initial_canvas(session_name, td.EXP1_TASK_TYPES[session_name])

    return initial_canvas

  def make_disconnected(session_name):
    def disconnected():
      event_impl.disconnected(td.map_g_id_2_game[session_name])

    return disconnected

  socketio.on_event('connect', make_init_canvas(session_name), name_space)
  socketio.on_event('my_echo', event_impl.test_message, name_space)
  socketio.on_event('disconnect_request', event_impl.disconnect_request,
                    name_space)
  socketio.on_event('my_ping', event_impl.ping_pong, name_space)
  socketio.on_event('disconnect', make_disconnected(session_name), name_space)

# only sessions (not tutorials) need these event handler commonly
for session_name in td.LIST_SESSIONS:
  name_space = '/' + td.EXP1_PAGENAMES[session_name]

  def make_done_task(session_name):
    def done_task(msg):
      event_impl.done_task(msg, session_name)

    return done_task

  def make_setting_event(session_name, name_space=name_space):
    def setting_event(msg):
      event_impl.setting_event(msg, td.map_g_id_2_game[session_name],
                               name_space)

    return setting_event

  socketio.on_event('done_task', make_done_task(session_name), name_space)
  socketio.on_event('setting_event', make_setting_event(session_name),
                    name_space)
