from web_experiment import socketio
import web_experiment.experiment1.events_impl as event_impl
import web_experiment.experiment1.task_data as td

for session_name in td.EXP1_PAGENAMES:
  name_space = '/' + td.EXP1_PAGENAMES[session_name]

  def initial_canvas(msg, session_name=session_name):
    event_impl.initial_canvas(session_name, td.EXP1_TASK_TYPES[session_name])

  def disconnected(g_id_2_game=td.map_g_id_2_game[session_name]):
    event_impl.disconnected(g_id_2_game)

  socketio.on_event('connect', initial_canvas, name_space)
  socketio.on_event('my_echo', event_impl.test_message, name_space)
  socketio.on_event('disconnect_request', event_impl.disconnect_request,
                    name_space)
  socketio.on_event('my_ping', event_impl.ping_pong, name_space)
  socketio.on_event('disconnect', disconnected, name_space)

# only sessions (not tutorials) need these event handler commonly
for session_name in td.LIST_SESSIONS:
  name_space = '/' + td.EXP1_PAGENAMES[session_name]

  def done_task(msg, session_name=session_name):
    event_impl.done_task(msg, session_name)

  def setting_event(msg,
                    g_id_2_game=td.map_g_id_2_game[session_name],
                    name_space=name_space):
    event_impl.setting_event(msg, g_id_2_game, name_space)

  socketio.on_event('done_task', done_task, name_space)
  socketio.on_event('setting_event', setting_event, name_space)
