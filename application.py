import os
from flask import (
    Flask, render_template, session, request, copy_current_request_context,
    has_request_context, has_app_context, url_for, redirect, flash, g
)
from flask_socketio import (
    SocketIO, emit, join_room, leave_room, close_room, rooms, disconnect
)
import json
import eventlet
import backend.auth as auth
import backend.db as db
import backend.survey as survey
from moving_luggage.simulator import simulator
import moving_luggage.constants as const


eventlet.monkey_patch() 
TRAJECTORY_DIR = '/data/tw2020_trajectory'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sseo_teamwork2020'

app.register_blueprint(auth.bp)
app.register_blueprint(survey.bp)
app.add_url_rule('/', endpoint='consent')

db.init_app(app)
if not os.path.exists(db.DATABASE_DIR):
    os.makedirs(db.DATABASE_DIR)

socketio = SocketIO(app, async_mode="eventlet")
game_core = simulator(log_dir=TRAJECTORY_DIR)

AGENT1_ID = 0
AGENT2_ID = 1


def init_db():
    with app.app_context():
        dtbs = db.get_db()
        with app.open_resource('schema.sql') as f:
            dtbs.cursor().executescript(f.read().decode('utf-8'))
        dtbs.commit()


##### pages
@app.route('/hello')
def hello():
    return 'Hello, World!'


@app.route('/experiment')
@auth.login_required
def experiment():
    cur_user = g.user
    print(cur_user)
    return render_template('experiment.html', cur_user=cur_user)


@app.route('/instruction', methods=('GET', 'POST'))
@auth.login_required
def instruction():
    return render_template('instruction.html')


##### socketio methods
def update_html_canvas(objs, room_id):
    objs_json = json.dumps(objs)
    socketio.emit('draw_canvas', objs_json, room=room_id)


def on_game_end(room_id):
    socketio.emit('game_end', room=room_id)


@socketio.on('run_experiment')
def run_experiment(msg):
    env_id = request.sid

    game_core.set_callback_renderer(update_html_canvas)
    game_core.set_callback_game_end(on_game_end)
    game_core.add_new_env(env_id, 25)
    game_core.connect_agent_id(env_id, AGENT1_ID)
    game_core.set_user_name(env_id, msg['data'])
    # game_core.run_game(env_id)

    obj_list = game_core.take_a_step_and_get_objs(
        env_id, AGENT1_ID, const.AgentActions.STAY)
    if obj_list is not None:
        update_html_canvas(obj_list, env_id)


@socketio.on('keydown_event')
def on_key_down(msg):
    env_id = request.sid

    agent_id = AGENT1_ID
    action = const.AgentActions.STAY

    key_code = msg["data"]
    if key_code == "ArrowLeft":  # Left
        agent_id = AGENT1_ID
        action = const.AgentActions.LEFT
    elif key_code == "ArrowRight":  # Right
        agent_id = AGENT1_ID
        action = const.AgentActions.RIGHT
    elif key_code == "ArrowUp":  # Up
        agent_id = AGENT1_ID
        action = const.AgentActions.UP
    elif key_code == "ArrowDown":  # Down
        agent_id = AGENT1_ID
        action = const.AgentActions.DOWN
    elif key_code == "p":  # p
        agent_id = AGENT1_ID
        action = const.AgentActions.HOLD
    elif key_code == "o":  # o
        agent_id = AGENT1_ID
        action = const.AgentActions.STAY
    elif key_code == "a":  # a
        agent_id = AGENT2_ID
        action = const.AgentActions.LEFT
    elif key_code == "d":  # d
        agent_id = AGENT2_ID
        action = const.AgentActions.RIGHT
    elif key_code == "w":  # w
        agent_id = AGENT2_ID
        action = const.AgentActions.UP
    elif key_code == "s":  # s
        agent_id = AGENT2_ID
        action = const.AgentActions.DOWN
    elif key_code == "t":  # t
        agent_id = AGENT2_ID
        action = const.AgentActions.HOLD
    elif key_code == "y":  # y
        agent_id = AGENT2_ID
        action = const.AgentActions.STAY

    # if agent_id is None and action is not None:
        # game_core.action_input(env_id, agent_id, action)
    obj_list = game_core.take_a_step_and_get_objs(env_id, agent_id, action)
    if obj_list is not None:
        update_html_canvas(obj_list, env_id)


@socketio.on('my_echo')
def test_message(message):
    print(message['data'])
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit(
        'my_response',
        {'data': message['data'], 'count': session['receive_count']})


@socketio.on('disconnect_request')
def disconnect_request():
    @copy_current_request_context
    def can_disconnect():
        disconnect()

    session['receive_count'] = session.get('receive_count', 0) + 1
    # for this emit we use a callback function
    # when the callback function is invoked we know that the message has been
    # received and it is safe to disconnect
    emit('my_response',
         {'data': 'Disconnected!', 'count': session['receive_count']},
         callback=can_disconnect)


@socketio.on('my_ping')
def ping_pong():
    emit('my_pong')


@socketio.on('connect')
def initial_canvas():
    env_dict = {
        'grid_x': game_core.grid_x, 'grid_y': game_core.grid_y, 'goals': []}

    def coord2idx(coord):
        return coord[1] * game_core.grid_x + coord[0]

    for pos in game_core.goal_pos:
        env_dict['goals'].append(coord2idx(pos))

    env_json = json.dumps(env_dict)
    emit('init_canvas', env_json)


@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected', request.sid)
    game_core.finish_game(request.sid)
  

if __name__ == '__main__':
    socketio.run(app)