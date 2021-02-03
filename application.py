import os
from flask import (
    Flask, render_template, session, request, copy_current_request_context,
    has_request_context, has_app_context, url_for, redirect, flash, g
)
from flask_socketio import (
    SocketIO, emit, join_room, leave_room, close_room, rooms, disconnect
)
import json
import tw_app
import eventlet
import auth
import db
import survey

eventlet.monkey_patch() 

# TRAJECTORY_DIR = '/data/tw2020_trajectory'
TRAJECTORY_DIR = './data/tw2020_trajectory'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sseo_teamwork2020'

app.register_blueprint(auth.bp)
app.register_blueprint(survey.bp)

app.add_url_rule('/', endpoint='consent')
db.init_app(app)

socketio = SocketIO(app, async_mode="eventlet")
teamwork_app = tw_app.CAppTeamwork(
    "initial_data.xml", tw_app.POLICY_RANDOM, log_dir=TRAJECTORY_DIR)

if not os.path.exists(db.DATABASE_DIR):
    os.makedirs(db.DATABASE_DIR)

current_user = None


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
def update_html_canvas(obj_list, cur_user):
    x = {"objects": obj_list}
    y = json.dumps(x)
    socketio.emit('draw_canvas', y, room=cur_user)


def on_game_end():
    global current_user
    current_user = None
    socketio.emit('game_end')


@socketio.on('run_experiment')
def run_experiment(msg):
    print("RUN")
    if teamwork_app.run_on_web(update_html_canvas, on_game_end, request.sid, msg["data"]):
        global current_user
        current_user = request.sid
    else:
        session['receive_count'] = session.get('receive_count', 0) + 1
        if current_user == request.sid:
            emit('my_response', {'data': "It is already running.", 'count': session['receive_count']})
        else:
            emit('my_response', {'data': "Running by another user currently ", 'count': session['receive_count']})


@socketio.on('keydown_event')
def on_key_down(msg):
    global current_user
    if current_user != request.sid:
        session['receive_count'] = session.get('receive_count', 0) + 1
        emit('my_response', {'data': "Running by another user currently", 'count': session['receive_count']})
        return

    key_code = msg["data"]
    key_name = None
    if key_code == 37:
        key_name = "Left"
    elif key_code == 39:
        key_name = "Right"
    elif key_code == 38:
        key_name = "Up"
    elif key_code == 40:
        key_name = "Down"
    elif key_code == 65:
        key_name = "a"
    elif key_code == 68:
        key_name = "d"
    elif key_code == 87:
        key_name = "w"
    elif key_code == 83:
        key_name = "s"
    elif key_code == 80:
        key_name = "p"
    elif key_code == 84:
        key_name = "t"
    elif key_code == 79:
        key_name = "o"
    elif key_code == 82:
        key_name = "r"
    elif key_code == 186:
        key_name = "semicolon"
    elif key_code == 70:
        key_name = "f"
    
    class keyevent():
        keysym: str = None
    
    ke = keyevent()
    ke.keysym = key_name

    teamwork_app.gui_key_cb(ke)


@socketio.on('my_echo' )
def test_message(message):
    print(message['data'])
    if message['data'] == 'test_json':
        x = {
            "name": "test_json_object",
            "objects": [
                {"type": "rectangle", "color": "blue", "x": 50, "y": 60, "w": 30, "h": 10},
                {"type": "circle", "color": "red", "x": 100, "y": 100, "r": 10},
                {"type": "polygon", "color": "green", "coord": (200, 200, 230, 200, 215, 170)},
                {"type": "text",  "color": "black", "content": "BLAH", "x": 100, "y": 100},
            ]
        }

        y = json.dumps(x)
        emit('my_json_response', y)
    else:
        session['receive_count'] = session.get('receive_count', 0) + 1
        emit('my_response', {'data': message['data'], 'count': session['receive_count']})


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
    grid_x, grid_y = teamwork_app.get_grid_size()
    emit('init_canvas', {'grid_x': grid_x, 'grid_y': grid_y})


@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected', request.sid)
    global current_user
    if current_user == request.sid:
        teamwork_app.end_application()
        current_user = None
  

if __name__ == '__main__':

    socketio.run(app)