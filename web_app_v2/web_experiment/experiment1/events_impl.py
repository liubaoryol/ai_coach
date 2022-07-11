from typing import Mapping, Hashable, Callable
import os
import time
import json
import logging
import copy
from flask import session, request, copy_current_request_context, current_app
from flask_socketio import emit, disconnect
from ai_coach_domain.box_push import BoxState, conv_box_idx_2_state, EventType
from ai_coach_domain.box_push.simulator import (BoxPushSimulator,
                                                BoxPushSimulator_AlwaysTogether,
                                                BoxPushSimulator_AlwaysAlone)
from ai_coach_domain.box_push.agent import (BoxPushInteractiveAgent,
                                            BoxPushAIAgent_Abstract)
from web_experiment import socketio
from web_experiment.models import db, User
import web_experiment.experiment1.helper as helper


ASK_LATENT = True
NOT_ASK_LATENT = False
SHOW_FAILURE = True
NOT_SHOW_FAILURE = False


def get_file_name(user_id, session_name):
  traj_dir = os.path.join(current_app.config["TRAJECTORY_PATH"], user_id)
  # save somewhere
  if not os.path.exists(traj_dir):
    os.makedirs(traj_dir)

  sec, msec = divmod(time.time() * 1000, 1000)
  time_stamp = '%s.%03d' % (time.strftime('%Y-%m-%d_%H_%M_%S',
                                          time.gmtime(sec)), msec)
  file_name = session_name + '_' + str(user_id) + '_' + time_stamp + '.txt'
  return os.path.join(traj_dir, file_name)


def initial_canvas(grid_x, grid_y):
  env_dict = {'grid_x': grid_x, 'grid_y': grid_y}

  env_json = json.dumps(env_dict)
  emit('init_canvas', env_json)


def test_message(message):
  # print(message['data'])
  session['receive_count'] = session.get('receive_count', 0) + 1
  emit('my_response', {
      'data': message['data'],
      'count': session['receive_count']
  })


def disconnect_request():
  @copy_current_request_context
  def can_disconnect():
    disconnect()

  session['receive_count'] = session.get('receive_count', 0) + 1
  # for this emit we use a callback function
  # when the callback function is invoked we know that the message has been
  # received and it is safe to disconnect
  emit('my_response', {
      'data': 'Exp1 disconnected!',
      'count': session['receive_count']
  },
       callback=can_disconnect)


def ping_pong():
  emit('my_pong')


def test_disconnect(id_2_game: Mapping[Hashable, BoxPushSimulator]):
  env_id = request.sid
  # finish current game
  if env_id in id_2_game:
    del id_2_game[env_id]
  # print('Exp1 client disconnected', env_id)


# socketio methods
def update_html_canvas(objs, room_id, ask_latent, name_space):
  objs["ask_latent"] = ask_latent
  objs_json = json.dumps(objs)
  str_emit = 'draw_canvas'
  socketio.emit(str_emit, objs_json, room=room_id, namespace=name_space)


def on_game_end(room_id, name_space, user_id, session_name, score, is_a):
  socketio.emit('game_end', room=room_id, namespace=name_space)
  user = User.query.filter_by(userid=user_id).first()

  if user is not None:
    best_score = get_best_score(user_id, is_a)
    if best_score > score:
      if is_a:
        user.best_a = score
      else:
        user.best_b = score

    setattr(user, session_name, True)
    db.session.commit()


def get_best_score(user_id, is_a):
  user = User.query.filter_by(userid=user_id).first()
  if is_a:
    return user.best_a
  else:
    return user.best_b


def are_agent_states_changed(dict_env_prev, game: BoxPushSimulator):
  KEY_A1_POS = "a1_pos"
  KEY_A2_POS = "a2_pos"
  KEY_BOX_STATES = "box_states"
  num_drops = len(dict_env_prev["drops"])
  num_goals = len(dict_env_prev["goals"])

  a1_pos_changed = False
  a2_pos_changed = False
  if dict_env_prev[KEY_A1_POS] != game.a1_pos:
    a1_pos_changed = True

  if dict_env_prev[KEY_A2_POS] != game.a2_pos:
    a2_pos_changed = True

  box_states_prev = dict_env_prev[KEY_BOX_STATES]
  a1_box_prev = -1
  a2_box_prev = -1
  for idx in range(len(box_states_prev)):
    state = conv_box_idx_2_state(box_states_prev[idx], num_drops, num_goals)
    if state[0] == BoxState.WithAgent1:  # with a1
      a1_box_prev = idx
    elif state[0] == BoxState.WithAgent2:  # with a2
      a2_box_prev = idx
    elif state[0] == BoxState.WithBoth:  # with both
      a1_box_prev = idx
      a2_box_prev = idx

  box_states = game.box_states
  a1_box = -1
  a2_box = -1
  for idx in range(len(box_states)):
    state = conv_box_idx_2_state(box_states[idx], num_drops, num_goals)
    if state[0] == BoxState.WithAgent1:  # with a1
      a1_box = idx
    elif state[0] == BoxState.WithAgent2:  # with a2
      a2_box = idx
    elif state[0] == BoxState.WithBoth:  # with both
      a1_box = idx
      a2_box = idx

  a1_hold_changed = False
  a2_hold_changed = False

  if a1_box_prev != a1_box:
    a1_hold_changed = True

  if a2_box_prev != a2_box:
    a2_hold_changed = True

  return (a1_pos_changed, a2_pos_changed, a1_hold_changed, a2_hold_changed,
          a1_box, a2_box)


def get_valid_box_to_pickup(game: BoxPushSimulator):
  num_drops = len(game.drops)
  num_goals = len(game.goals)

  valid_box = []

  box_states = game.box_states
  for idx in range(len(box_states)):
    state = conv_box_idx_2_state(box_states[idx], num_drops, num_goals)
    if state[0] in [BoxState.Original, BoxState.OnDropLoc]:  # with a1
      valid_box.append(idx)

  return valid_box


def action_event(msg, id_2_game, cb_on_hold_change, cb_game_finished,
                 name_space, prompt_on_change, auto_prompt, prompt_freq, intervention = False):
  env_id = request.sid
  action_name = msg["data"]
  align_a2_action = "aligned" in msg

  action = None
  if action_name == "Left":
    action = EventType.LEFT
  elif action_name == "Right":
    action = EventType.RIGHT
  elif action_name == "Up":
    action = EventType.UP
  elif action_name == "Down":
    action = EventType.DOWN
  elif action_name == "Pick Up":
    action = EventType.HOLD
  elif action_name == "Drop":
    action = EventType.UNHOLD
  elif action_name == "Stay":
    action = EventType.STAY

  if action is None:
    return

  game = id_2_game[env_id]  # type: BoxPushSimulator
  if game.is_finished():
    return

  dict_env_prev = copy.deepcopy(game.get_env_info())

  game.event_input(BoxPushSimulator.AGENT1, action, None)
  if align_a2_action:
    game.event_input(BoxPushSimulator.AGENT2, action, None)

  map_agent2action = game.get_joint_action()
  game.take_a_step(map_agent2action)
  
  # implement intervention during game play
  if intervention:
    helper.task_intervention(game.history, game)

  if not game.is_finished():
    (a1_pos_changed, a2_pos_changed, a1_hold_changed, a2_hold_changed, a1_box,
     a2_box) = are_agent_states_changed(dict_env_prev, game)
    unchanged_agents = []
    if not a1_pos_changed and not a1_hold_changed:
      unchanged_agents.append(0)
    if not a2_pos_changed and not a2_hold_changed:
      unchanged_agents.append(1)

    draw_overlay = False
    if prompt_on_change and (a1_hold_changed or a2_hold_changed):
      draw_overlay = True

    if cb_on_hold_change:
      cb_on_hold_change(game, a1_hold_changed, a2_hold_changed, a1_box, a2_box)

    dict_update = game.get_changed_objects()
    if dict_update is None:
      dict_update = {}

    dict_update["unchanged_agents"] = unchanged_agents

    if auto_prompt:
      session['action_count'] = session.get('action_count', 0) + 1
      if session['action_count'] >= prompt_freq:
        draw_overlay = True

    update_html_canvas(dict_update, env_id, draw_overlay, name_space)
  else:
    if cb_game_finished:
      cb_game_finished(game, env_id, name_space)



    

def task_end(env_id, game: BoxPushSimulator, user_id, session_name, game_type,
             map_info, name_space, is_task_a):
  file_name = get_file_name(user_id, session_name)
  header = game_type + "-" + session_name + "\n"
  header += "User ID: %s\n" % (str(user_id), )
  header += str(map_info)
  game.save_history(file_name, header)

  on_game_end(env_id, name_space, user_id, session_name, game.current_step,
              is_task_a)
  logging.info("User %s completed %s" % (user_id, session_name))


def set_latent(msg, id_2_game, name_space):
  env_id = request.sid
  latent = msg["data"]

  game = id_2_game[env_id]  # type: BoxPushSimulator
  if game.is_finished():
    return

  game.event_input(BoxPushSimulator.AGENT1, EventType.SET_LATENT, tuple(latent))

  dict_update = game.get_changed_objects()
  session['action_count'] = 0
  update_html_canvas(dict_update, env_id, NOT_ASK_LATENT, name_space)


def run_task_game(msg: Mapping, id_2_game: Mapping,
                  teammate_agent: BoxPushAIAgent_Abstract,
                  cb_set_init_latent: Callable[[BoxPushSimulator], None], mdp,
                  game_map, ask_latent, name_space, is_a):
  env_id = request.sid

  # run a game
  if env_id not in id_2_game:
    if is_a:
      id_2_game[env_id] = BoxPushSimulator_AlwaysTogether(env_id)
    else:
      id_2_game[env_id] = BoxPushSimulator_AlwaysAlone(env_id)

  game = id_2_game[env_id]  # type: BoxPushSimulator
  game.init_game(**game_map)

  agent1 = BoxPushInteractiveAgent()
  agent2 = teammate_agent
  game.set_autonomous_agent(agent1, agent2)

  if cb_set_init_latent:
    cb_set_init_latent(game)

  dict_update = game.get_env_info()
  dict_update["wall_dir"] = game_map["wall_dir"]
  dict_update["best_score"] = get_best_score(msg["user_id"], is_a)
  if dict_update is not None:
    session["action_count"] = 0
    update_html_canvas(dict_update, env_id, ask_latent, name_space)
