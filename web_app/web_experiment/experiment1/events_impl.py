import json
import logging
from flask import (session, request, current_app, url_for)
from flask_socketio import emit
from web_experiment import socketio
from web_experiment.models import User
import web_experiment.experiment1.task_data as td
import web_experiment.experiment1.page_base as pg
import web_experiment.experiment1.canvas_objects as co


def initial_canvas(session_name):
  cur_user = session.get('user_id')
  user = User.query.filter_by(userid=cur_user).first()

  env_id = request.sid
  td.map_g_id_2_game_data[session_name][env_id] = pg.UserGameData(user=user)

  done = getattr(user, session_name, False)

  user_game_data = td.map_g_id_2_game_data[session_name][env_id]
  user_game_data.num_pages = len(td.EXP1_GAMEPAGES[session_name])
  user_game_data.session_name = session_name
  user_game_data.save_path = current_app.config["TRAJECTORY_PATH"]

  if done:
    user_game_data.cur_page_idx = user_game_data.num_pages - 1
  else:
    user_game_data.cur_page_idx = 0

  cur_page = td.EXP1_GAMEPAGES[session_name][user_game_data.cur_page_idx]

  page_drawing_info = cur_page.init_user_data_and_get_drawing_info(
      user_game_data)
  commands, drawing_objs, drawing_order, animations = page_drawing_info

  # yapf: disable
  imgs = [
      {'name': co.IMG_ROBOT, 'src': url_for('exp1.static', filename='robot.svg')},  # noqa: E501
      {'name': co.IMG_WOMAN, 'src': url_for('exp1.static', filename='woman.svg')},  # noqa: E501
      {'name': co.IMG_MAN, 'src': url_for('exp1.static', filename='man.svg')},
      {'name': co.IMG_BOX, 'src': url_for('exp1.static', filename='box.svg')},
      {'name': co.IMG_TRASH_BAG, 'src': url_for('exp1.static', filename='trash_bag.svg')},  # noqa: E501
      {'name': co.IMG_WALL, 'src': url_for('exp1.static', filename='wall.svg')},
      {'name': co.IMG_GOAL, 'src': url_for('exp1.static', filename='goal.svg')},
      {'name': co.IMG_BOTH_BOX, 'src': url_for('exp1.static', filename='both_box.svg')},  # noqa: E501
      {'name': co.IMG_MAN_BAG, 'src': url_for('exp1.static', filename='man_bag.svg')},  # noqa: E501
      {'name': co.IMG_ROBOT_BAG, 'src': url_for('exp1.static', filename='robot_bag.svg')},  # noqa: E501
  ]
  # yapf: enable

  update_gamedata(commands=commands,
                  imgs=imgs,
                  drawing_objects=drawing_objs,
                  drawing_order=drawing_order,
                  animations=animations)


def disconnected(session_name):
  id_2_game_data = td.map_g_id_2_game_data[session_name]

  env_id = request.sid
  # finish current game
  if env_id in id_2_game_data:
    del id_2_game_data[env_id]


def button_clicked(button, session_name):
  env_id = request.sid
  user_game_data = td.map_g_id_2_game_data[session_name][env_id]
  page_idx = user_game_data.cur_page_idx
  prev_task_done = getattr(user_game_data.user, session_name, False)

  page_drawing_info = td.EXP1_GAMEPAGES[session_name][page_idx].button_clicked(
      user_game_data, button)

  if page_idx != user_game_data.cur_page_idx:
    updated_page = td.EXP1_GAMEPAGES[session_name][user_game_data.cur_page_idx]
    page_drawing_info = updated_page.init_user_data_and_get_drawing_info(
        user_game_data)

  updated_task_done = getattr(user_game_data.user, session_name, False)

  commands, drawing_objs, drawing_order, animations = page_drawing_info
  update_gamedata(commands=commands,
                  drawing_objects=drawing_objs,
                  drawing_order=drawing_order,
                  animations=animations)

  if not prev_task_done and updated_task_done:
    done_task(user_game_data.user.userid, session_name)


# emit methods
def update_gamedata(commands=None,
                    imgs=None,
                    drawing_objects=None,
                    drawing_order=None,
                    animations=None):
  update_data = {}
  if commands is not None:
    update_data["commands"] = commands

  if imgs is not None:
    update_data["imgs"] = imgs

  if drawing_objects is not None:
    update_data["drawing_objects"] = pg.get_objs_as_dictionary(drawing_objects)

  if drawing_order is not None:
    update_data["drawing_order"] = drawing_order

  if animations is not None:
    update_data["animations"] = animations

  objs_json = json.dumps(update_data)
  str_emit = 'update_gamedata'
  emit(str_emit, objs_json)


# socketio methods
def update_gamedata_from_server(room_id,
                                name_space,
                                commands=None,
                                imgs=None,
                                drawing_objects=None,
                                drawing_order=None,
                                animations=None):
  update_data = {}
  if commands is not None:
    update_data["commands"] = commands

  if imgs is not None:
    update_data["imgs"] = imgs

  if drawing_objects is not None:
    update_data["drawing_objects"] = pg.get_objs_as_dictionary(drawing_objects)

  if drawing_order is not None:
    update_data["drawing_order"] = drawing_order

  if animations is not None:
    update_data["animations"] = animations

  objs_json = json.dumps(update_data)
  str_emit = 'update_gamedata'
  socketio.emit(str_emit, objs_json, room=room_id, namespace=name_space)


def done_task(user_id, session_name):
  emit('task_end')
  logging.info("User %s completed %s" % (user_id, session_name))
