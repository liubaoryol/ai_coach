from typing import Sequence
import json
import logging
from flask import url_for
from flask_socketio import emit
from web_experiment import socketio
from web_experiment.exp_common.page_base import (CanvasPageBase, UserData,
                                                 get_objs_as_dictionary)
import web_experiment.exp_common.canvas_objects as co


def get_imgs():
  # yapf: disable
  imgs = [
      {'name': co.IMG_ROBOT, 'src': url_for('static', filename='boxpush_images/robot.svg')},  # noqa: E501
      {'name': co.IMG_WOMAN, 'src': url_for('static', filename='boxpush_images/woman.svg')},  # noqa: E501
      {'name': co.IMG_MAN, 'src': url_for('static', filename='boxpush_images/man.svg')},  # noqa: E501
      {'name': co.IMG_BOX, 'src': url_for('static', filename='boxpush_images/box.svg')},  # noqa: E501
      {'name': co.IMG_TRASH_BAG, 'src': url_for('static', filename='boxpush_images/trash_bag.svg')},  # noqa: E501
      {'name': co.IMG_WALL, 'src': url_for('static', filename='boxpush_images/wall.svg')},  # noqa: E501
      {'name': co.IMG_GOAL, 'src': url_for('static', filename='boxpush_images/goal.svg')},  # noqa: E501
      {'name': co.IMG_BOTH_BOX, 'src': url_for('static', filename='boxpush_images/both_box.svg')},  # noqa: E501
      {'name': co.IMG_MAN_BAG, 'src': url_for('static', filename='boxpush_images/man_bag.svg')},  # noqa: E501
      {'name': co.IMG_ROBOT_BAG, 'src': url_for('static', filename='boxpush_images/robot_bag.svg')},  # noqa: E501
  ]
  # yapf: enable

  return imgs


def initial_canvas(session_name: str, user_game_data: UserData,
                   page_lists: Sequence[CanvasPageBase]):

  num_pages = len(page_lists)
  user_game_data.data[UserData.NUM_PAGES] = num_pages
  user_game_data.data[UserData.SESSION_NAME] = session_name

  if user_game_data.data[UserData.SESSION_DONE]:
    cur_page_idx = num_pages - 1
  else:
    cur_page_idx = 0

  user_game_data.data[UserData.PAGE_IDX] = cur_page_idx
  cur_page = page_lists[cur_page_idx]

  cur_page.init_user_data(user_game_data)
  page_drawing_info = cur_page.get_updated_drawing_info(user_game_data)
  commands, drawing_objs, drawing_order, animations = page_drawing_info

  imgs = get_imgs()

  update_gamedata(commands=commands,
                  imgs=imgs,
                  drawing_objects=drawing_objs,
                  drawing_order=drawing_order,
                  animations=animations)


def button_clicked(button, user_game_data: UserData,
                   page_lists: Sequence[CanvasPageBase]):
  page_idx = user_game_data.data[UserData.PAGE_IDX]
  user = user_game_data.data[UserData.USER]

  prev_task_done = user_game_data.data[UserData.SESSION_DONE]
  dict_prev_game_data = user_game_data.get_data_to_compare()

  page_lists[page_idx].button_clicked(user_game_data, button)
  page_drawing_info = page_lists[page_idx].get_updated_drawing_info(
      user_game_data, button, dict_prev_game_data)

  new_page_idx = user_game_data.data[UserData.PAGE_IDX]
  if page_idx != new_page_idx:
    updated_page = page_lists[new_page_idx]
    updated_page.init_user_data(user_game_data)
    page_drawing_info = updated_page.get_updated_drawing_info(user_game_data)

  updated_task_done = user_game_data.data[UserData.SESSION_DONE]

  commands, drawing_objs, drawing_order, animations = page_drawing_info
  update_gamedata(commands=commands,
                  drawing_objects=drawing_objs,
                  drawing_order=drawing_order,
                  animations=animations)

  if not prev_task_done and updated_task_done:
    done_task(user.userid, user_game_data.data[UserData.SESSION_NAME])


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
    update_data["drawing_objects"] = get_objs_as_dictionary(drawing_objects)

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
    update_data["drawing_objects"] = get_objs_as_dictionary(drawing_objects)

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
