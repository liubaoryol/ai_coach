from typing import Mapping, Any, Sequence
import copy
import os
import time
from ai_coach_domain.box_push.simulator import (BoxPushSimulator_AlwaysTogether,
                                                BoxPushSimulator_AlwaysAlone,
                                                BoxPushSimulator)
from ai_coach_domain.box_push import EventType
from web_experiment.models import db, User
import web_experiment.experiment1.canvas_objects as co
import web_experiment.experiment1.page_base as pg


def get_file_name(save_path, user_id, session_name):
  traj_dir = os.path.join(save_path, user_id)
  # save somewhere
  if not os.path.exists(traj_dir):
    os.makedirs(traj_dir)

  sec, msec = divmod(time.time() * 1000, 1000)
  time_stamp = '%s.%03d' % (time.strftime('%Y-%m-%d_%H_%M_%S',
                                          time.gmtime(sec)), msec)
  file_name = session_name + '_' + str(user_id) + '_' + time_stamp + '.txt'
  return os.path.join(traj_dir, file_name)


def are_agent_states_changed(dict_prev_game: Mapping[str, Any],
                             dict_cur_game: Mapping[str, Any]):
  num_drops = len(dict_prev_game["drops"])
  num_goals = len(dict_prev_game["goals"])

  a1_pos_changed = False
  a2_pos_changed = False
  if dict_prev_game["a1_pos"] != dict_cur_game["a1_pos"]:
    a1_pos_changed = True

  if dict_prev_game["a2_pos"] != dict_cur_game["a2_pos"]:
    a2_pos_changed = True

  a1_box_prev, a2_box_prev = pg.get_holding_box_idx(
      dict_prev_game["box_states"], num_drops, num_goals)
  a1_box, a2_box = pg.get_holding_box_idx(dict_cur_game["box_states"],
                                          num_drops, num_goals)

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
    state = pg.conv_box_idx_2_state(box_states[idx], num_drops, num_goals)
    if state[0] in [pg.BoxState.Original, pg.BoxState.OnDropLoc]:  # with a1
      valid_box.append(idx)

  return valid_box


###############################################################################
# canvas page game
###############################################################################


class CanvasPageGame(pg.CanvasPageBase):
  def __init__(self,
               is_movers,
               manual_latent_selection,
               game_map,
               auto_prompt: bool = True,
               prompt_on_change: bool = True,
               prompt_freq: int = 5) -> None:
    super().__init__(True, True, True, True, is_movers)
    self._MANUAL_SELECTION = manual_latent_selection
    self._GAME_MAP = game_map

    self._PROMPT_ON_CHANGE = prompt_on_change
    self._PROMPT_FREQ = prompt_freq
    self._AUTO_PROMPT = auto_prompt

    self._AGENT1 = BoxPushSimulator.AGENT1
    self._AGENT2 = BoxPushSimulator.AGENT2

  def _init_user_data(self, user_game_data: pg.UserGameData):
    user_game_data.flags.done = False
    user_game_data.flags.aligned_a2_action = False
    user_game_data.flags.select = False

    if user_game_data.game is None:
      if self._IS_MOVERS:
        user_game_data.game = BoxPushSimulator_AlwaysTogether(None)
      else:
        user_game_data.game = BoxPushSimulator_AlwaysAlone(None)

    user_game_data.game.init_game(**self._GAME_MAP)

    user_game_data.flags.action_count = 0

  def button_clicked(self, user_game_data: pg.UserGameData, clicked_btn):
    '''
    user_game_data: NOTE - values will be updated
    return: commands, drawing_objs, drawing_order, animations
      drawing info
    '''

    if clicked_btn in co.ACTION_BUTTONS:
      original_page_idx = user_game_data.cur_page_idx
      dict_prev_game = copy.deepcopy(user_game_data.game.get_env_info())
      a1_act, a2_act, done = self.action_event(user_game_data, clicked_btn)

      if done:
        self._on_game_finished(user_game_data)
      else:
        self._on_action_taken(user_game_data, dict_prev_game, (a1_act, a2_act))

      if user_game_data.cur_page_idx != original_page_idx:
        # since user is not in this page anymore, we don't draw anything from
        # this page.
        return None, None, None, None

      dict_cur_game = user_game_data.game.get_env_info()
      if self._IS_MOVERS:
        best_score = user_game_data.user.best_a
      else:
        best_score = user_game_data.user.best_b

      commands = self._get_button_commands(clicked_btn, user_game_data.flags)
      updated_objs = self._get_updated_drawing_objects(dict_cur_game,
                                                       user_game_data.flags,
                                                       best_score, True)
      drawing_order = self._get_drawing_order(dict_cur_game,
                                              user_game_data.flags)
      anis = self._get_animations(dict_prev_game, dict_cur_game,
                                  self._IS_MOVERS)
      return commands, updated_objs, drawing_order, anis

    elif clicked_btn == co.BTN_SELECT:
      user_game_data.flags.select = True
      dict_cur_game = user_game_data.game.get_env_info()
      if self._IS_MOVERS:
        best_score = user_game_data.user.best_a
      else:
        best_score = user_game_data.user.best_b
      commands = self._get_button_commands(clicked_btn, user_game_data.flags)
      updated_objs = self._get_updated_drawing_objects(dict_cur_game,
                                                       user_game_data.flags,
                                                       best_score, False)
      drawing_order = self._get_drawing_order(dict_cur_game,
                                              user_game_data.flags)
      return commands, updated_objs, drawing_order, None
    elif co.is_sel_latent_btn(clicked_btn):
      latent = co.selbtn2latent(clicked_btn)
      if latent is not None:
        user_game_data.game.event_input(self._AGENT1, EventType.SET_LATENT,
                                        latent)
        user_game_data.flags.select = False
        user_game_data.flags.action_count = 0
        dict_cur_game = user_game_data.game.get_env_info()
        if self._IS_MOVERS:
          best_score = user_game_data.user.best_a
        else:
          best_score = user_game_data.user.best_b
        commands = self._get_button_commands(clicked_btn, user_game_data.flags)
        updated_objs = self._get_updated_drawing_objects(
            dict_cur_game, user_game_data.flags, best_score, False)
        drawing_order = self._get_drawing_order(dict_cur_game,
                                                user_game_data.flags)

        return commands, updated_objs, drawing_order, None

    return super().button_clicked(user_game_data, clicked_btn)

  def _get_instruction(self, flags: pg.GameFlags):
    if flags.select:
      return (
          "Please select your current destination among the circled options. " +
          "It can be the same destination as you had previously selected.")
    else:
      return (
          "Please choose your next action. If your destination has changed, " +
          "please update it using the select destination button.")

  def _get_drawing_order(self, game_env, flags: pg.GameFlags):
    drawing_order = []
    drawing_order.append(self.GAME_BORDER)

    drawing_order = drawing_order + self._game_scene_names(
        game_env, self._IS_MOVERS, flags)
    drawing_order = drawing_order + self._game_overlay_names(
        game_env, flags, not flags.select)
    drawing_order = drawing_order + co.ACTION_BUTTONS
    drawing_order.append(co.BTN_SELECT)

    drawing_order.append(self.TEXT_SCORE)

    drawing_order.append(self.RECT_INSTRUCTION)
    drawing_order.append(self.TEXT_INSTRUCTION)

    return drawing_order

  def _get_init_drawing_objects(
      self, game_env, flags: pg.GameFlags, score: int,
      best_score: int) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(game_env, flags, score,
                                                  best_score)

    select_disable = not self._MANUAL_SELECTION or flags.select
    dis_status = self._get_action_btn_disabled(game_env, flags)
    objs = self._get_btn_actions(*dis_status, select_disable=select_disable)
    for obj in objs:
      dict_objs[obj.name] = obj

    return dict_objs

  def _on_action_taken(self, user_cur_game_data: pg.UserGameData,
                       dict_prev_game: Mapping[str, Any],
                       tuple_actions: Sequence[Any]):
    '''
    user_cur_game_data: NOTE - values will be updated
    '''

    # set selection prompt status
    (a1_pos_changed, a2_pos_changed, a1_hold_changed, a2_hold_changed, a1_box,
     a2_box) = are_agent_states_changed(dict_prev_game,
                                        user_cur_game_data.game.get_env_info())

    select_latent = False
    if self._PROMPT_ON_CHANGE and (a1_hold_changed or a2_hold_changed):
      select_latent = True

    if self._AUTO_PROMPT:
      user_cur_game_data.flags.action_count += 1
      if user_cur_game_data.flags.action_count >= self._PROMPT_FREQ:
        select_latent = True

    user_cur_game_data.flags.select = select_latent

    # mental state update
    # possibly change the page to draw

  def _on_game_finished(self, user_game_data: pg.UserGameData):
    '''
    user_game_data: NOTE - values will be updated
    '''

    user_game_data.flags.done = True

    game = user_game_data.game
    user_id = user_game_data.user.userid

    # save trajectory
    file_name = get_file_name(user_game_data.save_path, user_id,
                              user_game_data.session_name)
    header = game.__class__.__name__ + "-" + user_game_data.session_name + "\n"
    header += "User ID: %s\n" % (str(user_id), )
    header += str(self._GAME_MAP)
    game.save_history(file_name, header)

    # update score
    user_game_data.score = game.current_step
    if self._IS_MOVERS:
      best_score = user_game_data.user.best_a
    else:
      best_score = user_game_data.user.best_b

    if best_score > user_game_data.score:
      user_game_data.user = User.query.filter_by(
          userid=user_game_data.user.userid).first()
      if self._IS_MOVERS:
        user_game_data.user.best_a = user_game_data.score
      else:
        user_game_data.user.best_b = user_game_data.score

      db.session.commit()

    # move to next page
    user_game_data.go_to_next_page()

  def _get_updated_drawing_objects(
      self,
      game_env,
      flags: pg.GameFlags,
      best_score: int = 9999,
      game_updated: bool = True) -> Mapping[str, co.DrawingObject]:
    dict_objs = {}
    if game_updated:
      for obj in self._game_scene(game_env, self._IS_MOVERS, flags, False):
        dict_objs[obj.name] = obj

      obj = self._get_score_obj(game_env["current_step"], best_score)
      dict_objs[obj.name] = obj

    for obj in self._game_overlay(game_env, flags, not flags.select):
      dict_objs[obj.name] = obj

    obj = self._get_instruction_objs(flags)[0]
    dict_objs[obj.name] = obj

    select_disable = not self._MANUAL_SELECTION or flags.select
    dis_status = self._get_action_btn_disabled(game_env, flags)
    action_btns = self._get_btn_actions(*dis_status,
                                        select_disable=select_disable)
    for obj in action_btns:
      dict_objs[obj.name] = obj

    return dict_objs

  def _get_button_commands(self, clicked_btn, flags: pg.GameFlags):
    return None

  def _get_animations(self, dict_prev_game: Mapping[str, Any],
                      dict_cur_game: Mapping[str, Any], is_movers: bool):
    (a1_pos_changed, a2_pos_changed, a1_hold_changed, a2_hold_changed, a1_box,
     a2_box) = are_agent_states_changed(dict_prev_game, dict_cur_game)

    unchanged_agents = []
    if not a1_pos_changed and not a1_hold_changed:
      unchanged_agents.append(0)
    if not a2_pos_changed and not a2_hold_changed:
      unchanged_agents.append(1)

    list_animations = []

    for agent_idx in unchanged_agents:
      obj_name = ""
      if agent_idx == 0:
        if a1_box < 0:
          obj_name = co.IMG_WOMAN if is_movers else co.IMG_MAN
        elif a1_box == a2_box:
          obj_name = co.IMG_BOTH_BOX
        else:
          obj_name = co.IMG_MAN_BAG
      else:
        if a2_box < 0:
          obj_name = co.IMG_ROBOT
        elif a1_box == a2_box:
          obj_name = co.IMG_BOTH_BOX
        else:
          obj_name = co.IMG_ROBOT_BAG

      amp = int(self.GAME_WIDTH / dict_cur_game["x_grid"] * 0.05)

      obj = {'type': 'vibrate', 'obj_name': obj_name, 'amplitude': amp}
      if obj not in list_animations:
        list_animations.append(obj)

    return list_animations

  def _get_action_btn_disabled(self, game_env, flags: pg.GameFlags):
    '''
    output order :
        btn_up, btn_down, btn_left, btn_right, btn_stay, btn_pickup, btn_drop
    '''
    if flags.select or flags.done:
      return True, True, True, True, True, True, True

    a1_latent = game_env["a1_latent"]
    if a1_latent is None:
      return False, False, False, False, False, True, True

    drop_ok = False
    pickup_ok = False
    num_drops = len(game_env["drops"])
    num_goals = len(game_env["goals"])
    a1_pos = game_env["a1_pos"]
    a1_box, _ = pg.get_holding_box_idx(game_env["box_states"], num_drops,
                                       num_goals)
    if a1_box >= 0:  # set drop action status
      if a1_latent[0] == 'origin' and a1_pos == game_env["boxes"][a1_box]:
        drop_ok = True
      else:
        for idx, coord in enumerate(game_env["goals"]):
          if a1_latent[0] == 'goal' and a1_latent[1] == idx and a1_pos == coord:
            drop_ok = True
            break
    else:  # set pickup action status
      for idx, bidx in enumerate(game_env["box_states"]):
        state = pg.conv_box_idx_2_state(bidx, num_drops, num_goals)
        coord = None
        if state[0] == pg.BoxState.Original:
          coord = game_env["boxes"][idx]
        elif state[0] == pg.BoxState.WithAgent2:
          coord = game_env["a2_pos"]

        if coord is not None:
          if a1_latent[0] == 'pickup' and a1_latent[
              1] == idx and a1_pos == coord:
            pickup_ok = True
            break

    return False, False, False, False, False, not pickup_ok, not drop_ok

  def action_event(self, user_game_data: pg.UserGameData, clicked_btn):
    '''
    user_game_data: NOTE - values will be updated
    '''
    game = user_game_data.game
    action = None
    if clicked_btn == co.BTN_LEFT:
      action = EventType.LEFT
    elif clicked_btn == co.BTN_RIGHT:
      action = EventType.RIGHT
    elif clicked_btn == co.BTN_UP:
      action = EventType.UP
    elif clicked_btn == co.BTN_DOWN:
      action = EventType.DOWN
    elif clicked_btn == co.BTN_STAY:
      action = EventType.STAY
    elif clicked_btn == co.BTN_PICK_UP:
      action = EventType.HOLD
    elif clicked_btn == co.BTN_DROP:
      action = EventType.UNHOLD

    # should not happen
    assert action is not None
    assert not game.is_finished()

    game.event_input(self._AGENT1, action, None)
    if user_game_data.flags.aligned_a2_action:
      game.event_input(self._AGENT2, action, None)

    # take actions
    map_agent2action = game.get_joint_action()
    game.take_a_step(map_agent2action)

    return (map_agent2action[self._AGENT1], map_agent2action[self._AGENT2],
            game.is_finished())
