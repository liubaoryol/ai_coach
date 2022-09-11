from typing import Mapping, Any
import copy
import web_experiment.exp_common.canvas_objects as co
from web_experiment.exp_common.page_base import Exp1UserData
from web_experiment.exp_common.page_exp1_game_base import BoxPushGamePageBase
from web_experiment.exp_common.page_rescue_game_base import RescueGamePageBase
from web_experiment.define import EDomainType


class UserDataReplay(Exp1UserData):
  TRAJECTORY = "trajectory"
  TRAJ_IDX = "traj_idx"

  def __init__(self, user) -> None:
    super().__init__(user)
    self.data[self.TRAJECTORY] = []
    self.data[self.TRAJ_IDX] = 0


class BoxPushReplayPage(BoxPushGamePageBase):
  def __init__(self, domain_type, partial_obs, game_map) -> None:
    super().__init__(domain_type, False, game_map, False, False, 0)
    self._PARTIAL_OBS = partial_obs

  def init_user_data(self, user_data: UserDataReplay):
    user_data.data[UserDataReplay.SELECT] = False
    user_data.data[UserDataReplay.PARTIAL_OBS] = self._PARTIAL_OBS
    user_data.data[UserDataReplay.SHOW_LATENT] = True

  def _get_game_env(self, user_data: UserDataReplay):
    traj_idx = user_data.data[UserDataReplay.TRAJ_IDX]
    dict_game = user_data.data[UserDataReplay.TRAJECTORY][traj_idx]
    game_env = copy.copy(dict_game)

    game_env['x_grid'] = self._GAME_MAP['x_grid']
    game_env['y_grid'] = self._GAME_MAP['y_grid']
    game_env['boxes'] = self._GAME_MAP['boxes']
    game_env['goals'] = self._GAME_MAP['goals']
    game_env['drops'] = self._GAME_MAP['drops']
    game_env['walls'] = self._GAME_MAP['walls']
    game_env['wall_dir'] = self._GAME_MAP['wall_dir']
    game_env['box_types'] = self._GAME_MAP['box_types']
    return game_env

  def get_updated_drawing_info(self,
                               user_data: UserDataReplay,
                               clicked_button: str = None,
                               data_to_compare: Mapping[str, Any] = None):
    dict_game = self._get_game_env(user_data)

    dict_objs = self.canvas_objects(dict_game, user_data)
    drawing_order = self.get_drawing_order(dict_game, user_data)
    dict_init_commands = {"clear": None}

    return dict_init_commands, dict_objs, drawing_order, None

  def _get_score_obj(self, user_data):
    dict_game = self._get_game_env(user_data)
    score = dict_game["current_step"]

    margin = 10
    text_score = "Time Taken: " + str(score) + "\n"
    return co.TextObject(
        self.TEXT_SCORE,
        (self.GAME_RIGHT + margin, int(co.CANVAS_HEIGHT * 0.9)),
        co.CANVAS_WIDTH - self.GAME_RIGHT - 2 * margin,
        24,
        text_score,
        text_align="right")

  def _get_instruction(self, user_game_data: Exp1UserData):
    return "Please review your actions and label your mental model at each step"

  def canvas_objects(self, dict_game, user_data):
    dict_objs = {}
    dict_objs[self.GAME_BORDER] = co.LineSegment(
        self.GAME_BORDER, (self.GAME_RIGHT, self.GAME_TOP),
        (self.GAME_RIGHT, self.GAME_BOTTOM))

    obj = self._get_instruction_objs(user_data)
    dict_objs[obj.name] = obj

    obj = self._get_score_obj(user_data)
    dict_objs[obj.name] = obj

    for obj in self._game_scene(dict_game, user_data, True):
      dict_objs[obj.name] = obj

    for obj in self._game_overlay(dict_game, user_data):
      dict_objs[obj.name] = obj

    return dict_objs

  def get_drawing_order(self, dict_game, user_data):
    drawing_order = []
    drawing_order.append(self.GAME_BORDER)

    drawing_order = (drawing_order +
                     self._game_scene_names(dict_game, user_data))
    drawing_order = (drawing_order +
                     self._game_overlay_names(dict_game, user_data))

    drawing_order.append(self.TEXT_SCORE)

    drawing_order.append(self.TEXT_INSTRUCTION)
    return drawing_order

  def button_clicked(self, user_data: UserDataReplay, clicked_btn: str):
    return super().button_clicked(user_data, clicked_btn)


class BoxPushReviewPage(BoxPushGamePageBase):
  def __init__(self, domain_type, partial_obs, game_map) -> None:
    super().__init__(domain_type, False, game_map, False, False, 0)
    self._PARTIAL_OBS = partial_obs

  def init_user_data(self, user_data: UserDataReplay):
    user_data.data[UserDataReplay.SELECT] = False
    user_data.data[UserDataReplay.PARTIAL_OBS] = self._PARTIAL_OBS
    user_data.data[UserDataReplay.SHOW_LATENT] = True

  def _get_game_env(self, user_data: UserDataReplay):
    traj_idx = user_data.data[UserDataReplay.TRAJ_IDX]
    dict_game = user_data.data[UserDataReplay.TRAJECTORY][traj_idx]
    game_env = copy.copy(dict_game)

    game_env['x_grid'] = self._GAME_MAP['x_grid']
    game_env['y_grid'] = self._GAME_MAP['y_grid']
    game_env['boxes'] = self._GAME_MAP['boxes']
    game_env['goals'] = self._GAME_MAP['goals']
    game_env['drops'] = self._GAME_MAP['drops']
    game_env['walls'] = self._GAME_MAP['walls']
    game_env['wall_dir'] = self._GAME_MAP['wall_dir']
    game_env['box_types'] = self._GAME_MAP['box_types']
    return game_env

  def _get_fix_destination(self, disable):
    ctrl_btn_w = int(self.GAME_WIDTH / 12)
    x_ctrl_cen = int(self.GAME_RIGHT + (co.CANVAS_WIDTH - self.GAME_RIGHT) / 2)
    y_ctrl_cen = int(co.CANVAS_HEIGHT * 0.65)
    font_size = 20
    btn_select = co.ButtonRect(co.BTN_SELECT,
                               (x_ctrl_cen, y_ctrl_cen + ctrl_btn_w * 2),
                               (ctrl_btn_w * 4, ctrl_btn_w),
                               font_size,
                               "Fix Destination",
                               disable=disable)
    return btn_select

  def get_updated_drawing_info(self,
                               user_data: UserDataReplay,
                               clicked_button: str = None,
                               data_to_compare: Mapping[str, Any] = None):
    dict_game = self._get_game_env(user_data)

    dict_objs = self.canvas_objects(dict_game, user_data)
    drawing_order = self.get_drawing_order(dict_game, user_data)
    dict_init_commands = {"clear": None}

    return dict_init_commands, dict_objs, drawing_order, None

  def _get_score_obj(self, user_data):
    dict_game = self._get_game_env(user_data)
    score = dict_game["current_step"]

    margin = 10
    text_score = "Time Taken: " + str(score) + "\n"
    return co.TextObject(
        self.TEXT_SCORE,
        (self.GAME_RIGHT + margin, int(co.CANVAS_HEIGHT * 0.9)),
        co.CANVAS_WIDTH - self.GAME_RIGHT - 2 * margin,
        24,
        text_score,
        text_align="right")

  def _get_instruction(self, user_game_data: Exp1UserData):
    return "Please review your actions and label your mental model at each step"

  def canvas_objects(self, dict_game, user_data: UserDataReplay):
    dict_objs = {}
    dict_objs[self.GAME_BORDER] = co.LineSegment(
        self.GAME_BORDER, (self.GAME_RIGHT, self.GAME_TOP),
        (self.GAME_RIGHT, self.GAME_BOTTOM))

    obj = self._get_instruction_objs(user_data)
    dict_objs[obj.name] = obj

    obj = self._get_score_obj(user_data)
    dict_objs[obj.name] = obj

    for obj in self._game_scene(dict_game, user_data, True):
      dict_objs[obj.name] = obj

    for obj in self._game_overlay(dict_game, user_data):
      dict_objs[obj.name] = obj

    max_idx = len(user_data.data[UserDataReplay.TRAJECTORY])
    fix_disable = (user_data.data[UserDataReplay.TRAJ_IDX] == max_idx - 1)
    obj = self._get_fix_destination(fix_disable)
    dict_objs[obj.name] = obj

    return dict_objs

  def get_drawing_order(self, dict_game, user_data):
    drawing_order = []
    drawing_order.append(self.GAME_BORDER)

    drawing_order = (drawing_order +
                     self._game_scene_names(dict_game, user_data))
    drawing_order = (drawing_order +
                     self._game_overlay_names(dict_game, user_data))

    drawing_order.append(self.TEXT_SCORE)

    drawing_order.append(self.TEXT_INSTRUCTION)
    drawing_order.append(co.BTN_SELECT)
    return drawing_order

  def button_clicked(self, user_data: UserDataReplay, clicked_btn: str):
    if clicked_btn == co.BTN_SELECT:
      user_data.data[Exp1UserData.SELECT] = True
      return
    elif self.is_sel_latent_btn(clicked_btn):
      latent = self.selbtn2latent(clicked_btn)
      if latent is not None:
        user_data.data[Exp1UserData.SELECT] = False
        traj_idx = user_data.data[UserDataReplay.TRAJ_IDX]
        dict_game = user_data.data[UserDataReplay.TRAJECTORY][traj_idx]
        dict_game["a1_latent"] = latent
        return

    return super().button_clicked(user_data, clicked_btn)


class RescueReplayPage(RescueGamePageBase):
  def __init__(self, partial_obs, game_map) -> None:
    super().__init__(False, game_map, False, False, 0)
    self._PARTIAL_OBS = partial_obs

  def init_user_data(self, user_data: UserDataReplay):
    user_data.data[UserDataReplay.SELECT] = False
    user_data.data[UserDataReplay.PARTIAL_OBS] = self._PARTIAL_OBS
    user_data.data[UserDataReplay.SHOW_LATENT] = True

  def _get_game_env(self, user_data: UserDataReplay):
    traj_idx = user_data.data[UserDataReplay.TRAJ_IDX]
    dict_game = user_data.data[UserDataReplay.TRAJECTORY][traj_idx]
    game_env = copy.copy(dict_game)

    game_env['places'] = self._GAME_MAP['places']
    game_env['routes'] = self._GAME_MAP['routes']
    game_env['connections'] = self._GAME_MAP['connections']
    game_env['work_locations'] = self._GAME_MAP['work_locations']
    game_env['work_info'] = self._GAME_MAP['work_info']
    return game_env

  def get_updated_drawing_info(self,
                               user_data: UserDataReplay,
                               clicked_button: str = None,
                               data_to_compare: Mapping[str, Any] = None):

    dict_game = self._get_game_env(user_data)

    dict_objs = self.canvas_objects(dict_game, user_data)
    drawing_order = self.get_drawing_order(dict_game, user_data)
    dict_init_commands = {"clear": None}

    return dict_init_commands, dict_objs, drawing_order, None

  def _get_score_obj(self, user_data):
    dict_game = self._get_game_env(user_data)
    score = dict_game["current_step"]

    margin = 10
    text_score = "Time Taken: " + str(score) + "\n"
    return co.TextObject(
        self.TEXT_SCORE,
        (self.GAME_RIGHT + margin, int(co.CANVAS_HEIGHT * 0.9)),
        co.CANVAS_WIDTH - self.GAME_RIGHT - 2 * margin,
        24,
        text_score,
        text_align="right")

  def _get_instruction(self, user_game_data: Exp1UserData):
    return "Please review your actions and label your mental model at each step"

  def canvas_objects(self, dict_game, user_data):
    dict_objs = {}
    dict_objs[self.GAME_BORDER] = co.LineSegment(
        self.GAME_BORDER, (self.GAME_RIGHT, self.GAME_TOP),
        (self.GAME_RIGHT, self.GAME_BOTTOM))

    obj = self._get_instruction_objs(user_data)
    dict_objs[obj.name] = obj

    obj = self._get_score_obj(user_data)
    dict_objs[obj.name] = obj

    for obj in self._game_scene(dict_game, user_data, True):
      dict_objs[obj.name] = obj

    for obj in self._game_overlay(dict_game, user_data):
      dict_objs[obj.name] = obj

    return dict_objs

  def get_drawing_order(self, dict_game, user_data):
    drawing_order = []
    drawing_order.append(self.GAME_BORDER)

    drawing_order = (drawing_order +
                     self._game_scene_names(dict_game, user_data))
    drawing_order = (drawing_order +
                     self._game_overlay_names(dict_game, user_data))

    drawing_order.append(self.TEXT_SCORE)

    drawing_order.append(self.TEXT_INSTRUCTION)
    return drawing_order

  def button_clicked(self, user_data: UserDataReplay, clicked_btn: str):
    return super().button_clicked(user_data, clicked_btn)


class RescueReviewPage(RescueGamePageBase):
  def __init__(self, partial_obs, game_map) -> None:
    super().__init__(False, game_map, False, False, 0)
    self._PARTIAL_OBS = partial_obs

  def init_user_data(self, user_data: UserDataReplay):
    user_data.data[UserDataReplay.SELECT] = False
    user_data.data[UserDataReplay.PARTIAL_OBS] = self._PARTIAL_OBS
    user_data.data[UserDataReplay.SHOW_LATENT] = True

  def _get_game_env(self, user_data: UserDataReplay):
    traj_idx = user_data.data[UserDataReplay.TRAJ_IDX]
    dict_game = user_data.data[UserDataReplay.TRAJECTORY][traj_idx]
    game_env = copy.copy(dict_game)

    game_env['places'] = self._GAME_MAP['places']
    game_env['routes'] = self._GAME_MAP['routes']
    game_env['connections'] = self._GAME_MAP['connections']
    game_env['work_locations'] = self._GAME_MAP['work_locations']
    game_env['work_info'] = self._GAME_MAP['work_info']
    return game_env

  def _get_fix_destination(self, disable):
    ctrl_btn_w = int(self.GAME_WIDTH / 12)
    x_ctrl_cen = int(self.GAME_RIGHT + (co.CANVAS_WIDTH - self.GAME_RIGHT) / 2)
    y_ctrl_cen = int(co.CANVAS_HEIGHT * 0.65)
    font_size = 20
    btn_select = co.ButtonRect(co.BTN_SELECT,
                               (x_ctrl_cen, y_ctrl_cen + ctrl_btn_w * 2),
                               (ctrl_btn_w * 4, ctrl_btn_w),
                               font_size,
                               "Fix Destination",
                               disable=disable)
    return btn_select

  def get_updated_drawing_info(self,
                               user_data: UserDataReplay,
                               clicked_button: str = None,
                               data_to_compare: Mapping[str, Any] = None):
    dict_game = self._get_game_env(user_data)

    dict_objs = self.canvas_objects(dict_game, user_data)
    drawing_order = self.get_drawing_order(dict_game, user_data)
    dict_init_commands = {"clear": None}

    return dict_init_commands, dict_objs, drawing_order, None

  def _get_score_obj(self, user_data):
    dict_game = self._get_game_env(user_data)
    score = dict_game["current_step"]

    margin = 10
    text_score = "Time Taken: " + str(score) + "\n"
    return co.TextObject(
        self.TEXT_SCORE,
        (self.GAME_RIGHT + margin, int(co.CANVAS_HEIGHT * 0.9)),
        co.CANVAS_WIDTH - self.GAME_RIGHT - 2 * margin,
        24,
        text_score,
        text_align="right")

  def _get_instruction(self, user_game_data: Exp1UserData):
    return "Please review your actions and label your mental model at each step"

  def canvas_objects(self, dict_game, user_data: UserDataReplay):
    dict_objs = {}
    dict_objs[self.GAME_BORDER] = co.LineSegment(
        self.GAME_BORDER, (self.GAME_RIGHT, self.GAME_TOP),
        (self.GAME_RIGHT, self.GAME_BOTTOM))

    obj = self._get_instruction_objs(user_data)
    dict_objs[obj.name] = obj

    obj = self._get_score_obj(user_data)
    dict_objs[obj.name] = obj

    for obj in self._game_scene(dict_game, user_data, True):
      dict_objs[obj.name] = obj

    for obj in self._game_overlay(dict_game, user_data):
      dict_objs[obj.name] = obj

    max_idx = len(user_data.data[UserDataReplay.TRAJECTORY])
    fix_disable = (user_data.data[UserDataReplay.TRAJ_IDX] == max_idx - 1)
    obj = self._get_fix_destination(fix_disable)
    dict_objs[obj.name] = obj

    return dict_objs

  def get_drawing_order(self, dict_game, user_data):
    drawing_order = []
    drawing_order.append(self.GAME_BORDER)

    drawing_order = (drawing_order +
                     self._game_scene_names(dict_game, user_data))
    drawing_order = (drawing_order +
                     self._game_overlay_names(dict_game, user_data))

    drawing_order.append(self.TEXT_SCORE)

    drawing_order.append(self.TEXT_INSTRUCTION)
    drawing_order.append(co.BTN_SELECT)
    return drawing_order

  def button_clicked(self, user_data: UserDataReplay, clicked_btn: str):
    if clicked_btn == co.BTN_SELECT:
      user_data.data[Exp1UserData.SELECT] = True
      return
    elif self.is_sel_latent_btn(clicked_btn):
      latent = self.selbtn2latent(clicked_btn)
      if latent is not None:
        user_data.data[Exp1UserData.SELECT] = False
        traj_idx = user_data.data[UserDataReplay.TRAJ_IDX]
        dict_game = user_data.data[UserDataReplay.TRAJECTORY][traj_idx]
        dict_game["a1_latent"] = latent
        return

    return super().button_clicked(user_data, clicked_btn)
