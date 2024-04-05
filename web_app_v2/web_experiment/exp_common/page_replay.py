from typing import Mapping, Any, Optional
import copy
import web_experiment.exp_common.canvas_objects as co
from web_experiment.exp_common.page_base import Exp1UserData
from web_experiment.exp_common.page_exp1_game_base import BoxPushGamePageBase
from web_experiment.exp_common.page_rescue_base import RescueGamePageBase


class UserDataReplay(Exp1UserData):
  TRAJECTORY = "trajectory"
  TRAJ_IDX = "traj_idx"
  USER_FIX = "user_fix"
  LATENT_COLLECTED = "latent_collected"
  LATENT_PREDICTED = "latent_predicted"

  def __init__(self, user=None) -> None:
    super().__init__(user)
    self.data[self.TRAJECTORY] = []
    self.data[self.TRAJ_IDX] = 0
    self.data[self.USER_FIX] = None  # type: Optional[Mapping]
    self.data[self.LATENT_COLLECTED] = None
    self.data[self.LATENT_PREDICTED] = None


class MixinCanvasPageReplay:
  '''
  this mixin is intentionally implemented to hiject some virtual methods 
                                                        (init_user_data, ...)
  do not define __init__ so that this mixin does not hiject __init__ method
  '''
  def set_flags(self, partial_obs):
    self._PARTIAL_OBS = partial_obs
    self._MANUAL_SELECTION = False
    self._AUTO_PROMPT = False
    self._PROMPT_ON_CHANGE = False
    self._PROMPT_FREQ = 0

  def init_user_data(self, user_data: UserDataReplay):
    user_data.data[UserDataReplay.PAGE_DONE] = False
    user_data.data[UserDataReplay.SELECT] = False
    user_data.data[UserDataReplay.PARTIAL_OBS] = self._PARTIAL_OBS

  def get_updated_drawing_info(self,
                               user_data: UserDataReplay,
                               clicked_button: str = None,
                               data_to_compare: Mapping[str, Any] = None):
    dict_game = self._get_game_env(user_data)

    dict_objs = self.canvas_objects(dict_game, user_data)
    dict_init_commands = {"clear": None}

    return dict_init_commands, dict_objs, None

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

  def get_drawing_order(self, user_data):
    dict_game = self._get_game_env(user_data)
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
    return


class MixinCanvasPageReview(MixinCanvasPageReplay):
  '''
  this mixin is intentionally implemented to hiject some virtual methods 
                                                        (canvas_objects, ...)
  do not define __init__ so that this mixin does not hiject __init__ method
  '''
  def _get_fix_destination(self, disable_fix, disable_left, disable_right):
    font_size = 20

    x_ctrl_cen = int(self.GAME_RIGHT + (co.CANVAS_WIDTH - self.GAME_RIGHT) / 2)
    y_ctrl_cen = int(co.CANVAS_HEIGHT * 0.8)
    ctrl_btn_h = int(self.GAME_HEIGHT / 12)

    btn_select = co.ButtonRect(co.BTN_SELECT, (x_ctrl_cen, y_ctrl_cen),
                               (ctrl_btn_h * 4, ctrl_btn_h),
                               font_size,
                               "Fix Destination",
                               disable=disable_fix)

    y_ctrl_cen_arrow = int(co.CANVAS_HEIGHT * 0.6)
    arrow_btn_width = ctrl_btn_h * 2
    x_offset = int((arrow_btn_width + self.GAME_WIDTH / 30) / 2)

    btn_left = co.ButtonRect(co.BTN_LEFT,
                             (x_ctrl_cen - x_offset, y_ctrl_cen_arrow),
                             (arrow_btn_width, ctrl_btn_h),
                             font_size,
                             "LEFT",
                             disable=disable_left)
    btn_right = co.ButtonRect(co.BTN_RIGHT,
                              (x_ctrl_cen + x_offset, y_ctrl_cen_arrow),
                              (arrow_btn_width, ctrl_btn_h),
                              font_size,
                              "RIGHT",
                              disable=disable_right)
    return btn_select, btn_left, btn_right

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("You can use the LEFT and RIGHT buttons below to view your " +
            "task progress and marked destinations. " +
            "You can also use the SCROLL BAR above. " +
            "If you marked your destination incorrectly, please fix it using " +
            "the \"Fix Destination\" button below.")

  def canvas_objects(self, dict_game, user_data: UserDataReplay):
    dict_objs = super().canvas_objects(dict_game, user_data)

    max_idx = len(user_data.data[UserDataReplay.TRAJECTORY])
    fix_disable = (user_data.data[UserDataReplay.TRAJ_IDX] == max_idx - 1)
    disable_left = (user_data.data[UserDataReplay.TRAJ_IDX] == 0)

    btn_select, btn_left, btn_right = self._get_fix_destination(
        fix_disable, disable_left, fix_disable)
    dict_objs[btn_select.name] = btn_select
    dict_objs[btn_left.name] = btn_left
    dict_objs[btn_right.name] = btn_right

    return dict_objs

  def get_drawing_order(self, user_data):
    dict_game = self._get_game_env(user_data)
    drawing_order = []
    drawing_order.append(self.GAME_BORDER)

    drawing_order = (drawing_order +
                     self._game_scene_names(dict_game, user_data))
    drawing_order = (drawing_order +
                     self._game_overlay_names(dict_game, user_data))

    drawing_order.append(self.TEXT_SCORE)

    drawing_order.append(self.TEXT_INSTRUCTION)
    drawing_order.append(co.BTN_SELECT)
    drawing_order.append(co.BTN_LEFT)
    drawing_order.append(co.BTN_RIGHT)
    return drawing_order

  def button_clicked(self, user_data: UserDataReplay, clicked_btn: str):
    if clicked_btn == co.BTN_SELECT:
      user_data.data[UserDataReplay.SELECT] = (
          not user_data.data[UserDataReplay.SELECT])
      return
    elif self.is_sel_latent_btn(clicked_btn):
      latent = self.selbtn2latent(clicked_btn)
      if latent is not None:
        user_data.data[UserDataReplay.SELECT] = False
        traj_idx = user_data.data[UserDataReplay.TRAJ_IDX]
        dict_game = user_data.data[UserDataReplay.TRAJECTORY][traj_idx]
        dict_game["a1_latent"] = latent
        user_data.data[UserDataReplay.USER_FIX][traj_idx] = latent
        return
    elif clicked_btn == co.BTN_LEFT:
      new_idx = max(user_data.data[UserDataReplay.TRAJ_IDX] - 1, 0)
      user_data.data[UserDataReplay.TRAJ_IDX] = new_idx
      return
    elif clicked_btn == co.BTN_RIGHT:
      max_idx = len(user_data.data[UserDataReplay.TRAJECTORY])
      new_idx = min(user_data.data[UserDataReplay.TRAJ_IDX] + 1, max_idx - 1)
      user_data.data[UserDataReplay.TRAJ_IDX] = new_idx
      return


class MixinBoxPushGameEnv:
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

  def _get_score_text(self, user_data):
    max_len = len(user_data.data[UserDataReplay.TRAJECTORY]) - 1
    dict_game = self._get_game_env(user_data)
    score = dict_game["current_step"]
    text = "Time Step: " + str(score) + " / " + str(max_len)
    return text


class MixinRescueGameEnv:
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

  def _get_score_text(self, user_data):
    max_len = len(user_data.data[UserDataReplay.TRAJECTORY]) - 1
    dict_game = self._get_game_env(user_data)

    score = dict_game["score"]
    step = dict_game["current_step"]

    text = "Time Step: " + str(step) + " / " + str(max_len) + "\n"
    text += "Score: " + str(score)

    return text


# inheritance order matters
class BoxPushReplayPage(MixinCanvasPageReplay, MixinBoxPushGameEnv,
                        BoxPushGamePageBase):
  def __init__(self, domain_type, partial_obs, game_map) -> None:
    super().__init__(domain_type, game_map, True)
    self.set_flags(partial_obs)


# inheritance order matters
class BoxPushReviewPage(MixinCanvasPageReview, MixinBoxPushGameEnv,
                        BoxPushGamePageBase):
  def __init__(self, domain_type, partial_obs, game_map) -> None:
    super().__init__(domain_type, game_map, True)
    self.set_flags(partial_obs)


# inheritance order matters
class RescueReplayPage(MixinCanvasPageReplay, MixinRescueGameEnv,
                       RescueGamePageBase):
  def __init__(self, partial_obs, game_map) -> None:
    super().__init__(game_map, True)
    self.set_flags(partial_obs)


# inheritance order matters
class RescueReviewPage(MixinCanvasPageReview, MixinRescueGameEnv,
                       RescueGamePageBase):
  def __init__(self, partial_obs, game_map) -> None:
    super().__init__(game_map, True)
    self.set_flags(partial_obs)
