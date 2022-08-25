from typing import Mapping
import web_experiment.experiment1.page_base as pg
import web_experiment.experiment1.canvas_objects as co
from web_experiment.models import User, db


class CanvasPageStart(pg.CanvasPageBase):
  def __init__(self, is_movers) -> None:
    super().__init__(True, True, True, False, is_movers)

  def _init_user_data(self, user_game_data: pg.UserGameData):
    '''
    user_game_data: NOTE - values will be updated
    '''
    user_game_data.flags.done = False
    user_game_data.flags.select = False

  def button_clicked(self, user_game_data: pg.UserGameData, clicked_btn):
    '''
    user_game_data: NOTE - values will be updated
    return: commands, drawing_objs, drawing_order, animations
      drawing info
    '''

    if clicked_btn == co.BTN_START:
      user_game_data.go_to_next_page()
      return None, None, None, None

    return super().button_clicked(user_game_data, clicked_btn)

  def _get_instruction(self, flags: pg.GameFlags):
    return "Click the \"Start\" button to begin the task."

  def _get_drawing_order(self, game_env=None, flags: pg.GameFlags = None):
    drawing_order = [self.GAME_BORDER]

    drawing_order.append(co.BTN_START)

    drawing_order = drawing_order + co.ACTION_BUTTONS
    drawing_order.append(co.BTN_SELECT)

    drawing_order.append(self.TEXT_SCORE)
    drawing_order.append(self.RECT_INSTRUCTION)
    drawing_order.append(self.TEXT_INSTRUCTION)

    return drawing_order

  def _get_init_drawing_objects(
      self,
      game_env=None,
      flags: pg.GameFlags = None,
      score: int = 0,
      best_score: int = 9999) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(game_env, flags, score,
                                                  best_score)
    objs = self._get_btn_actions(True, True, True, True, True, True, True, True)
    for obj in objs:
      dict_objs[obj.name] = obj

    start_btn_obj = self._get_btn_start(False)
    dict_objs[start_btn_obj.name] = start_btn_obj

    return dict_objs


class CanvasPageWarning(pg.CanvasPageBase):
  TEXT_WARNING = "text_warning"
  BTN_REAL_START = "btn_real_start"

  def __init__(self, is_movers) -> None:
    super().__init__(True, False, True, False, is_movers)

  def _init_user_data(self, user_game_data: pg.UserGameData):
    '''
    user_game_data: NOTE - values will be updated
    '''
    user_game_data.flags.done = False
    user_game_data.flags.select = False

  def button_clicked(self, user_game_data: pg.UserGameData, clicked_btn):
    '''
    user_game_data: NOTE - values will be updated
    return: commands, drawing_objs, drawing_order, animations
      drawing info
    '''

    if clicked_btn == self.BTN_REAL_START:
      user_game_data.go_to_next_page()
      return None, None, None, None

    return super().button_clicked(user_game_data, clicked_btn)

  def _get_drawing_order(self, game_env=None, flags: pg.GameFlags = None):
    drawing_order = [self.GAME_BORDER]

    drawing_order.append(self.BTN_REAL_START)
    drawing_order.append(self.TEXT_WARNING)

    drawing_order = drawing_order + co.ACTION_BUTTONS
    drawing_order.append(co.BTN_SELECT)

    drawing_order.append(self.TEXT_SCORE)

    return drawing_order

  def _get_init_drawing_objects(
      self,
      game_env=None,
      flags: pg.GameFlags = None,
      score: int = 0,
      best_score: int = 9999) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(game_env, flags, score,
                                                  best_score)
    objs = self._get_btn_actions(True, True, True, True, True, True, True, True)
    for obj in objs:
      dict_objs[obj.name] = obj

    text = ("Please review the instructions for this session listed above. " +
            "When you are ready, press next to begin.")
    font_size = 30
    obj = co.TextObject(
        self.TEXT_WARNING,
        (self.GAME_LEFT, int(self.GAME_TOP + self.GAME_HEIGHT / 3 - font_size)),
        self.GAME_WIDTH,
        font_size,
        text,
        text_align="center",
        text_baseline="middle")
    dict_objs[obj.name] = obj

    obj = co.ButtonRect(self.BTN_REAL_START,
                        (int(self.GAME_LEFT + self.GAME_WIDTH / 2),
                         int(self.GAME_TOP + self.GAME_HEIGHT * 0.6)),
                        (100, 50), 20, "Next")
    dict_objs[obj.name] = obj

    return dict_objs


class CanvasPageEnd(pg.CanvasPageBase):
  TEXT_END = "text_end"

  def __init__(self, is_movers) -> None:
    super().__init__(False, False, True, False, is_movers)

  def _init_user_data(self, user_game_data: pg.UserGameData):
    '''
    user_game_data: NOTE - values will be updated
    '''
    user_game_data.flags.done = False
    user_game_data.flags.select = False
    if not getattr(user_game_data.user, user_game_data.session_name):
      user_game_data.user = User.query.filter_by(
          userid=user_game_data.user.userid).first()
      setattr(user_game_data.user, user_game_data.session_name, True)
      db.session.commit()

  def button_clicked(self, user_game_data: pg.UserGameData, clicked_btn):
    '''
    user_game_data: NOTE - values will be updated
    return: commands, drawing_objs, drawing_order, animations
      drawing info
    '''
    return None, None, None, None

  def _get_drawing_order(self, game_env=None, flags: pg.GameFlags = None):
    drawing_order = [self.GAME_BORDER]
    drawing_order.append(self.TEXT_END)
    drawing_order.append(self.TEXT_SCORE)

    return drawing_order

  def _get_init_drawing_objects(
      self,
      game_env=None,
      flags: pg.GameFlags = None,
      score: int = 0,
      best_score: int = 9999) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(game_env, flags, score,
                                                  best_score)

    text = ("This session is now complete. " +
            "Please proceed to the survey using the button below.")
    font_size = 30
    obj = co.TextObject(self.TEXT_END,
                        (0, int(co.CANVAS_HEIGHT / 2 - font_size)),
                        co.CANVAS_WIDTH,
                        font_size,
                        text,
                        text_align="center",
                        text_baseline="middle")
    dict_objs[obj.name] = obj

    return dict_objs
