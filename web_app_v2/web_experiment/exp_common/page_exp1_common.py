from typing import Mapping
from web_experiment.define import ExpType, EDomainType
import web_experiment.exp_common.canvas_objects as co
from web_experiment.exp_common.page_base import ExperimentPageBase, Exp1UserData
from web_experiment.models import db, ExpIntervention, ExpDataCollection


class CanvasPageStart(ExperimentPageBase):
  def __init__(self, domain_type: EDomainType) -> None:
    super().__init__(True, True, True, domain_type)

  def init_user_data(self, user_game_data: Exp1UserData):
    return super().init_user_data(user_game_data)

  def button_clicked(self, user_game_data: Exp1UserData, clicked_btn: str):
    '''
    user_game_data: NOTE - values will be updated
    return: commands, drawing_objs, drawing_order, animations
      drawing info
    '''
    if clicked_btn == co.BTN_START:
      user_game_data.go_to_next_page()
      return

    return super().button_clicked(user_game_data, clicked_btn)

  def _get_instruction(self, user_game_data: Exp1UserData):
    return "Click the \"Start\" button to begin the task."

  def _get_drawing_order(self, user_game_data: Exp1UserData = None):
    drawing_order = super()._get_drawing_order(user_game_data)

    drawing_order.append(co.BTN_START)

    drawing_order.append(self.TEXT_SCORE)
    drawing_order.append(self.TEXT_INSTRUCTION)

    return drawing_order

  def _get_init_drawing_objects(
      self, user_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_data)

    start_btn_obj = self._get_btn_start(False)
    dict_objs[start_btn_obj.name] = start_btn_obj

    return dict_objs


class CanvasPageWarning(ExperimentPageBase):
  TEXT_WARNING = "text_warning"
  BTN_REAL_START = "btn_real_start"

  def __init__(self, domain_type: EDomainType) -> None:
    super().__init__(True, True, True, domain_type)

  def init_user_data(self, user_game_data: Exp1UserData):
    return super().init_user_data(user_game_data)

  def button_clicked(self, user_game_data: Exp1UserData, clicked_btn: str):

    if clicked_btn == self.BTN_REAL_START:
      user_game_data.go_to_next_page()
      return

    return super().button_clicked(user_game_data, clicked_btn)

  def _get_drawing_order(self, user_game_data: Exp1UserData = None):
    drawing_order = super()._get_drawing_order(user_game_data)

    drawing_order.append(self.BTN_REAL_START)
    drawing_order.append(self.TEXT_WARNING)

    drawing_order.append(self.TEXT_SCORE)

    return drawing_order

  def _get_init_drawing_objects(
      self, user_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_data)

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


class CanvasPageEnd(ExperimentPageBase):
  TEXT_END = "text_end"

  def __init__(self, domain_type) -> None:
    super().__init__(False, False, True, domain_type)

  def init_user_data(self, user_game_data: Exp1UserData):
    user_game_data.data[Exp1UserData.PAGE_DONE] = False

    user = user_game_data.data[Exp1UserData.USER]
    user_id = user.userid
    session_name = user_game_data.data[Exp1UserData.SESSION_NAME]
    if user_game_data.data[Exp1UserData.EXP_TYPE] == ExpType.Data_collection:
      exp = ExpDataCollection.query.filter_by(subject_id=user_id).first()
    elif user_game_data.data[Exp1UserData.EXP_TYPE] == ExpType.Intervention:
      exp = ExpIntervention.query.filter_by(subject_id=user_id).first()

    if not getattr(exp, session_name):
      setattr(exp, session_name, True)
      db.session.commit()
      user_game_data.data[Exp1UserData.SESSION_DONE] = True

  def button_clicked(self, user_game_data: Exp1UserData, clicked_btn: str):
    return

  def _get_drawing_order(self, user_game_data: Exp1UserData = None):
    drawing_order = super()._get_drawing_order(user_game_data)

    drawing_order.append(self.TEXT_END)
    drawing_order.append(self.TEXT_SCORE)

    return drawing_order

  def _get_init_drawing_objects(
      self, user_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_data)

    text = ("This session is now complete. " +
            "Please proceed using the button below.")
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


class CanvasPageTutorialStart(ExperimentPageBase):
  BTN_TUTORIAL_START = "btn_tutorial_start"

  def __init__(self, domain_type) -> None:
    super().__init__(False, False, False, domain_type)

  def init_user_data(self, user_game_data: Exp1UserData):
    return super().init_user_data(user_game_data)

  def _get_init_drawing_objects(
      self, user_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_data)

    pos = (int(co.CANVAS_WIDTH / 2), int(co.CANVAS_HEIGHT / 2))
    size = (int(self.GAME_WIDTH / 2), int(self.GAME_HEIGHT / 5))
    obj = co.ButtonRect(self.BTN_TUTORIAL_START, pos, size, 30,
                        "Interactive Tutorial\n(Click to Start)")
    dict_objs[obj.name] = obj

    return dict_objs

  def _get_drawing_order(self, user_game_data: Exp1UserData = None):
    drawing_order = super()._get_drawing_order(user_game_data)
    drawing_order.append(self.BTN_TUTORIAL_START)
    return drawing_order

  def button_clicked(self, user_game_data: Exp1UserData, clicked_btn: str):
    if clicked_btn == self.BTN_TUTORIAL_START:
      user_game_data.go_to_next_page()
      return

    return super().button_clicked(user_game_data, clicked_btn)


class CanvasPageInstruction(CanvasPageStart):
  def _get_init_drawing_objects(
      self, user_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_data)

    obj = dict_objs[co.BTN_START]  # type: co.ButtonRect
    obj.disable = True  # disable start btn

    obj_inst = dict_objs[self.TEXT_INSTRUCTION]  # type: co.TextObject
    x_cen = int(obj_inst.pos[0] + obj_inst.width * 0.5)
    y_cen = int(self.GAME_HEIGHT / 5)
    radius = y_cen * 0.1

    obj = self._get_spotlight(x_cen, y_cen, radius)
    dict_objs[obj.name] = obj

    objs = self._get_btn_prev_next(False, False)
    for obj in objs:
      dict_objs[obj.name] = obj

    return dict_objs

  def _get_drawing_order(self, user_game_data: Exp1UserData = None):
    drawing_order = super()._get_drawing_order(user_game_data)

    drawing_order.append(self.SPOTLIGHT)
    drawing_order.append(co.BTN_PREV)
    drawing_order.append(co.BTN_NEXT)

    return drawing_order

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("Prompts will be shown here. Please read each prompt carefully. " +
            "Click the “Next” button to proceed and “Back” button to " +
            "go to the previous prompt.")


class CanvasPageTutorialGameStart(CanvasPageStart):
  def _get_init_drawing_objects(
      self, user_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_data)
    obj = dict_objs[co.BTN_START]  # type: co.ButtonRect
    obj.disable = False  # enable start btn

    objs = self._get_btn_prev_next(False, True)
    for obj in objs:
      dict_objs[obj.name] = obj

    return dict_objs

  def _get_drawing_order(self, user_game_data: Exp1UserData = None):
    drawing_order = super()._get_drawing_order(user_game_data)

    drawing_order.append(co.BTN_PREV)
    drawing_order.append(co.BTN_NEXT)

    return drawing_order

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("At the start of each task, " +
            "you will see the screen shown on the left. " +
            "Click the “Start” button to begin the task.")
