import abc
from typing import Mapping, Any
from aic_domain.simulator import Simulator
import web_experiment.exp_common.canvas_objects as co
from web_experiment.define import EDomainType


class UserData:
  '''
  user data that should be valid only during flask-socketio session
  '''
  PAGE_IDX = "cur_page_idx"
  NUM_PAGES = "num_pages"
  SESSION_NAME = "session_name"
  USER = "user"
  EXP_TYPE = "exp_type"
  GROUP_ID = "group_id"
  SESSION_DONE = "session_done"
  DRAW_OBJ_NAMES = "draw_obj_names"
  PAGE_DONE = "page_done"

  def __init__(self, user) -> None:
    self.data = {
        self.PAGE_IDX: 0,
        self.NUM_PAGES: 0,
        self.SESSION_NAME: "",
        self.USER: user,
        self.EXP_TYPE: "",
        self.SESSION_DONE: False,
        self.GROUP_ID: None,
        self.DRAW_OBJ_NAMES: set(),
        self.PAGE_DONE: False
    }

  def go_to_next_page(self):
    self.data[self.PAGE_DONE] = True
    cur_page_idx = self.data[self.PAGE_IDX]
    num_pages = self.data[self.NUM_PAGES]
    if cur_page_idx + 1 < num_pages:
      self.data[self.PAGE_IDX] += 1

  def go_to_prev_page(self):
    self.data[self.PAGE_DONE] = True
    cur_page_idx = self.data[self.PAGE_IDX]
    if cur_page_idx - 1 >= 0:
      self.data[self.PAGE_IDX] -= 1

  def get_data_to_compare(self) -> Mapping[str, Any]:
    return None


def get_objs_as_dictionary(drawing_objs: Mapping[str, co.DrawingObject]):
  return [obj.get_dictionary() for obj in drawing_objs.values()]


###############################################################################
# canvas page base
###############################################################################


class CanvasPageBase(abc.ABC):
  @abc.abstractmethod
  def __init__(self) -> None:
    pass

  @abc.abstractmethod
  def init_user_data(self, user_data: UserData):
    '''
    user_data: NOTE - values will be updated
    '''
    user_data.data[UserData.PAGE_DONE] = False
    pass

  @abc.abstractmethod
  def get_updated_drawing_info(self,
                               user_data: UserData,
                               clicked_button: str = None,
                               data_to_compare: Mapping[str, Any] = None):
    '''
    user_data: should NOT be changed here
    return: commands, drawing_objs, drawing_order, animations
    '''
    return None, None, None

  @abc.abstractmethod
  def get_drawing_order(self, user_data: UserData = None):
    return None

  @abc.abstractmethod
  def button_clicked(self, user_data: UserData, clicked_btn: str):
    '''
    user_data: NOTE - values will be updated
    '''
    pass


class CanvasPageError(CanvasPageBase):
  TEXT_ERROR = "text_error"

  def __init__(self, error_msg) -> None:
    super().__init__()
    self.ERROR_MSG = error_msg

  def init_user_data(self, user_data: UserData):
    return super().init_user_data(user_data)

  def get_updated_drawing_info(self,
                               user_data: UserData,
                               clicked_button: str = None,
                               data_to_compare: Mapping[str, Any] = None):
    '''
    user_data: should NOT be changed here
    return: commands, drawing_objs, drawing_order, animations
    '''
    drawing_obj = self._get_init_drawing_objects(user_data)
    commands = {"clear": None}
    return commands, drawing_obj, None

  def button_clicked(self, user_data: UserData, clicked_btn: str):
    return super().button_clicked(user_data, clicked_btn)

  def get_drawing_order(self, user_game_data=None):
    return [self.TEXT_ERROR]

  def _get_init_drawing_objects(self,
                                user_data) -> Mapping[str, co.DrawingObject]:
    font_size = 30
    obj = co.TextObject(self.TEXT_ERROR,
                        (0, int(co.CANVAS_HEIGHT / 2 - font_size)),
                        co.CANVAS_WIDTH,
                        font_size,
                        self.ERROR_MSG,
                        text_align="center",
                        text_baseline="middle")

    return {obj.name: obj}


class Exp1UserData(UserData):
  '''
  user data that should be valid only during flask-socketio session
  '''
  GAME = "game"  # key to game object
  SELECT = "select"  # latent state selection mode (overlay on)
  GAME_DONE = "game_done"
  ACTION_COUNT = "action_count"
  PARTIAL_OBS = "partial_obs"  # use partial observability or not
  SCORE = "score"
  SAVE_PATH = "save_path"
  USER_LABELS = "user_labels"
  USER_LABEL_PATH = "user_label_path"
  PREV_INFERENCE = "prev_inference"
  INTERVENTION = "intervention"
  INTERVENTION_HISTORY = "intervention_history"

  def __init__(self, user) -> None:
    super().__init__(user)
    self.data[Exp1UserData.GAME] = None
    self.data[Exp1UserData.SELECT] = False
    self.data[Exp1UserData.GAME_DONE] = False
    self.data[Exp1UserData.ACTION_COUNT] = 0
    self.data[Exp1UserData.PARTIAL_OBS] = True
    self.data[Exp1UserData.SCORE] = 0
    self.data[Exp1UserData.SAVE_PATH] = ""
    self.data[Exp1UserData.USER_LABELS] = []
    self.data[Exp1UserData.USER_LABEL_PATH] = ""
    self.data[Exp1UserData.PREV_INFERENCE] = None
    self.data[Exp1UserData.INTERVENTION] = None  # either None or latent index
    self.data[Exp1UserData.INTERVENTION_HISTORY] = []

  def get_game_ref(self) -> Simulator:
    return self.data[Exp1UserData.GAME]

  def set_game(self, game: Simulator):
    self.data[Exp1UserData.GAME] = game

  def get_data_to_compare(self) -> Mapping[str, Any]:
    game = self.get_game_ref()
    if game is None:
      return None
    else:
      return self.get_game_ref().get_env_info()


class ExperimentPageBase(CanvasPageBase):
  GAME_BORDER = "ls_line"
  TEXT_INSTRUCTION = "text_inst"
  TEXT_SCORE = "text_score"
  SPOTLIGHT = "Spotlight"

  GAME_LEFT = 0
  GAME_TOP = 0
  GAME_WIDTH = co.CANVAS_HEIGHT
  GAME_HEIGHT = co.CANVAS_HEIGHT
  GAME_RIGHT = GAME_LEFT + GAME_WIDTH
  GAME_BOTTOM = GAME_TOP + GAME_HEIGHT

  def __init__(self, show_border, show_instr, show_score,
               domain_type: EDomainType) -> None:
    self._SHOW_BORDER = show_border
    self._SHOW_INSTRUCTION = show_instr
    self._SHOW_SCORE = show_score
    self._DOMAIN_TYPE = domain_type

  def get_updated_drawing_info(self,
                               user_data: UserData,
                               clicked_button: str = None,
                               dict_prev_scene_data: Mapping[str, Any] = None):

    drawing_objs = self._get_init_drawing_objects(user_data)
    commands = self._get_init_commands(user_data)

    animations = None

    return commands, drawing_objs, animations

  @abc.abstractmethod
  def init_user_data(self, user_game_data: UserData):
    '''
    user_game_data: NOTE - values will be updated
    '''
    super().init_user_data(user_game_data)
    pass

  @abc.abstractmethod
  def button_clicked(self, user_game_data: UserData, clicked_btn: str):
    '''
    user_game_data: NOTE - values will be updated
    return: commands, drawing_objs, drawing_order, animations
      drawing info
    '''

    if clicked_btn == co.BTN_NEXT:
      user_game_data.go_to_next_page()
      return

    if clicked_btn == co.BTN_PREV:
      user_game_data.go_to_prev_page()
      return

  def _get_instruction(self, user_game_data: UserData):
    return ""

  def get_drawing_order(self, user_data: UserData = None):
    return self._get_drawing_order(user_data)

  @abc.abstractmethod
  def _get_drawing_order(self, user_game_data: UserData = None):
    drawing_order = []
    if self._SHOW_BORDER:
      drawing_order.append(self.GAME_BORDER)

    return drawing_order

  def _get_init_drawing_objects(
      self, user_data: UserData) -> Mapping[str, co.DrawingObject]:

    dict_objs = {}
    if self._SHOW_BORDER:
      dict_objs[self.GAME_BORDER] = co.LineSegment(
          self.GAME_BORDER, (self.GAME_RIGHT, self.GAME_TOP),
          (self.GAME_RIGHT, self.GAME_BOTTOM))

    if self._SHOW_INSTRUCTION:
      obj = self._get_instruction_objs(user_data)
      dict_objs[obj.name] = obj

    if self._SHOW_SCORE:
      obj = self._get_score_obj(user_data)
      dict_objs[obj.name] = obj

    return dict_objs

  def _get_init_commands(self, user_data: UserData):
    return {"clear": None}

  def _get_btn_prev_next(self, prev_disable=True, next_disable=True):
    margin = 10
    font_size = 18
    width = int((co.CANVAS_WIDTH - self.GAME_RIGHT) / 4)
    height = int(width * 0.5)
    pos_next = (int(co.CANVAS_WIDTH - width * 0.5 - margin),
                int(co.CANVAS_HEIGHT * 0.5 - 0.5 * height - margin))
    pos_prev = (int(self.GAME_RIGHT + width * 0.5 + margin),
                int(co.CANVAS_HEIGHT * 0.5 - 0.5 * height - margin))
    size = (width, height)
    btn_next = co.ButtonRect(co.BTN_NEXT,
                             pos_next,
                             size,
                             font_size,
                             "Next",
                             disable=next_disable)
    btn_prev = co.ButtonRect(co.BTN_PREV,
                             pos_prev,
                             size,
                             font_size,
                             "Prev",
                             disable=prev_disable)
    return btn_prev, btn_next

  def _get_btn_start(self, start_disable=True):
    start_btn_width = int(self.GAME_WIDTH / 3)
    start_btn_height = int(self.GAME_HEIGHT / 10)
    x_cen = int(self.GAME_LEFT + self.GAME_WIDTH / 2)
    y_cen = int(self.GAME_TOP + self.GAME_HEIGHT / 2)

    start_btn_obj = co.ButtonRect(co.BTN_START, (x_cen, y_cen),
                                  (start_btn_width, start_btn_height),
                                  30,
                                  "Start",
                                  disable=start_disable)
    return start_btn_obj

  def _get_spotlight(self, x_cen, y_cen, radius):
    outer_ltwh = (0, 0, co.CANVAS_WIDTH, co.CANVAS_HEIGHT)

    margin = 5
    pos = (self.GAME_RIGHT + margin, margin)
    size = (co.CANVAS_WIDTH - pos[0] - margin, int(self.GAME_HEIGHT * 0.5))
    return co.ClippedRectangle(self.SPOTLIGHT,
                               outer_ltwh,
                               list_circle=[(x_cen, y_cen, radius)],
                               list_rect=[(*pos, *size)])

  def _get_instruction_objs(self, user_data: UserData):
    margin = 10
    pos = (self.GAME_RIGHT + margin, margin)
    width = co.CANVAS_WIDTH - pos[0] - margin
    text_instr = co.TextObject(self.TEXT_INSTRUCTION, pos, width, 18,
                               self._get_instruction(user_data))

    return text_instr

  def _get_score_text(self, user_data: Exp1UserData):
    return "Time Taken: 0"

  def _get_score_obj(self, user_data: Exp1UserData):
    margin = 10
    text_score = self._get_score_text(user_data)
    num_line = len(text_score.split('\n'))
    font_size = 20
    return co.TextObject(
        self.TEXT_SCORE,
        (self.GAME_RIGHT + margin,
         int(co.CANVAS_HEIGHT - num_line * font_size * 1.1 - margin)),
        co.CANVAS_WIDTH - self.GAME_RIGHT - 2 * margin,
        font_size,
        text_score,
        text_align="right")
