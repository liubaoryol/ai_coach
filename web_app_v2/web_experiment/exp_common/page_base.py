import abc
from typing import Mapping, Any
from ai_coach_domain.simulator import Simulator
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
  SESSION_DONE = "session_done"

  def __init__(self, user) -> None:
    self.data = {
        self.PAGE_IDX: 0,
        self.NUM_PAGES: 0,
        self.SESSION_NAME: "",
        self.USER: user,
        self.EXP_TYPE: "",
        self.SESSION_DONE: False
    }

  def go_to_next_page(self):
    cur_page_idx = self.data[self.PAGE_IDX]
    num_pages = self.data[self.NUM_PAGES]
    if cur_page_idx + 1 < num_pages:
      self.data[self.PAGE_IDX] += 1

  def go_to_prev_page(self):
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
    return None, None, None, None

  @abc.abstractmethod
  def button_clicked(self, user_data: UserData, clicked_btn: str):
    '''
    user_data: NOTE - values will be updated
    '''
    pass


class Exp1UserData(UserData):
  '''
  user data that should be valid only during flask-socketio session
  '''
  GAME = "game"
  SELECT = "select"
  GAME_DONE = "game_done"
  ACTION_COUNT = "action_count"
  PARTIAL_OBS = "partial_obs"
  SCORE = "score"
  SAVE_PATH = "save_path"
  SHOW_LATENT = "show_latent"
  COLLECT_LATENT = "collect_latent"

  def __init__(self, user) -> None:
    super().__init__(user)
    self.data[Exp1UserData.GAME] = None
    self.data[Exp1UserData.SELECT] = False
    self.data[Exp1UserData.GAME_DONE] = False
    self.data[Exp1UserData.ACTION_COUNT] = 0
    self.data[Exp1UserData.PARTIAL_OBS] = True
    self.data[Exp1UserData.SCORE] = 0
    self.data[Exp1UserData.SAVE_PATH] = ""
    self.data[Exp1UserData.SHOW_LATENT] = False
    self.data[Exp1UserData.COLLECT_LATENT] = True

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
  RECT_INSTRUCTION = "rect_inst"
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
    drawing_order = self._get_drawing_order(user_data)
    commands = self._get_init_commands(user_data)

    animations = None

    return commands, drawing_objs, drawing_order, animations

  @abc.abstractmethod
  def init_user_data(self, user_game_data: UserData):
    '''
    user_game_data: NOTE - values will be updated
    '''
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
      for obj in self._get_instruction_objs(user_data):
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
    return co.CircleSpotlight(self.SPOTLIGHT, outer_ltwh, (x_cen, y_cen),
                              radius)

  def _get_instruction_objs(self, user_data: UserData):
    margin = 10
    pos = (self.GAME_RIGHT + margin, margin)
    width = co.CANVAS_WIDTH - pos[0] - margin
    text_instr = co.TextObject(self.TEXT_INSTRUCTION, pos, width, 18,
                               self._get_instruction(user_data))

    margin = 5
    pos = (self.GAME_RIGHT + margin, margin)
    size = (co.CANVAS_WIDTH - pos[0] - margin, int(self.GAME_HEIGHT * 0.5))
    rect_instr = co.Rectangle(self.RECT_INSTRUCTION, pos, size, "white")

    return text_instr, rect_instr

  def _get_score_obj(self, user_data: Exp1UserData):
    if self._DOMAIN_TYPE in [EDomainType.Movers, EDomainType.Cleanup]:
      score = user_data.data[Exp1UserData.SCORE]
      if self._DOMAIN_TYPE == EDomainType.Movers:
        best_score = user_data.data[Exp1UserData.USER].best_a
      else:
        best_score = user_data.data[Exp1UserData.USER].best_b

      margin = 10
      text_score = "Time Taken: " + str(score) + "\n"
      if best_score == 999:
        text_score += "(Your Best: - )"
      else:
        text_score += "(Your Best: " + str(best_score) + ")"
      return co.TextObject(
          self.TEXT_SCORE,
          (self.GAME_RIGHT + margin, int(co.CANVAS_HEIGHT * 0.9)),
          co.CANVAS_WIDTH - self.GAME_RIGHT - 2 * margin,
          24,
          text_score,
          text_align="right")
    elif self._DOMAIN_TYPE == EDomainType.Rescue:
      score = user_data.data[Exp1UserData.SCORE]
      # if self._IS_MOVERS:
      #   best_score = user_data.data[RescueUserData.USER].best_a
      # else:
      #   best_score = user_data.data[RescueUserData.USER].best_b
      best_score = 999

      margin = 10
      text_score = "Score: " + str(score) + "\n"
      if best_score == 999:
        text_score += "(Your Best: - )"
      else:
        text_score += "(Your Best: " + str(best_score) + ")"
      return co.TextObject(
          self.TEXT_SCORE,
          (self.GAME_RIGHT + margin, int(co.CANVAS_HEIGHT * 0.9)),
          co.CANVAS_WIDTH - self.GAME_RIGHT - 2 * margin,
          24,
          text_score,
          text_align="right")
    else:
      raise ValueError(f"{self._DOMAIN_TYPE} is not implemented")
