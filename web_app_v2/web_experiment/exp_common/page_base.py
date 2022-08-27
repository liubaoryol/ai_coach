import abc
from typing import Mapping, Any
import web_experiment.exp_common.canvas_objects as co


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
