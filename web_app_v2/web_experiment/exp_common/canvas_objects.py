from typing import Sequence, Tuple, Optional
from ai_coach_domain.rescue import PlaceName

CANVAS_WIDTH = 900
CANVAS_HEIGHT = 600

BTN_UP = "Up"
BTN_DOWN = "Down"
BTN_LEFT = "Left"
BTN_RIGHT = "Right"
BTN_STAY = "Stay"
BTN_PICK_UP = "Pick Up"
BTN_DROP = "Drop"
BTN_NEXT = "Next"
BTN_PREV = "Prev"
BTN_SELECT = "Select Destination"
BTN_START = "Start"

ACTION_BUTTONS = [
    BTN_UP, BTN_DOWN, BTN_LEFT, BTN_RIGHT, BTN_STAY, BTN_PICK_UP, BTN_DROP
]

JOYSTICK_BUTTONS = [BTN_UP, BTN_DOWN, BTN_LEFT, BTN_RIGHT, BTN_STAY]

SEL_LAYER = "translucent layer"
BOX_ORIGIN = "box origin"
CUR_LATENT = "human latent"
PO_LAYER = "po layer"

# images for box push domain
IMG_ROBOT = 'robot'
IMG_WOMAN = 'woman'
IMG_MAN = 'man'
IMG_BOX = 'box'
IMG_TRASH_BAG = 'trash_bag'
IMG_WALL = 'wall'
IMG_GOAL = 'goal'
IMG_BOTH_BOX = 'both_box'
IMG_MAN_BAG = 'man_bag'
IMG_ROBOT_BAG = 'robot_bag'

# images for rescue domain
IMG_WORK = "work"
IMG_POLICE_CAR = "police_car"
IMG_FIRE_ENGINE = "fire_engine"
IMG_ROUTE = "route"
IMG_BACKGROUND = "background"
IMG_POLICE_STATION = "police_station"
IMG_FIRE_STATION = "fire_station"
IMG_CITY_HALL = "city_hall"
IMG_CAMPSITE = "campsite"
IMG_MALL = "mall"
IMG_BRIDGE = "bridge"
IMG_HUMAN = "human"
IMG_HOSPITAL = "hospital"
IMG_AMBULANCE = "ambulance"
IMG_TOWER = "tower"
IMG_THUNDER = "thunder"

################################################################################
# Classes here should always match with corresponding javascript classes
################################################################################


class DrawingObject:

  def __init__(self, name: str):
    self.name = name

  def get_dictionary(self):
    return self.__dict__


class ClippedRectangle(DrawingObject):

  def __init__(self,
               name: str,
               outer_ltwh: Sequence[int],
               list_circle: Optional[Sequence[Tuple[int, int, int]]] = None,
               list_rect: Optional[Sequence[Tuple[int, int, int, int]]] = None,
               list_rect_radii: Optional[Sequence[Tuple[int, int, int,
                                                        int]]] = None,
               fill_color: str = "grey",
               alpha: float = 0.3):
    super().__init__(name)
    self.obj_type = "ClippedRectangle"
    self.outer_ltwh = outer_ltwh
    self.list_rect = list_rect
    self.list_rect_radii = list_rect_radii
    self.list_circle = list_circle
    self.fill_color = fill_color
    self.alpha = alpha


class LineSegment(DrawingObject):

  def __init__(self,
               name: str,
               start: Sequence[int],
               end: Sequence[int],
               linewidth: int = 1,
               line_color: str = "black",
               alpha: float = 1.0):
    super().__init__(name)
    self.obj_type = "LineSegment"
    self.start = start
    self.end = end
    self.line_color = line_color
    self.alpha = alpha
    self.linewidth = linewidth


class Curve(DrawingObject):

  def __init__(self,
               name: str,
               coords: Sequence[Tuple[int, int]],
               linewidth: int = 1,
               line_color: str = "black",
               alpha: float = 1.0):
    super().__init__(name)
    self.obj_type = "Curve"
    self.coords = coords
    self.line_color = line_color
    self.alpha = alpha
    self.linewidth = linewidth


class Primitive(DrawingObject):

  def __init__(self,
               name: str,
               fill_color: str = "black",
               line_color: str = "black",
               alpha: float = 1.0,
               fill: bool = True,
               border: bool = True,
               linewidth: int = 1):
    super().__init__(name)
    self.fill_color = fill_color
    self.line_color = line_color
    self.alpha = alpha
    self.fill = fill
    self.border = border
    self.linewidth = linewidth


class Rectangle(Primitive):

  def __init__(self,
               name: str,
               pos: Sequence[int],
               size: Sequence[int],
               fill_color: str = "black",
               line_color: str = "black",
               alpha: float = 1.0,
               fill: bool = True,
               border: bool = False,
               linewidth: int = 1):
    super().__init__(name,
                     fill_color=fill_color,
                     line_color=line_color,
                     alpha=alpha,
                     fill=fill,
                     border=border,
                     linewidth=linewidth)
    self.obj_type = "Rectangle"
    self.pos = pos
    self.size = size


class Ellipse(Primitive):

  def __init__(self,
               name: str,
               pos: Sequence[int],
               size: Sequence[int],
               fill_color: str = "black",
               line_color: str = "black",
               alpha: float = 1.0,
               fill: bool = True,
               border: bool = False,
               linewidth: int = 1):
    super().__init__(name,
                     fill_color=fill_color,
                     line_color=line_color,
                     alpha=alpha,
                     fill=fill,
                     border=border,
                     linewidth=linewidth)
    self.obj_type = "Ellipse"
    self.pos = pos
    self.size = size


class Circle(Primitive):

  def __init__(self,
               name: str,
               pos: Sequence[int],
               radius: int,
               fill_color: str = "black",
               line_color: str = "black",
               alpha: float = 1.0,
               fill: bool = True,
               border: bool = False,
               linewidth: int = 1):
    super().__init__(name,
                     fill_color=fill_color,
                     line_color=line_color,
                     alpha=alpha,
                     fill=fill,
                     border=border,
                     linewidth=linewidth)
    self.obj_type = "Circle"
    self.pos = pos
    self.radius = radius


class BlinkCircle(Circle):

  def __init__(self,
               name: str,
               pos: Sequence[int],
               radius: int,
               fill_color: str = "black",
               line_color: str = "black",
               alpha: float = 1.0,
               fill: bool = True,
               border: bool = False,
               linewidth: int = 1):
    super().__init__(name,
                     pos=pos,
                     radius=radius,
                     fill_color=fill_color,
                     line_color=line_color,
                     alpha=alpha,
                     fill=fill,
                     border=border,
                     linewidth=linewidth)
    self.obj_type = "BlinkCircle"


class ButtonObject(Primitive):

  def __init__(self,
               name: str,
               pos: Sequence[int],
               font_size: int,
               text: str,
               disable: bool = False,
               text_align: str = "center",
               text_baseline: str = "middle",
               text_color: str = "black",
               fill_color: str = "black",
               line_color: str = "black",
               alpha: float = 1.0,
               fill: bool = False,
               border: bool = True,
               linewidth: int = 1):
    super().__init__(name,
                     fill_color=fill_color,
                     line_color=line_color,
                     alpha=alpha,
                     fill=fill,
                     border=border,
                     linewidth=linewidth)
    self.pos = pos
    self.font_size = font_size
    self.text = text
    self.disable = disable
    self.text_align = text_align
    self.text_baseline = text_baseline
    self.text_color = text_color


class ButtonRect(ButtonObject):

  def __init__(self,
               name: str,
               pos: Sequence[int],
               size: Sequence[int],
               font_size: int,
               text: str,
               disable: bool = False,
               text_align: str = "center",
               text_baseline: str = "middle",
               text_color: str = "black",
               fill_color: str = "black",
               line_color: str = "black",
               alpha: float = 1.0,
               fill: bool = False,
               border: bool = True,
               linewidth: int = 1):
    super().__init__(name,
                     pos=pos,
                     font_size=font_size,
                     text=text,
                     disable=disable,
                     text_align=text_align,
                     text_baseline=text_baseline,
                     text_color=text_color,
                     fill_color=fill_color,
                     line_color=line_color,
                     alpha=alpha,
                     fill=fill,
                     border=border,
                     linewidth=linewidth)
    self.size = size
    self.obj_type = "ButtonRect"


class ButtonCircle(ButtonObject):

  def __init__(self,
               name: str,
               pos: Sequence[int],
               radius: int,
               font_size: int,
               text: str,
               disable: bool = False,
               text_align: str = "center",
               text_baseline: str = "middle",
               text_color: str = "black",
               fill_color: str = "black",
               line_color: str = "black",
               alpha: float = 1,
               fill: bool = False,
               border: bool = True,
               linewidth: int = 1):
    super().__init__(name,
                     pos,
                     font_size,
                     text,
                     disable,
                     text_align,
                     text_baseline,
                     text_color,
                     fill_color,
                     line_color,
                     alpha,
                     fill,
                     border,
                     linewidth=linewidth)
    self.radius = radius
    self.obj_type = "ButtonCircle"


class ThickArrow(ButtonObject):

  def __init__(self,
               name: str,
               pos: Sequence[int],
               dir_vec: Tuple[float, float],
               width: float,
               disable: bool = False,
               fill_color: str = "black"):
    super().__init__(name,
                     pos=pos,
                     font_size=20,
                     text="",
                     disable=disable,
                     fill_color=fill_color,
                     fill=True,
                     border=False)
    self.obj_type = "ThickArrow"
    self.width = width
    self.dir_vec = dir_vec


class JoystickObject(ButtonObject):

  def __init__(self,
               name: str,
               pos: Sequence[int],
               width: int,
               fill_color: str = "black",
               disable: bool = False):
    super().__init__(name,
                     pos=pos,
                     font_size=20,
                     text="",
                     disable=disable,
                     fill_color=fill_color,
                     fill=True,
                     border=False)
    self.width = width


class JoystickUp(JoystickObject):

  def __init__(self,
               pos: Sequence[int],
               width: int,
               fill_color: str = "black",
               disable: bool = False):
    super().__init__(BTN_UP, pos, width, fill_color, disable)
    self.obj_type = "JoystickUp"


class JoystickDown(JoystickObject):

  def __init__(self,
               pos: Sequence[int],
               width: int,
               fill_color: str = "black",
               disable: bool = False):
    super().__init__(BTN_DOWN, pos, width, fill_color, disable)
    self.obj_type = "JoystickDown"


class JoystickLeft(JoystickObject):

  def __init__(self,
               pos: Sequence[int],
               width: int,
               fill_color: str = "black",
               disable: bool = False):
    super().__init__(BTN_LEFT, pos, width, fill_color, disable)
    self.obj_type = "JoystickLeft"


class JoystickRight(JoystickObject):

  def __init__(self,
               pos: Sequence[int],
               width: int,
               fill_color: str = "black",
               disable: bool = False):
    super().__init__(BTN_RIGHT, pos, width, fill_color, disable)
    self.obj_type = "JoystickRight"


class JoystickStay(JoystickObject):

  def __init__(self,
               pos: Sequence[int],
               width: int,
               fill_color: str = "black",
               disable: bool = False):
    super().__init__(BTN_STAY, pos, width, fill_color, disable)
    self.obj_type = "JoystickStay"


class TextObject(DrawingObject):

  def __init__(self,
               name: str,
               pos: Sequence[int],
               width: int,
               font_size: int,
               text: str,
               text_align: str = "left",
               text_baseline: str = "top",
               text_color: str = "black"):
    super().__init__(name)
    self.pos = pos
    self.width = width
    self.font_size = font_size
    self.text = text
    self.text_align = text_align
    self.text_baseline = text_baseline
    self.text_color = text_color
    self.obj_type = "TextObject"


class GameObject(DrawingObject):

  def __init__(self, name: str, pos: Sequence[int], size: Sequence[int],
               angle: float, img_name: str):
    super().__init__(name)
    self.pos = pos
    self.size = size
    self.angle = angle
    self.img_name = img_name
    self.obj_type = "GameObject"


class SelectingCircle(ButtonCircle):

  def __init__(self, name: str, pos: Sequence[int], radius: int, font_size: int,
               text: str):
    super().__init__(name,
                     pos,
                     radius,
                     font_size,
                     text,
                     disable=False,
                     fill_color="red",
                     line_color="red",
                     alpha=0.8,
                     fill=False,
                     border=True)
