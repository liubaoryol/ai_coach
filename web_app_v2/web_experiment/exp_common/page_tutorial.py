from typing import Mapping, Sequence, Any
from web_experiment.exp_common.page_base import Exp1UserData, ExperimentPageBase
from web_experiment.exp_common.page_exp1_game_base import get_holding_box_idx
from web_experiment.exp_common.page_exp1_common import CanvasPageStart
from web_experiment.exp_common.page_boxpushv2_base import BoxPushV2GamePage
import web_experiment.exp_common.canvas_objects as co
from web_experiment.models import db, ExpDataCollection, ExpIntervention
from web_experiment.define import ExpType, EDomainType
from aic_domain.agent import InteractiveAgent
from aic_domain.box_push_v2 import (conv_box_state_2_idx, EventType, BoxState,
                                    BoxPushSimulatorV2)
from aic_domain.box_push_v2.agent import (BoxPushAIAgent_PO_Indv,
                                          BoxPushAIAgent_PO_Team)


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


class CanvasPageTutorialBase(BoxPushV2GamePage):
  CLICKED_BTNS = "clicked_btn"
  RED_CIRCLE = "red_circle"

  def __init__(self, domain_type, latent_collection: bool = True) -> None:
    super().__init__(domain_type, latent_collection)
    self._MANUAL_SELECTION = self._PROMPT_ON_CHANGE = self._AUTO_PROMPT = False

  # _base methods: to avoid using super() method at downstream classes
  def _base_init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    game = user_game_data.get_game_ref()
    agent1 = InteractiveAgent()
    agent2 = InteractiveAgent()
    game.set_autonomous_agent(agent1, agent2)

    PICKUP_BOX = 1 if self._DOMAIN_TYPE == EDomainType.Movers else 0
    game.event_input(self._AGENT1, EventType.SET_LATENT, ("pickup", PICKUP_BOX))
    user_game_data.data[Exp1UserData.SELECT] = False
    user_game_data.data[Exp1UserData.PARTIAL_OBS] = False

  def _base_get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_game_data)

    btn_prev, btn_next = self._get_btn_prev_next(False, False)
    dict_objs[btn_prev.name] = btn_prev
    dict_objs[btn_next.name] = btn_next

    return dict_objs

  def _base_button_clicked(self, user_game_data: Exp1UserData,
                           clicked_btn: str):
    return super().button_clicked(user_game_data, clicked_btn)

  def _base_get_updated_drawing_objects(
      self,
      user_data: Exp1UserData,
      dict_prev_game: Mapping[str,
                              Any] = None) -> Mapping[str, co.DrawingObject]:
    return super()._get_updated_drawing_objects(user_data, dict_prev_game)

  def _base_on_action_taken(self, user_game_data: Exp1UserData,
                            dict_prev_game: Mapping[str, Any],
                            tuple_actions: Sequence[Any]):
    return super()._on_action_taken(user_game_data, dict_prev_game,
                                    tuple_actions)

  def init_user_data(self, user_game_data: Exp1UserData):
    self._base_init_user_data(user_game_data)

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    return self._base_get_init_drawing_objects(user_game_data)

  def _get_red_circle(self, x_cen, y_cen, radius):
    return co.BlinkCircle(self.RED_CIRCLE, (x_cen, y_cen),
                          radius,
                          line_color="red",
                          fill=False,
                          border=True,
                          linewidth=3)

  def _get_drawing_order(self, user_game_data: Exp1UserData):
    drawing_order = super()._get_drawing_order(user_game_data)

    additional_objs_order = [self.SPOTLIGHT, co.BTN_PREV, co.BTN_NEXT]
    if not self._LATENT_COLLECTION:
      additional_objs_order.append(self.RED_CIRCLE)

    for obj_name in additional_objs_order:
      if obj_name in user_game_data.data[Exp1UserData.DRAW_OBJ_NAMES]:
        drawing_order.append(obj_name)

    return drawing_order

  def _on_game_finished(self, user_game_data: Exp1UserData):
    '''
    user_game_data: NOTE - values will be updated
    '''

    game = user_game_data.get_game_ref()
    user_game_data.data[Exp1UserData.GAME_DONE] = True
    game.reset_game()
    user_game_data.data[Exp1UserData.SCORE] = game.current_step
    self.init_user_data(user_game_data)


class CanvasPageJoystick(CanvasPageTutorialBase):
  def __init__(self, domain_type, latent_collection: bool = True) -> None:
    super().__init__(domain_type, latent_collection)

  def init_user_data(self, user_game_data: Exp1UserData):
    self._base_init_user_data(user_game_data)

    user_game_data.data[self.CLICKED_BTNS] = set()
    user_game_data.data[Exp1UserData.SELECT] = False
    user_game_data.data[Exp1UserData.PARTIAL_OBS] = False
    game = user_game_data.get_game_ref()
    game.event_input(self._AGENT1, EventType.SET_LATENT, None)

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = self._base_get_init_drawing_objects(user_game_data)

    dict_objs[co.BTN_NEXT].disable = True

    obj = dict_objs[co.BTN_STAY]  # type: co.JoystickStay
    center = obj.pos
    radius = int(obj.width * 1.7)
    obj = self._get_spotlight(*center, radius)
    dict_objs[obj.name] = obj

    return dict_objs

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("During the task, you control the human player. " +
            "You can move the human player by clicking the motion buttons. " +
            "Once you have tested each motion button " +
            "(left, right, up, and down), " +
            "please click on the \"Next\" button to continue.")

  def button_clicked(self, user_game_data: Exp1UserData, clicked_btn: str):
    if clicked_btn in co.JOYSTICK_BUTTONS and clicked_btn != co.BTN_STAY:
      user_game_data.data[self.CLICKED_BTNS].add(clicked_btn)

    return self._base_button_clicked(user_game_data, clicked_btn)

  def _get_button_commands(self, clicked_btn, user_data: Exp1UserData):
    if clicked_btn in co.JOYSTICK_BUTTONS and clicked_btn != co.BTN_STAY:
      return {"delete": [self.SPOTLIGHT, self.RED_CIRCLE]}

    return None

  def _get_updated_drawing_objects(
      self,
      user_data: Exp1UserData,
      dict_prev_game: Mapping[str,
                              Any] = None) -> Mapping[str, co.DrawingObject]:
    dict_objs = self._base_get_updated_drawing_objects(user_data,
                                                       dict_prev_game)

    btn_prev, btn_next = self._get_btn_prev_next(False, True)
    dict_objs[btn_prev.name] = btn_prev
    dict_objs[btn_next.name] = btn_next

    clicked_btns = user_data.data[self.CLICKED_BTNS]  # type: set
    if len(clicked_btns) == 0:
      for obj_name in clicked_btns:
        obj = dict_objs[obj_name]  # type: co.JoystickObject
        obj.fill_color = "LightGreen"
    else:
      obj = dict_objs[btn_next.name]  # type: co.ButtonRect
      obj.disable = False

    return dict_objs


class CanvsPageWaitBtn(CanvasPageJoystick):
  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = self._base_get_init_drawing_objects(user_game_data)

    dict_objs[co.BTN_NEXT].disable = True

    obj = dict_objs[co.BTN_STAY]  # type: co.JoystickStay

    center = obj.pos
    radius = int(obj.width * 0.8)
    obj = self._get_spotlight(*center, radius)
    dict_objs[obj.name] = obj

    if not self._LATENT_COLLECTION:
      obj = self._get_red_circle(*center, radius)
      dict_objs[obj.name] = obj

    return dict_objs

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("You can wait by clicking on the circular button. " +
            "Once you test the wait button, " +
            "please click on the \"Next\" button to continue.")

  def button_clicked(self, user_game_data: Exp1UserData, clicked_btn: str):
    if clicked_btn == co.BTN_STAY:
      user_game_data.data[self.CLICKED_BTNS].add(clicked_btn)

    return self._base_button_clicked(user_game_data, clicked_btn)

  def _get_button_commands(self, clicked_btn, user_data: Exp1UserData):
    if clicked_btn == co.BTN_STAY:
      return {"delete": [self.SPOTLIGHT, self.RED_CIRCLE]}

    return None


class CanvasPageInvalidAction(CanvasPageTutorialBase):
  def __init__(self, domain_type, latent_collection: bool = True) -> None:
    super().__init__(domain_type, latent_collection)

  def init_user_data(self, user_game_data: Exp1UserData):
    self._base_init_user_data(user_game_data)

    user_game_data.data[Exp1UserData.SELECT] = False
    user_game_data.data[Exp1UserData.PARTIAL_OBS] = False
    game = user_game_data.get_game_ref()
    game.event_input(self._AGENT1, EventType.SET_LATENT, None)

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("If you take an invalid action (e.g., try to move into a wall), " +
            "the human player will just vibrate on the spot.")


class CanvasPageJoystickShort(CanvasPageTutorialBase):
  def __init__(self, domain_type, latent_collection: bool = True) -> None:
    super().__init__(domain_type, latent_collection)

  def init_user_data(self, user_game_data: Exp1UserData):
    self._base_init_user_data(user_game_data)

    game = user_game_data.get_game_ref()
    game.event_input(self._AGENT1, EventType.SET_LATENT, None)

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("During the task, you control the human player. " +
            "You can move the human player similar to the previous task. " +
            "If you take an invalid action (e.g., try to move into a wall), " +
            "the human player will just vibrate on the spot.")


class CanvasPageOnlyHuman(CanvasPageTutorialBase):
  def __init__(self, domain_type, latent_collection: bool = True) -> None:
    super().__init__(domain_type, latent_collection)

  def init_user_data(self, user_game_data: Exp1UserData):
    self._base_init_user_data(user_game_data)

    user_game_data.data[Exp1UserData.SELECT] = False
    user_game_data.data[Exp1UserData.PARTIAL_OBS] = False
    game = user_game_data.get_game_ref()
    game.event_input(self._AGENT1, EventType.SET_LATENT, None)

  def _get_instruction(self, user_game_data: Exp1UserData):
    return (
        "While the success of the task depends on both you and the robot, " +
        "you cannot control the robot. You can only control the human player. "
        + "The robot moves autonomously.")


class CanvasPageGoToTarget(CanvasPageTutorialBase):
  def __init__(self, domain_type, latent_collection: bool = True) -> None:
    super().__init__(domain_type, latent_collection)

  def _get_instruction(self, user_game_data: Exp1UserData):
    object_type = ("box"
                   if self._DOMAIN_TYPE == EDomainType.Movers else "trash bag")

    if self._LATENT_COLLECTION:
      inst = (
          "The red circle indicates your current destination. " +
          f"Please move to the {object_type}" +
          " (using the motion buttons) and try to pick it. The pick button will"
          + " be available only when you are at the correct destination.")
    else:
      inst = ("Let's figure out how to play this task step by step. " +
              f"First, please move to the bottom {object_type} " +
              "(circled in red).")

    return inst

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = self._base_get_init_drawing_objects(user_game_data)
    dict_objs[co.BTN_NEXT].disable = True

    if not self._LATENT_COLLECTION:
      game_env = user_game_data.get_game_ref().get_env_info()
      center_pos, radius = self._get_latent_pos_overlay(game_env)
      if center_pos is not None:
        obj = self._get_red_circle(*center_pos, radius)
        dict_objs[obj.name] = obj

    return dict_objs

  def _on_action_taken(self, user_game_data: Exp1UserData,
                       dict_prev_game: Mapping[str, Any],
                       tuple_actions: Sequence[Any]):
    self._base_on_action_taken(user_game_data, dict_prev_game, tuple_actions)

    game = user_game_data.get_game_ref()
    game_env = game.get_env_info()
    user_game_data.data[Exp1UserData.SCORE] = game_env["current_step"]

    a1_latent = game_env["a1_latent"]
    if a1_latent is not None and a1_latent[0] == "pickup":
      box_coord = game_env["boxes"][a1_latent[1]]
      a1_pos = game_env["a1_pos"]
      if a1_pos == box_coord:
        user_game_data.go_to_next_page()


class CanvasPagePickUpTargetAttempt(CanvasPageTutorialBase):
  def __init__(self, domain_type, latent_collection: bool = True) -> None:
    super().__init__(domain_type, latent_collection)

  def init_user_data(self, user_game_data: Exp1UserData):
    self._base_init_user_data(user_game_data)

    game = user_game_data.get_game_ref()

    PICKUP_BOX = 1 if self._DOMAIN_TYPE == EDomainType.Movers else 0
    game.a1_pos = game.boxes[PICKUP_BOX]
    game.current_step = user_game_data.data[Exp1UserData.SCORE]
    game.event_input(self._AGENT1, EventType.SET_LATENT, ("pickup", PICKUP_BOX))

    user_game_data.data[self.CLICKED_BTNS] = set()

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = self._base_get_init_drawing_objects(user_game_data)

    obj = dict_objs[co.BTN_PICK_UP]  # type: co.ButtonRect
    pos = obj.pos
    radi = int(obj.size[0] * 0.6)
    obj = self._get_spotlight(*pos, radi)
    dict_objs[obj.name] = obj

    dict_objs[co.BTN_NEXT].disable = True

    objs = self._get_btn_actions(True, True, True, True, True, False, True)
    for obj in objs:
      dict_objs[obj.name] = obj

    if self._LATENT_COLLECTION:
      obj = self._get_btn_select(True)
      dict_objs[obj.name] = obj

    return dict_objs

  def _get_instruction(self, user_game_data: Exp1UserData):
    if self._DOMAIN_TYPE == EDomainType.Movers:
      return ("Now, please pick it up using the (\"Pick Up\" button). " +
              "You will notice that you cannot pick up the box alone. " +
              "You have to pick it up together with the robot. " +
              "Once you have clicked the \"Pick Up\" button, " +
              "please click on the \"Next\" button to continue.")
    else:
      return ("Now, please pick it up using the (pick button). " +
              "You will notice that you can pick up the trash bag alone. " +
              "You don't need to wait for the robot.")

  def _on_action_taken(self, user_game_data: Exp1UserData,
                       dict_prev_game: Mapping[str, Any],
                       tuple_actions: Sequence[Any]):
    self._base_on_action_taken(user_game_data, dict_prev_game, tuple_actions)

    game = user_game_data.get_game_ref()
    game_env = game.get_env_info()
    user_game_data.data[Exp1UserData.SCORE] = game_env["current_step"]

    if self._DOMAIN_TYPE == EDomainType.Cleanup:
      num_drops = len(game_env["drops"])
      num_goals = len(game_env["goals"])
      a1_box, _ = get_holding_box_idx(game_env["box_states"], num_drops,
                                      num_goals)
      if a1_box >= 0:
        user_game_data.go_to_next_page()

  def button_clicked(self, user_game_data: Exp1UserData, clicked_btn: str):
    if clicked_btn == co.BTN_PICK_UP:
      user_game_data.data[self.CLICKED_BTNS].add(co.BTN_PICK_UP)

    return self._base_button_clicked(user_game_data, clicked_btn)

  def _get_button_commands(self, clicked_btn, user_data: Exp1UserData):
    if clicked_btn == co.BTN_PICK_UP:
      return {"delete": [self.SPOTLIGHT, self.RED_CIRCLE]}

    return None

  def _get_updated_drawing_objects(
      self,
      user_data: Exp1UserData,
      dict_prev_game: Mapping[str,
                              Any] = None) -> Mapping[str, co.DrawingObject]:
    dict_objs = self._base_get_updated_drawing_objects(user_data,
                                                       dict_prev_game)

    if co.BTN_PICK_UP not in user_data.data[self.CLICKED_BTNS]:
      objs = self._get_btn_actions(True, True, True, True, True, False, True)
      for obj in objs:
        dict_objs[obj.name] = obj

      if self._LATENT_COLLECTION:
        obj = self._get_btn_select(True)
        dict_objs[obj.name] = obj
    else:
      _, btn_next = self._get_btn_prev_next(False, False)
      dict_objs[btn_next.name] = btn_next

    return dict_objs


class CanvasPagePickUpTarget(CanvasPageTutorialBase):
  def __init__(self, domain_type, latent_collection: bool = True) -> None:
    super().__init__(domain_type, latent_collection)

  def init_user_data(self, user_game_data: Exp1UserData):
    self._base_init_user_data(user_game_data)

    agent1 = InteractiveAgent()

    init_states = ([0] * len(self._GAME_MAP["boxes"]),
                   self._GAME_MAP["a1_init"], self._GAME_MAP["a2_init"])
    if self._DOMAIN_TYPE == EDomainType.Movers:
      agent2 = BoxPushAIAgent_PO_Team(init_states,
                                      self._TEAMMATE_POLICY,
                                      agent_idx=self._AGENT2)
    else:
      agent2 = InteractiveAgent()

    game = user_game_data.get_game_ref()
    game.set_autonomous_agent(agent1, agent2)

    PICKUP_BOX = 1 if self._DOMAIN_TYPE == EDomainType.Movers else 0
    game.event_input(self._AGENT1, EventType.SET_LATENT, ("pickup", PICKUP_BOX))
    if self._DOMAIN_TYPE == EDomainType.Movers:
      game.event_input(self._AGENT2, EventType.SET_LATENT,
                       ("pickup", PICKUP_BOX))

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = self._base_get_init_drawing_objects(user_game_data)
    dict_objs[co.BTN_NEXT].disable = True

    if not self._LATENT_COLLECTION:
      game_env = user_game_data.get_game_ref().get_env_info()
      center_pos, radius = self._get_latent_pos_overlay(game_env)
      if center_pos is not None:
        obj = self._get_red_circle(*center_pos, radius)
        dict_objs[obj.name] = obj

    return dict_objs

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("Please move to the box (red circle) and try to pick it again. " +
            "This time your robot teammate will also move to it. " +
            "Wait for your teammate at the box and pick it up together.")

  def _on_action_taken(self, user_game_data: Exp1UserData,
                       dict_prev_game: Mapping[str, Any],
                       tuple_actions: Sequence[Any]):
    self._base_on_action_taken(user_game_data, dict_prev_game, tuple_actions)

    game = user_game_data.get_game_ref()
    game_env = game.get_env_info()
    user_game_data.data[Exp1UserData.SCORE] = game_env["current_step"]

    num_drops = len(game_env["drops"])
    num_goals = len(game_env["goals"])
    a1_box, _ = get_holding_box_idx(game_env["box_states"], num_drops,
                                    num_goals)
    if a1_box >= 0:
      user_game_data.go_to_next_page()


class CanvasPageGoToGoal(CanvasPageTutorialBase):
  def __init__(self, domain_type, latent_collection: bool = True) -> None:
    super().__init__(domain_type, latent_collection)

  def init_user_data(self, user_game_data: Exp1UserData):
    self._base_init_user_data(user_game_data)

    game = user_game_data.get_game_ref()  # type: BoxPushSimulatorV2
    agent1 = InteractiveAgent()

    init_states = ([0] * len(self._GAME_MAP["boxes"]),
                   self._GAME_MAP["a1_init"], self._GAME_MAP["a2_init"])
    if self._DOMAIN_TYPE == EDomainType.Movers:
      agent2 = BoxPushAIAgent_PO_Team(init_states,
                                      self._TEAMMATE_POLICY,
                                      agent_idx=self._AGENT2)
    else:
      agent2 = InteractiveAgent()

    game.set_autonomous_agent(agent1, agent2)
    game.current_step = user_game_data.data[Exp1UserData.SCORE]
    PICKUP_BOX = 1 if self._DOMAIN_TYPE == EDomainType.Movers else 0
    if self._DOMAIN_TYPE == EDomainType.Movers:
      game.a1_pos = game.boxes[PICKUP_BOX]
      game.a2_pos = game.boxes[PICKUP_BOX]
      game.box_states[PICKUP_BOX] = conv_box_state_2_idx(
          (BoxState.WithBoth, None), len(game.drops))
      game.event_input(self._AGENT1, EventType.SET_LATENT, ("goal", 0))
      game.event_input(self._AGENT2, EventType.SET_LATENT, ("goal", 0))
      agent2.assumed_tup_states = (game.box_states, game.a1_pos, game.a2_pos)
    else:
      game.a1_pos = game.boxes[PICKUP_BOX]
      game.box_states[PICKUP_BOX] = conv_box_state_2_idx(
          (BoxState.WithAgent1, None), len(game.drops))
      game.event_input(self._AGENT1, EventType.SET_LATENT, ("goal", 0))

  def _on_action_taken(self, user_game_data: Exp1UserData,
                       dict_prev_game: Mapping[str, Any],
                       tuple_actions: Sequence[Any]):
    self._base_on_action_taken(user_game_data, dict_prev_game, tuple_actions)

    game = user_game_data.get_game_ref()
    user_game_data.data[Exp1UserData.SCORE] = game.current_step

    game_env = game.get_env_info()
    num_drops = len(game_env["drops"])
    num_goals = len(game_env["goals"])
    a1_box, _ = get_holding_box_idx(game_env["box_states"], num_drops,
                                    num_goals)
    if a1_box < 0:
      user_game_data.go_to_next_page()

  def _get_instruction(self, user_game_data: Exp1UserData):
    object_type = ("box"
                   if self._DOMAIN_TYPE == EDomainType.Movers else "trash bag")

    return ("After picking up the " + object_type + ", you need to drop it " +
            "at the truck. " + "Please carry the " + object_type +
            " to the truck and drop it there.")

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = self._base_get_init_drawing_objects(user_game_data)
    dict_objs[co.BTN_NEXT].disable = True

    if not self._LATENT_COLLECTION:
      game_env = user_game_data.get_game_ref().get_env_info()
      center_pos, radius = self._get_latent_pos_overlay(game_env)
      if center_pos is not None:
        obj = self._get_red_circle(*center_pos, radius)
        dict_objs[obj.name] = obj

    return dict_objs


class CanvasPageRespawn(CanvasPageTutorialBase):
  def __init__(self, domain_type, latent_collection: bool = True) -> None:
    super().__init__(domain_type, latent_collection)

  def init_user_data(self, user_game_data: Exp1UserData):
    # game no need to be initialized

    game = user_game_data.get_game_ref()
    game.set_autonomous_agent()
    game.event_input(self._AGENT1, EventType.SET_LATENT, None)
    user_game_data.data[Exp1UserData.ACTION_COUNT] = 0
    user_game_data.data[Exp1UserData.SELECT] = False
    user_game_data.data[Exp1UserData.PARTIAL_OBS] = False

  def _get_instruction(self, user_game_data: Exp1UserData):
    object_type = ("box"
                   if self._DOMAIN_TYPE == EDomainType.Movers else "trash bag")
    return ("Well done! Once you drop the box, " +
            "you will automatically step out of the truck. " +
            "You cannot step on the truck without a " + object_type + ".")


class CanvasPageScore(CanvasPageTutorialBase):
  def __init__(self, domain_type, latent_collection: bool = True) -> None:
    super().__init__(domain_type, latent_collection)

  def init_user_data(self, user_game_data: Exp1UserData):
    # game no need to be initialized

    user_game_data.data[Exp1UserData.ACTION_COUNT] = 0
    user_game_data.data[Exp1UserData.SELECT] = False
    user_game_data.data[Exp1UserData.PARTIAL_OBS] = False

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = self._base_get_init_drawing_objects(user_game_data)

    objs = self._get_btn_actions(True, True, True, True, True, True, True)
    for obj in objs:
      dict_objs[obj.name] = obj

    if self._LATENT_COLLECTION:
      obj = self._get_btn_select(True)
      dict_objs[obj.name] = obj

    obj = dict_objs[self.TEXT_SCORE]  # type: co.TextObject
    x_cen = int(obj.pos[0] + 0.95 * obj.width)
    y_cen = int(obj.pos[1] + 0.5 * obj.font_size)
    radius = int(obj.font_size * 2)
    obj = self._get_spotlight(x_cen, y_cen, radius)
    dict_objs[obj.name] = obj

    return dict_objs

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("You might have noticed that as you were doing the" +
            " task the \"Time Taken\" counter (shown below) was increasing. " +
            "Your goal is to complete the task as fast as possible " +
            "(i.e., with the least amount of time taken).")


class CanvasPagePartialObs(CanvasPageTutorialBase):
  def __init__(self, domain_type, latent_collection: bool = True) -> None:
    super().__init__(domain_type, latent_collection)

  def init_user_data(self, user_game_data: Exp1UserData):
    self._base_init_user_data(user_game_data)

    user_game_data.data[Exp1UserData.PARTIAL_OBS] = True

  def _get_instruction(self, user_game_data: Exp1UserData):
    return (
        "Unlike the tutorial so far, during the TASK sessions, You cannot " +
        "observe the full game environment. You can see your robot " +
        "teammate or objects only if they are in close proximity. " +
        "Similarly, your robot teammate also does NOT know where you are " +
        "unless you are in close proximity.")


class CanvasPagePORobot(CanvasPageTutorialBase):
  def __init__(self, domain_type, latent_collection: bool = True) -> None:
    super().__init__(domain_type, latent_collection)

  def init_user_data(self, user_game_data: Exp1UserData):
    self._base_init_user_data(user_game_data)
    agent1 = InteractiveAgent()
    init_states = ([0] * len(self._GAME_MAP["boxes"]),
                   self._GAME_MAP["a1_init"], self._GAME_MAP["a2_init"])
    if self._DOMAIN_TYPE == EDomainType.Movers:
      agent2 = BoxPushAIAgent_PO_Team(init_states,
                                      self._TEAMMATE_POLICY,
                                      agent_idx=self._AGENT2)
    else:
      agent2 = InteractiveAgent()

    game = user_game_data.get_game_ref()  # type: BoxPushSimulatorV2

    game.set_autonomous_agent(agent1, agent2)
    game.event_input(self._AGENT1, EventType.SET_LATENT, None)

    user_game_data.data[Exp1UserData.PARTIAL_OBS] = True
    user_game_data.data[Exp1UserData.SELECT] = False

  def _get_instruction(self, user_game_data: Exp1UserData):
    return (
        "Your robot teammate also does NOT know where you are unless you are in "
        + "its neighborhood. The robot will choose or update its destination " +
        "independently of you.")


class CanvasPageTarget(CanvasPageTutorialBase):
  def __init__(self, domain_type) -> None:
    super().__init__(domain_type, True)
    self._MANUAL_SELECTION = False
    self._PROMPT_ON_CHANGE = self._AUTO_PROMPT = True

  def init_user_data(self, user_game_data: Exp1UserData):
    self._base_init_user_data(user_game_data)

    user_game_data.data[Exp1UserData.PARTIAL_OBS] = True

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = self._base_get_init_drawing_objects(user_game_data)

    # obj = dict_objs[self.TEXT_INSTRUCTION]  # type: co.TextObject
    # x_cen = int(obj.pos[0] + 0.5 * obj.width)
    # y_cen = int(self.GAME_HEIGHT / 5)
    # radius = int(y_cen * 0.1)
    # dict_objs[self.SPOTLIGHT] = self._get_spotlight(x_cen, y_cen, radius)

    objs = self._get_btn_actions(True, True, True, True, True, True, True)
    for obj in objs:
      dict_objs[obj.name] = obj

    if self._LATENT_COLLECTION:
      obj = self._get_btn_select(True)
      dict_objs[obj.name] = obj

    return dict_objs

  def _get_instruction(self, user_game_data: Exp1UserData):
    return (
        "In the TASK sessions, you will have to select your next destination" +
        " using the \"Select Destination\" button. " +
        "Let's see how to do this! Please click on " +
        "the \"Next\" button to continue.")


class CanvasPageLatent(CanvasPageTutorialBase):
  def __init__(self, domain_type) -> None:
    super().__init__(domain_type, True)
    self._MANUAL_SELECTION = self._PROMPT_ON_CHANGE = self._AUTO_PROMPT = True

  def init_user_data(self, user_game_data: Exp1UserData):
    self._base_init_user_data(user_game_data)

    user_game_data.data[Exp1UserData.PARTIAL_OBS] = True

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = self._base_get_init_drawing_objects(user_game_data)
    obj = dict_objs[co.BTN_SELECT]  # type: co.ButtonRect
    obj = self._get_spotlight(*obj.pos, int(obj.size[0] * 0.6))
    dict_objs[obj.name] = obj

    dict_objs[co.BTN_NEXT].disable = True

    objs = self._get_btn_actions(True, True, True, True, True, True, True)
    for obj in objs:
      dict_objs[obj.name] = obj

    obj = self._get_btn_select(False)
    dict_objs[obj.name] = obj

    return dict_objs

  def _get_instruction(self, user_game_data: Exp1UserData):
    object_type = ("box"
                   if self._DOMAIN_TYPE == EDomainType.Movers else "trash bag")
    return (
        "First, click the \"Select Destination\" button. " +
        "Possible destinations are circled in red and shown as an overlay. " +
        "Please click on your current destination (i.e., the " + object_type +
        " which you are planning to pick next).")

  def _get_button_commands(self, clicked_btn, user_data: Exp1UserData):
    if clicked_btn == co.BTN_SELECT:
      return {"delete": [self.SPOTLIGHT, self.RED_CIRCLE]}

    return None

  def button_clicked(self, user_game_data: Exp1UserData, clicked_btn: str):
    if self.is_sel_latent_btn(clicked_btn):
      user_game_data.go_to_next_page()

    return self._base_button_clicked(user_game_data, clicked_btn)


class CanvasPageTutorialPlain(CanvasPageTutorialBase):
  def __init__(self, domain_type, latent_collection: bool = True) -> None:
    super().__init__(domain_type, latent_collection)

  def init_user_data(self, user_game_data: Exp1UserData):
    # game no need to be initialized
    user_game_data.data[Exp1UserData.ACTION_COUNT] = 0
    user_game_data.data[Exp1UserData.SELECT] = False
    user_game_data.data[Exp1UserData.PARTIAL_OBS] = True

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ""

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = self._base_get_init_drawing_objects(user_game_data)

    objs = self._get_btn_actions(True, True, True, True, True, True, True)
    for obj in objs:
      dict_objs[obj.name] = obj

    if self._LATENT_COLLECTION:
      obj = self._get_btn_select(True)
      dict_objs[obj.name] = obj

    return dict_objs


class CanvasPageSelResult(CanvasPageTutorialPlain):
  def __init__(self, domain_type, is_2nd) -> None:
    super().__init__(domain_type, True)
    self._IS_2ND = is_2nd
    self._MANUAL_SELECTION = False
    self._PROMPT_ON_CHANGE = self._AUTO_PROMPT = True

  def _get_instruction(self, user_game_data: Exp1UserData):
    if self._IS_2ND:
      return ("Great! As before, your choice is marked with the flashing red " +
              "circle and you have selected your next destination.")
    else:
      return (
          "Well done! Now you can see your choice is marked with the " +
          "flashing red circle and you have selected your next destination. " +
          "The selection will NOT inform the robot of anything. " +
          "Your robot teammate does NOT know your selected destination and " +
          "will choose its destination independently of you.")


class CanvasPageImportance(CanvasPageTutorialPlain):
  def __init__(self, domain_type) -> None:
    super().__init__(domain_type, True)

  def _get_instruction(self, user_game_data: Exp1UserData):
    return (
        "Selecting the correct destination is VERY IMPORTANT for our research. "
        + "Any time you change your mind about your destination, " +
        "please use the \"Select Destination\" button " +
        "to indicate your updated destination.")


class CanvasPageSelPrompt(CanvasPageTutorialBase):
  def __init__(self, domain_type) -> None:
    super().__init__(domain_type, True)
    self._MANUAL_SELECTION = False
    self._AUTO_PROMPT = self._PROMPT_ON_CHANGE = True
    self._PROMPT_FREQ = 3

  def init_user_data(self, user_game_data: Exp1UserData):
    self._base_init_user_data(user_game_data)

    user_game_data.data[Exp1UserData.PARTIAL_OBS] = True
    game = user_game_data.get_game_ref()
    game.event_input(self._AGENT1, EventType.SET_LATENT, None)

  def _get_instruction(self, user_game_data: Exp1UserData):
    return (
        "We will also prompt the destination selection auto- matically " +
        "and periodically during the TASK sessions. Please move the human " +
        "player several steps. When the destination selection is prompted" +
        ", please click on your current destination.")

  def button_clicked(self, user_game_data: Exp1UserData, clicked_btn: str):
    if self.is_sel_latent_btn(clicked_btn):
      user_game_data.go_to_next_page()

    return self._base_button_clicked(user_game_data, clicked_btn)

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = self._base_get_init_drawing_objects(user_game_data)

    dict_objs[co.BTN_NEXT].disable = True

    return dict_objs


class CanvasPageExpGoal(CanvasPageTutorialPlain):
  def _get_instruction(self, user_game_data: Exp1UserData):
    inst = (
        "Remember that in the TASK session, " +
        "your goal is to move all the boxes to the truck as soon as possible." +
        " You cannot pick up a box alone. ")
    if self._LATENT_COLLECTION:
      inst += ("Also, you can only pick up or " +
               "drop a box at the place circled in red.")
    return inst


class CanvasPageMiniGame(CanvasPageTutorialBase):
  def __init__(self, domain_type, latent_collection=True) -> None:
    super().__init__(domain_type, latent_collection)
    self._MANUAL_SELECTION = latent_collection
    self._AUTO_PROMPT = latent_collection
    self._PROMPT_ON_CHANGE = latent_collection

  def _get_instruction(self, user_game_data: Exp1UserData):
    return (
        "Now, we are at the final step of the TUTORIAL. Feel free to " +
        "interact with the interface and get familiar with the task. You" +
        " can also press the \"Prev\" button to revisit any of the previous " +
        "prompts.\n Once you are ready, please proceed to the TASK " +
        "sessions (using the button at the bottom of this page).")

  def init_user_data(self, user_game_data: Exp1UserData):
    self._base_init_user_data(user_game_data)

    game = user_game_data.get_game_ref()
    agent1 = InteractiveAgent()
    init_states = ([0] * len(self._GAME_MAP["boxes"]),
                   self._GAME_MAP["a1_init"], self._GAME_MAP["a2_init"])
    if self._DOMAIN_TYPE == EDomainType.Movers:
      agent2 = BoxPushAIAgent_PO_Team(init_states,
                                      self._TEAMMATE_POLICY,
                                      agent_idx=self._AGENT2)
    else:
      agent2 = BoxPushAIAgent_PO_Indv(init_states,
                                      self._TEAMMATE_POLICY,
                                      agent_idx=self._AGENT2)

    game.set_autonomous_agent(agent1, agent2)
    game.event_input(self._AGENT1, EventType.SET_LATENT, ("pickup", 0))

    user_game_data.data[Exp1UserData.PARTIAL_OBS] = True
    user_game_data.data[Exp1UserData.SELECT] = self._LATENT_COLLECTION

    # set task done
    user = user_game_data.data[Exp1UserData.USER]
    session_name = user_game_data.data[Exp1UserData.SESSION_NAME]
    user_id = user.userid
    if user_game_data.data[Exp1UserData.EXP_TYPE] == ExpType.Data_collection:
      exp = ExpDataCollection.query.filter_by(subject_id=user_id).first()
    elif user_game_data.data[Exp1UserData.EXP_TYPE] == ExpType.Intervention:
      exp = ExpIntervention.query.filter_by(subject_id=user_id).first()
    if not getattr(exp, session_name):
      setattr(exp, session_name, True)
      db.session.commit()
      user_game_data.data[Exp1UserData.SESSION_DONE] = True

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = self._base_get_init_drawing_objects(user_game_data)

    dict_objs[co.BTN_NEXT].disable = True

    return dict_objs
