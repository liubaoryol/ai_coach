from typing import Mapping, Sequence, Any
from web_experiment.exp_common.page_base import Exp1UserData, ExperimentPageBase
from web_experiment.exp_common.page_exp1_game_base import get_holding_box_idx
from web_experiment.exp_common.page_exp1_common import CanvasPageStart
from web_experiment.exp_common.page_boxpushv2_base import BoxPushV2GamePage
import web_experiment.exp_common.canvas_objects as co
from web_experiment.models import db, ExpDataCollection, ExpIntervention
from web_experiment.define import ExpType, EDomainType
from ai_coach_domain.agent import InteractiveAgent
from ai_coach_domain.box_push import conv_box_state_2_idx, EventType, BoxState
from ai_coach_domain.box_push.agent import (BoxPushAIAgent_PO_Indv,
                                            BoxPushAIAgent_Team2,
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
                        "Interactive Tutorial")
    dict_objs[obj.name] = obj

    return dict_objs

  def _get_drawing_order(self, user_game_data: Exp1UserData = None):
    return [self.BTN_TUTORIAL_START]

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
    drawing_order = [self.GAME_BORDER]
    drawing_order.append(co.BTN_START)
    drawing_order = drawing_order + co.ACTION_BUTTONS
    drawing_order.append(co.BTN_SELECT)
    drawing_order.append(self.TEXT_SCORE)

    drawing_order.append(self.SPOTLIGHT)

    drawing_order.append(self.TEXT_INSTRUCTION)
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
    drawing_order = [self.GAME_BORDER]
    drawing_order.append(co.BTN_START)
    drawing_order = drawing_order + co.ACTION_BUTTONS
    drawing_order.append(co.BTN_SELECT)
    drawing_order.append(self.TEXT_SCORE)

    drawing_order.append(self.TEXT_INSTRUCTION)
    drawing_order.append(co.BTN_PREV)
    drawing_order.append(co.BTN_NEXT)

    return drawing_order

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("At the start of each task, " +
            "you will see the screen shown on the left. " +
            "Click the “Start” button to begin the task.")


class CanvasPageTutorialBase(BoxPushV2GamePage):
  CLICKED_BTNS = "clicked_btn"

  def __init__(self,
               domain_type,
               manual_latent_selection,
               game_map,
               auto_prompt: bool = True,
               prompt_on_change: bool = True) -> None:
    super().__init__(domain_type, manual_latent_selection, game_map,
                     auto_prompt, prompt_on_change, 5)

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    game = user_game_data.get_game_ref()
    agent1 = InteractiveAgent()
    agent2 = InteractiveAgent()
    game.set_autonomous_agent(agent1, agent2)

    PICKUP_BOX = 1 if self._DOMAIN_TYPE == EDomainType.Movers else 0
    game.event_input(self._AGENT1, EventType.SET_LATENT, ("pickup", PICKUP_BOX))
    user_game_data.data[Exp1UserData.SELECT] = False
    user_game_data.data[Exp1UserData.SHOW_LATENT] = True
    user_game_data.data[Exp1UserData.PARTIAL_OBS] = False

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_game_data)

    btn_prev, btn_next = self._get_btn_prev_next(False, False)
    dict_objs[btn_prev.name] = btn_prev
    dict_objs[btn_next.name] = btn_next

    return dict_objs

  def _get_drawing_order(self, user_game_data: Exp1UserData):
    drawing_order = []
    drawing_order.append(self.GAME_BORDER)

    dict_game = user_game_data.get_game_ref().get_env_info()

    drawing_order = (drawing_order +
                     self._game_scene_names(dict_game, user_game_data))
    drawing_order = (drawing_order +
                     self._game_overlay_names(dict_game, user_game_data))
    drawing_order = drawing_order + co.ACTION_BUTTONS
    drawing_order.append(co.BTN_SELECT)

    drawing_order.append(self.TEXT_SCORE)

    drawing_order.append(self.SPOTLIGHT)
    drawing_order.append(self.TEXT_INSTRUCTION)
    drawing_order.append(co.BTN_PREV)
    drawing_order.append(co.BTN_NEXT)

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
  def __init__(self, domain_type, game_map) -> None:
    super().__init__(domain_type, False, game_map, False, False)

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    user_game_data.data[self.CLICKED_BTNS] = set()
    user_game_data.data[Exp1UserData.SELECT] = False
    user_game_data.data[Exp1UserData.PARTIAL_OBS] = False
    game = user_game_data.get_game_ref()
    game.event_input(self._AGENT1, EventType.SET_LATENT, None)

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_game_data)

    dict_objs[co.BTN_NEXT].disable = True

    obj = dict_objs[co.BTN_STAY]  # type: co.JoystickStay
    obj = self._get_spotlight(*obj.pos, int(obj.width * 1.7))
    dict_objs[obj.name] = obj

    return dict_objs

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("During the task, you control the human player. " +
            "You can move the human player by clicking the motion buttons. " +
            "Once you have pressed all five buttons " +
            "(left, right, up, down, and wait), " +
            "please click on the “Next” button to continue.")

  def button_clicked(self, user_game_data: Exp1UserData, clicked_btn: str):
    if clicked_btn in co.JOYSTICK_BUTTONS:
      user_game_data.data[self.CLICKED_BTNS].add(clicked_btn)

    return super().button_clicked(user_game_data, clicked_btn)

  def _get_button_commands(self, clicked_btn, user_data: Exp1UserData):
    if clicked_btn in co.JOYSTICK_BUTTONS:
      return {"delete": [self.SPOTLIGHT]}

    return None

  def _get_updated_drawing_objects(
      self,
      user_data: Exp1UserData,
      dict_prev_game: Mapping[str,
                              Any] = None) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_updated_drawing_objects(user_data, dict_prev_game)

    btn_prev, btn_next = self._get_btn_prev_next(False, True)
    dict_objs[btn_prev.name] = btn_prev
    dict_objs[btn_next.name] = btn_next

    clicked_btns = user_data.data[self.CLICKED_BTNS]  # type: set
    if len(clicked_btns) != 5:
      for obj_name in clicked_btns:
        obj = dict_objs[obj_name]  # type: co.JoystickObject
        obj.fill_color = "LightGreen"
    else:
      obj = dict_objs[btn_next.name]  # type: co.ButtonRect
      obj.disable = False

    return dict_objs


class CanvasPageInvalidAction(CanvasPageTutorialBase):
  def __init__(self, domain_type, game_map) -> None:
    super().__init__(domain_type, False, game_map, False, False)

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("If you take an invalid action (e.g., try to move into a wall), " +
            "the human player will just vibrate on the spot.")


class CanvasPageJoystickShort(CanvasPageTutorialBase):
  def __init__(self, domain_type, game_map) -> None:
    super().__init__(domain_type, False, game_map, False, False)

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    game = user_game_data.get_game_ref()
    game.event_input(self._AGENT1, EventType.SET_LATENT, None)

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("During the task, you control the human player. " +
            "You can move the human player similar to the previous task. " +
            "If you take an invalid action (e.g., try to move into a wall), " +
            "the human player will just vibrate on the spot.")


class CanvasPageOnlyHuman(CanvasPageTutorialBase):
  def __init__(self, domain_type, game_map) -> None:
    super().__init__(domain_type, False, game_map, False, False)

  def _get_instruction(self, user_game_data: Exp1UserData):
    return (
        "While the success of the task depends on both you and the robot, " +
        "you cannot control the robot. You can only control the human player. "
        + "The robot moves autonomously.")


class CanvasPageGoToTarget(CanvasPageTutorialBase):
  def __init__(self, domain_type, game_map) -> None:
    super().__init__(domain_type, False, game_map, False, False)

  def _get_instruction(self, user_game_data: Exp1UserData):
    object_type = ("box"
                   if self._DOMAIN_TYPE == EDomainType.Movers else "trash bag")
    return (
        "The red circle indicates your current destination and provides" +
        " you a hint on where to move next. Please move to the " + object_type +
        " (using the motion buttons) and try to pick it. The pick button will" +
        " be available only when you are at the correct destination.")

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_game_data)
    dict_objs[co.BTN_NEXT].disable = True

    return dict_objs

  def _on_action_taken(self, user_game_data: Exp1UserData,
                       dict_prev_game: Mapping[str, Any],
                       tuple_actions: Sequence[Any]):
    super()._on_action_taken(user_game_data, dict_prev_game, tuple_actions)

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
  def __init__(self, domain_type, game_map) -> None:
    super().__init__(domain_type, False, game_map, False, False)

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    game = user_game_data.get_game_ref()

    PICKUP_BOX = 1 if self._DOMAIN_TYPE == EDomainType.Movers else 0
    game.a1_pos = game.boxes[PICKUP_BOX]
    game.current_step = user_game_data.data[Exp1UserData.SCORE]
    game.event_input(self._AGENT1, EventType.SET_LATENT, ("pickup", PICKUP_BOX))

    user_game_data.data[self.CLICKED_BTNS] = set()

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_game_data)

    obj = dict_objs[co.BTN_PICK_UP]  # type: co.ButtonRect
    obj = self._get_spotlight(*obj.pos, int(obj.size[0] * 0.6))
    dict_objs[obj.name] = obj

    dict_objs[co.BTN_NEXT].disable = True

    objs = self._get_btn_actions(True, True, True, True, True, False, True,
                                 True)
    for obj in objs:
      dict_objs[obj.name] = obj

    return dict_objs

  def _get_instruction(self, user_game_data: Exp1UserData):
    if self._DOMAIN_TYPE == EDomainType.Movers:
      return ("Now, please pick it up using the (pick button). " +
              "You will notice that you cannot pick up the box alone. " +
              "You have to pick it up together with the robot.")
    else:
      return ("Now, please pick it up using the (pick button). " +
              "You will notice that you can pick up the trash bag alone. " +
              "You don't need to wait for the robot.")

  def _on_action_taken(self, user_game_data: Exp1UserData,
                       dict_prev_game: Mapping[str, Any],
                       tuple_actions: Sequence[Any]):
    super()._on_action_taken(user_game_data, dict_prev_game, tuple_actions)

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

    return super().button_clicked(user_game_data, clicked_btn)

  def _get_button_commands(self, clicked_btn, user_data: Exp1UserData):
    if clicked_btn == co.BTN_PICK_UP:
      return {"delete": [self.SPOTLIGHT]}

    return None

  def _get_updated_drawing_objects(
      self,
      user_data: Exp1UserData,
      dict_prev_game: Mapping[str,
                              Any] = None) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_updated_drawing_objects(user_data, dict_prev_game)

    if co.BTN_PICK_UP not in user_data.data[self.CLICKED_BTNS]:
      objs = self._get_btn_actions(True, True, True, True, True, False, True,
                                   True)
      for obj in objs:
        dict_objs[obj.name] = obj
    else:
      _, btn_next = self._get_btn_prev_next(False, False)
      dict_objs[btn_next.name] = btn_next

    return dict_objs


class CanvasPagePickUpTarget(CanvasPageTutorialBase):
  def __init__(self, domain_type, game_map) -> None:
    super().__init__(domain_type, False, game_map, False, False)

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

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
    dict_objs = super()._get_init_drawing_objects(user_game_data)
    dict_objs[co.BTN_NEXT].disable = True

    return dict_objs

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("Please move to the box (red circle) and try to pick it again. " +
            "This time your robot teammate will also move to it. " +
            "Wait for your teammate at the box and pick it up together.")

  def _on_action_taken(self, user_game_data: Exp1UserData,
                       dict_prev_game: Mapping[str, Any],
                       tuple_actions: Sequence[Any]):
    super()._on_action_taken(user_game_data, dict_prev_game, tuple_actions)

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
  def __init__(self, domain_type, game_map) -> None:
    super().__init__(domain_type, False, game_map, False, False)

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    game = user_game_data.get_game_ref()
    agent1 = InteractiveAgent()

    if self._DOMAIN_TYPE == EDomainType.Movers:
      agent2 = BoxPushAIAgent_Team2(self._TEAMMATE_POLICY)
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
    else:
      game.a1_pos = game.boxes[PICKUP_BOX]
      game.box_states[PICKUP_BOX] = conv_box_state_2_idx(
          (BoxState.WithAgent1, None), len(game.drops))
      game.event_input(self._AGENT1, EventType.SET_LATENT, ("goal", 0))

  def _on_action_taken(self, user_game_data: Exp1UserData,
                       dict_prev_game: Mapping[str, Any],
                       tuple_actions: Sequence[Any]):
    super()._on_action_taken(user_game_data, dict_prev_game, tuple_actions)

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

    return ("After picking up the " + object_type +
            ", you need to drop it at the truck. " + "Please carry the " +
            object_type + " to the truck and drop it there.")

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_game_data)
    dict_objs[co.BTN_NEXT].disable = True

    return dict_objs


class CanvasPageRespawn(CanvasPageTutorialBase):
  def __init__(self, domain_type, game_map) -> None:
    super().__init__(domain_type, False, game_map, False, False)

  def init_user_data(self, user_game_data: Exp1UserData):
    # game no need to be initialized

    game = user_game_data.get_game_ref()
    game.set_autonomous_agent()
    game.event_input(self._AGENT1, EventType.SET_LATENT, None)
    user_game_data.data[Exp1UserData.ACTION_COUNT] = 0
    user_game_data.data[Exp1UserData.SELECT] = False

  def _get_instruction(self, user_game_data: Exp1UserData):
    object_type = ("box"
                   if self._DOMAIN_TYPE == EDomainType.Movers else "trash bag")
    return ("Well done! Note that you will be stepped out from the goal once " +
            " you drop the " + object_type + ". You cannot step on the goal " +
            "without " + object_type + ".")


class CanvasPageScore(CanvasPageTutorialBase):
  def __init__(self, domain_type, game_map) -> None:
    super().__init__(domain_type, False, game_map, False, False)

  def init_user_data(self, user_game_data: Exp1UserData):
    # game no need to be initialized

    user_game_data.data[Exp1UserData.ACTION_COUNT] = 0
    user_game_data.data[Exp1UserData.SELECT] = False

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_game_data)

    objs = self._get_btn_actions(True, True, True, True, True, True, True, True)
    for obj in objs:
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
  def __init__(self, domain_type, game_map) -> None:
    super().__init__(domain_type, False, game_map, False, False)

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    user_game_data.data[Exp1UserData.PARTIAL_OBS] = True

  def _get_instruction(self, user_game_data: Exp1UserData):
    return (
        "Unlike the tutorial so far, during the TEST sessions, " +
        "You cannot observe the full environment of the game. You can see " +
        "your robot teammate or objects only if they are " +
        "at your neighborhood.")


class CanvasPageTarget(CanvasPageTutorialBase):
  def __init__(self, domain_type, game_map) -> None:
    super().__init__(domain_type, False, game_map, False, False)

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    user_game_data.data[Exp1UserData.PARTIAL_OBS] = True

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_game_data)

    # obj = dict_objs[self.TEXT_INSTRUCTION]  # type: co.TextObject
    # x_cen = int(obj.pos[0] + 0.5 * obj.width)
    # y_cen = int(self.GAME_HEIGHT / 5)
    # radius = int(y_cen * 0.1)
    # dict_objs[self.SPOTLIGHT] = self._get_spotlight(x_cen, y_cen, radius)

    objs = self._get_btn_actions(True, True, True, True, True, True, True, True)
    for obj in objs:
      dict_objs[obj.name] = obj

    return dict_objs

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("In the TEST sessions, you will have to select your next target" +
            " using the \"Select Destination\" button. It is very important " +
            "to provide your destination instantly whenever you change your " +
            "target in your mind. Let's see how to do this! Please click on " +
            "the \"Next\" button to continue.")


class CanvasPageLatent(CanvasPageTutorialBase):
  def __init__(self, domain_type, game_map) -> None:
    super().__init__(domain_type, True, game_map, True, True)

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    user_game_data.data[Exp1UserData.PARTIAL_OBS] = True

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_game_data)
    obj = dict_objs[co.BTN_SELECT]  # type: co.ButtonRect
    obj = self._get_spotlight(*obj.pos, int(obj.size[0] * 0.6))
    dict_objs[obj.name] = obj

    dict_objs[co.BTN_NEXT].disable = True

    objs = self._get_btn_actions(True, True, True, True, True, True, True,
                                 False)
    for obj in objs:
      dict_objs[obj.name] = obj

    return dict_objs

  def _get_instruction(self, user_game_data: Exp1UserData):
    object_type = ("box"
                   if self._DOMAIN_TYPE == EDomainType.Movers else "trash bag")
    return ("First, click the “Select Destination” button. " +
            "Possible destinations are numbered and shown as an overlay. " +
            "Please click on your current destination (i.e., the " +
            object_type + " which you are planning to pick next).")

  def _get_button_commands(self, clicked_btn, user_data: Exp1UserData):
    if clicked_btn == co.BTN_SELECT:
      return {"delete": [self.SPOTLIGHT]}

    return None

  def button_clicked(self, user_game_data: Exp1UserData, clicked_btn: str):
    if self.is_sel_latent_btn(clicked_btn):
      user_game_data.go_to_next_page()

    return super().button_clicked(user_game_data, clicked_btn)


class CanvasPageSelResult(CanvasPageTutorialBase):
  def __init__(self, domain_type, game_map, is_2nd) -> None:
    super().__init__(domain_type, False, game_map, True, True)
    self._IS_2ND = is_2nd

  def init_user_data(self, user_game_data: Exp1UserData):
    # game no need to be initialized
    user_game_data.data[Exp1UserData.ACTION_COUNT] = 0
    user_game_data.data[Exp1UserData.SELECT] = False
    user_game_data.data[Exp1UserData.PARTIAL_OBS] = True

  def _get_instruction(self, user_game_data: Exp1UserData):
    if self._IS_2ND:
      return ("Great! As before, your choice is marked with the red circle " +
              "and you have selected your next destination.")
    else:
      return ("Well done! Now you can see your choice is marked with the red" +
              " circle and you have selected your next destination.")

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_game_data)

    objs = self._get_btn_actions(True, True, True, True, True, True, True, True)
    for obj in objs:
      dict_objs[obj.name] = obj

    return dict_objs


class CanvasPageSelPrompt(CanvasPageTutorialBase):
  def __init__(self, domain_type, game_map) -> None:
    super().__init__(domain_type, False, game_map, True, True)
    self._PROMPT_FREQ = 3

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    user_game_data.data[Exp1UserData.PARTIAL_OBS] = True
    game = user_game_data.get_game_ref()
    game.event_input(self._AGENT1, EventType.SET_LATENT, None)

  def _get_instruction(self, user_game_data: Exp1UserData):
    return (
        "We will also prompt the destination selection auto- matically " +
        "and periodically during the TEST sessions. Please move the human " +
        "player several steps. When the destination selection is prompted" +
        ", please click on your current destination.")

  def button_clicked(self, user_game_data: Exp1UserData, clicked_btn: str):
    if self.is_sel_latent_btn(clicked_btn):
      user_game_data.go_to_next_page()

    return super().button_clicked(user_game_data, clicked_btn)

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_game_data)

    dict_objs[co.BTN_NEXT].disable = True

    return dict_objs


class CanvasPageMiniGame(CanvasPageTutorialBase):
  def __init__(self, domain_type, game_map) -> None:
    super().__init__(domain_type, True, game_map, True, True)

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("Now, we are at the final step of the tutorial. Feel free to " +
            "interact with the interface and get familiar with the task. You" +
            " can also press the back button to revisit any of the previous " +
            "prompts.\n Once you are ready, please proceed to the TEST " +
            "sessions (using the button at the bottom of this page).")

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

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
    user_game_data.data[Exp1UserData.SELECT] = True

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
    dict_objs = super()._get_init_drawing_objects(user_game_data)

    dict_objs[co.BTN_NEXT].disable = True

    return dict_objs
