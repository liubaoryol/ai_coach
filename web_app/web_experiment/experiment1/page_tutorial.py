from typing import Mapping, Sequence, Any
import web_experiment.experiment1.page_base as pg
import web_experiment.experiment1.page_game as pgg
from web_experiment.experiment1.page_exp1_common import CanvasPageStart
import web_experiment.experiment1.canvas_objects as co
from web_experiment.models import db, User

from ai_coach_domain.box_push.agent import (BoxPushSimpleAgent,
                                            InteractiveAgent)
from ai_coach_domain.box_push import conv_box_state_2_idx


class CanvasPageTutorialStart(pg.CanvasPageBase):
  BTN_TUTORIAL_START = "btn_tutorial_start"

  def __init__(self, is_movers) -> None:
    super().__init__(False, False, False, False, is_movers)

  def _init_user_data(self, user_game_data: pg.UserGameData):
    super()._init_user_data(user_game_data)

  def _get_init_drawing_objects(self,
                                game_env=None,
                                flags: pg.GameFlags = None,
                                score: int = 0,
                                best_score: int = 9999):
    dict_objs = super()._get_init_drawing_objects(game_env, flags, score,
                                                  best_score)

    pos = (int(co.CANVAS_WIDTH / 2), int(co.CANVAS_HEIGHT / 2))
    size = (int(self.GAME_WIDTH / 2), int(self.GAME_HEIGHT / 5))
    obj = co.ButtonRect(self.BTN_TUTORIAL_START, pos, size, 30,
                        "Interactive Tutorial")
    dict_objs[obj.name] = obj

    return dict_objs

  def _get_drawing_order(self, game_env=None, flags: pg.GameFlags = None):
    return [self.BTN_TUTORIAL_START]

  def button_clicked(self, user_game_data: pg.UserGameData, clicked_btn):
    if clicked_btn == self.BTN_TUTORIAL_START:
      user_game_data.go_to_next_page()
      return None, None, None, None

    return super().button_clicked(user_game_data, clicked_btn)


class CanvasPageInstruction(CanvasPageStart):
  def _get_init_drawing_objects(self,
                                game_env=None,
                                flags: pg.GameFlags = None,
                                score: int = 0,
                                best_score: int = 9999):
    dict_objs = super()._get_init_drawing_objects(game_env, flags, score,
                                                  best_score)
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

  def _get_drawing_order(self, game_env=None, flags: pg.GameFlags = None):
    drawing_order = [self.GAME_BORDER]
    drawing_order.append(co.BTN_START)
    drawing_order = drawing_order + co.ACTION_BUTTONS
    drawing_order.append(co.BTN_SELECT)
    drawing_order.append(self.TEXT_SCORE)

    drawing_order.append(self.SPOTLIGHT)

    drawing_order.append(self.RECT_INSTRUCTION)
    drawing_order.append(self.TEXT_INSTRUCTION)
    drawing_order.append(co.BTN_PREV)
    drawing_order.append(co.BTN_NEXT)

    return drawing_order

  def _get_instruction(self, flags: pg.GameFlags):
    return ("Prompts will be shown here. Please read each prompt carefully. " +
            "Click the “Next” button to proceed and “Back” button to " +
            "go to the previous prompt.")


class CanvasPageTutorialGameStart(CanvasPageStart):
  def _get_init_drawing_objects(self,
                                game_env=None,
                                flags: pg.GameFlags = None,
                                score: int = 0,
                                best_score: int = 9999):
    dict_objs = super()._get_init_drawing_objects(game_env, flags, score,
                                                  best_score)
    obj = dict_objs[co.BTN_START]  # type: co.ButtonRect
    obj.disable = False  # enable start btn

    objs = self._get_btn_prev_next(False, True)
    for obj in objs:
      dict_objs[obj.name] = obj

    return dict_objs

  def _get_drawing_order(self, game_env=None, flags: pg.GameFlags = None):
    drawing_order = [self.GAME_BORDER]
    drawing_order.append(co.BTN_START)
    drawing_order = drawing_order + co.ACTION_BUTTONS
    drawing_order.append(co.BTN_SELECT)
    drawing_order.append(self.TEXT_SCORE)

    drawing_order.append(self.RECT_INSTRUCTION)
    drawing_order.append(self.TEXT_INSTRUCTION)
    drawing_order.append(co.BTN_PREV)
    drawing_order.append(co.BTN_NEXT)

    return drawing_order

  def _get_instruction(self, flags: pg.GameFlags):
    return ("At the start of each task, " +
            "you will see the screen shown on the left. " +
            "Click the “Start” button to begin the task.")


class CanvasPageTutorialBase(pgg.CanvasPageGame):
  def __init__(self,
               is_movers,
               manual_latent_selection,
               game_map,
               auto_prompt: bool = True,
               prompt_on_change: bool = True) -> None:
    super().__init__(is_movers, manual_latent_selection, game_map, auto_prompt,
                     prompt_on_change, 3)

  def _init_user_data(self, user_game_data: pg.UserGameData):
    super()._init_user_data(user_game_data)

    agent1 = InteractiveAgent()
    agent2 = InteractiveAgent()
    user_game_data.game.set_autonomous_agent(agent1, agent2)
    user_game_data.game.event_input(self._AGENT1, pgg.EventType.SET_LATENT,
                                    ("pickup", 2))
    user_game_data.flags.select = False

  def _get_init_drawing_objects(
      self, game_env, flags: pg.GameFlags, score: int,
      best_score: int) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(game_env, flags, score,
                                                  best_score)

    btn_prev, btn_next = self._get_btn_prev_next(False, False)
    dict_objs[btn_prev.name] = btn_prev
    dict_objs[btn_next.name] = btn_next

    return dict_objs

  def _get_drawing_order(self, game_env, flags: pg.GameFlags):
    drawing_order = []
    drawing_order.append(self.GAME_BORDER)

    drawing_order = drawing_order + self._game_scene_names(
        game_env, self._IS_MOVERS, flags)
    drawing_order = drawing_order + self._game_overlay_names(
        game_env, flags, not flags.select)
    drawing_order = drawing_order + co.ACTION_BUTTONS
    drawing_order.append(co.BTN_SELECT)

    drawing_order.append(self.TEXT_SCORE)

    drawing_order.append(self.SPOTLIGHT)
    drawing_order.append(self.RECT_INSTRUCTION)
    drawing_order.append(self.TEXT_INSTRUCTION)
    drawing_order.append(co.BTN_PREV)
    drawing_order.append(co.BTN_NEXT)

    return drawing_order

  def _on_game_finished(self, user_game_data: pg.UserGameData):
    '''
    user_game_data: NOTE - values will be updated
    '''

    user_game_data.flags.done = True
    user_game_data.game.reset_game()
    user_game_data.score = user_game_data.game.current_step
    self._init_user_data(user_game_data)


class CanvasPageJoystick(CanvasPageTutorialBase):
  def __init__(self, is_movers, game_map) -> None:
    super().__init__(is_movers, False, game_map, False, False)

  def _init_user_data(self, user_game_data: pg.UserGameData):
    super()._init_user_data(user_game_data)

    user_game_data.flags.clicked_btn.clear()

  def _get_init_drawing_objects(
      self, game_env, flags: pg.GameFlags, score: int,
      best_score: int) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(game_env, flags, score,
                                                  best_score)

    dict_objs[co.BTN_NEXT].disable = True

    obj = dict_objs[co.BTN_STAY]  # type: co.JoystickStay
    obj = self._get_spotlight(*obj.pos, int(obj.width * 1.7))
    dict_objs[obj.name] = obj

    return dict_objs

  def _get_instruction(self, flags: pg.GameFlags):
    return ("During the task, you control the human player. " +
            "You can move the human player by clicking the motion buttons. " +
            "Once you have pressed all five buttons " +
            "(left, right, up, down, and wait), " +
            "please click on the “Next” button to continue.")

  def button_clicked(self, user_game_data: pg.UserGameData, clicked_btn):
    if clicked_btn in co.JOYSTICK_BUTTONS:
      user_game_data.flags.clicked_btn.add(clicked_btn)

    return super().button_clicked(user_game_data, clicked_btn)

  def _get_button_commands(self, clicked_btn, flags: pg.GameFlags):
    if clicked_btn in co.JOYSTICK_BUTTONS:
      return {"delete": [self.SPOTLIGHT]}

    return None

  def _get_updated_drawing_objects(
      self,
      game_env,
      flags: pg.GameFlags,
      best_score: int = 9999,
      game_updated: bool = True) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_updated_drawing_objects(game_env, flags,
                                                     best_score, game_updated)

    btn_prev, btn_next = self._get_btn_prev_next(False, True)
    dict_objs[btn_prev.name] = btn_prev
    dict_objs[btn_next.name] = btn_next

    if len(flags.clicked_btn) != 5:
      for obj_name in flags.clicked_btn:
        obj = dict_objs[obj_name]  # type: co.JoystickObject
        obj.fill_color = "LightGreen"
    else:
      obj = dict_objs[btn_next.name]  # type: co.ButtonRect
      obj.disable = False

    return dict_objs


class CanvasPageInvalidAction(CanvasPageTutorialBase):
  def __init__(self, is_movers, game_map) -> None:
    super().__init__(is_movers, False, game_map, False, False)

  def _get_instruction(self, flags: pg.GameFlags):
    return ("If you take an invalid action (e.g., try to move into a wall), " +
            "the human player will just vibrate on the spot.")


class CanvasPageJoystickShort(CanvasPageTutorialBase):
  def __init__(self, is_movers, game_map) -> None:
    super().__init__(is_movers, False, game_map, False, False)

  def _get_instruction(self, flags: pg.GameFlags):
    return ("During the task, you control the human player. " +
            "You can move the human player similar to the previous task. " +
            "If you take an invalid action (e.g., try to move into a wall), " +
            "the human player will just vibrate on the spot.")


class CanvasPageOnlyHuman(CanvasPageTutorialBase):
  def __init__(self, is_movers, game_map) -> None:
    super().__init__(is_movers, False, game_map, False, False)

  def _get_instruction(self, flags: pg.GameFlags):
    return (
        "While the success of the task depends on both you and the robot, " +
        "you cannot control the robot. You can only control the human player. "
        + "The robot moves autonomously.")


class CanvasPageGoToTarget(CanvasPageTutorialBase):
  def __init__(self, is_movers, game_map) -> None:
    super().__init__(is_movers, False, game_map, False, False)

  def _get_instruction(self, flags: pg.GameFlags):
    object_type = "box" if self._IS_MOVERS else "trash bag"
    return (
        "The red circle indicates your current destination and provides" +
        " you a hint on where to move next. Please move to the " + object_type +
        " (using the motion buttons) and try to pick it. The pick button will" +
        " be available only when you are at the correct destination.")

  def _get_init_drawing_objects(
      self, game_env, flags: pg.GameFlags, score: int,
      best_score: int) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(game_env, flags, score,
                                                  best_score)
    dict_objs[co.BTN_NEXT].disable = True

    return dict_objs

  def _on_action_taken(self, user_cur_game_data: pg.UserGameData,
                       dict_prev_game: Mapping[str, Any],
                       tuple_actions: Sequence[Any]):
    super()._on_action_taken(user_cur_game_data, dict_prev_game, tuple_actions)

    game_env = user_cur_game_data.game.get_env_info()
    user_cur_game_data.score = game_env["current_step"]

    a1_latent = game_env["a1_latent"]
    if a1_latent is not None and a1_latent[0] == "pickup":
      box_coord = game_env["boxes"][a1_latent[1]]
      a1_pos = game_env["a1_pos"]
      if a1_pos == box_coord:
        user_cur_game_data.go_to_next_page()


class CanvasPagePickUpTarget(CanvasPageTutorialBase):
  def __init__(self, is_movers, game_map) -> None:
    super().__init__(is_movers, False, game_map, False, False)

  def _init_user_data(self, user_game_data: pg.UserGameData):
    super()._init_user_data(user_game_data)

    agent1 = InteractiveAgent()
    if self._IS_MOVERS:
      agent2 = BoxPushSimpleAgent(self._AGENT2, self._GAME_MAP["x_grid"],
                                  self._GAME_MAP["y_grid"],
                                  self._GAME_MAP["boxes"],
                                  self._GAME_MAP["goals"],
                                  self._GAME_MAP["walls"],
                                  self._GAME_MAP["drops"])
    else:
      agent2 = InteractiveAgent()

    user_game_data.game.set_autonomous_agent(agent1, agent2)

    PICKUP_BOX = 2
    user_game_data.game.a1_pos = user_game_data.game.boxes[PICKUP_BOX]
    user_game_data.game.current_step = user_game_data.score
    user_game_data.game.event_input(self._AGENT1, pgg.EventType.SET_LATENT,
                                    ("pickup", PICKUP_BOX))
    if self._IS_MOVERS:
      user_game_data.game.event_input(self._AGENT2, pgg.EventType.SET_LATENT,
                                      ("pickup", PICKUP_BOX))
    user_game_data.flags.clicked_btn.clear()

  def _get_init_drawing_objects(
      self, game_env, flags: pg.GameFlags, score: int,
      best_score: int) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(game_env, flags, score,
                                                  best_score)

    obj = dict_objs[co.BTN_PICK_UP]  # type: co.ButtonRect
    obj = self._get_spotlight(*obj.pos, int(obj.size[0] * 0.6))
    dict_objs[obj.name] = obj

    dict_objs[co.BTN_NEXT].disable = True

    objs = self._get_btn_actions(True, True, True, True, True, False, True,
                                 True)
    for obj in objs:
      dict_objs[obj.name] = obj

    return dict_objs

  def _get_instruction(self, flags: pg.GameFlags):
    if self._IS_MOVERS:
      return ("Now, please pick it up using the (pick button). " +
              "You will notice that you cannot pick up the box alone. " +
              "You have to pick it up together with the robot.")
    else:
      return ("Now, please pick it up using the (pick button). " +
              "You will notice that you can pick up the trash bag alone. " +
              "You don't need to wait for the robot.")

  def _on_action_taken(self, user_cur_game_data: pg.UserGameData,
                       dict_prev_game: Mapping[str, Any],
                       tuple_actions: Sequence[Any]):
    super()._on_action_taken(user_cur_game_data, dict_prev_game, tuple_actions)

    game_env = user_cur_game_data.game.get_env_info()
    user_cur_game_data.score = game_env["current_step"]

    num_drops = len(game_env["drops"])
    num_goals = len(game_env["goals"])
    a1_box, _ = pg.get_holding_box_idx(game_env["box_states"], num_drops,
                                       num_goals)
    if a1_box >= 0:
      user_cur_game_data.go_to_next_page()

  def button_clicked(self, user_game_data: pg.UserGameData, clicked_btn):
    if clicked_btn == co.BTN_PICK_UP:
      user_game_data.flags.clicked_btn.add(co.BTN_PICK_UP)

    return super().button_clicked(user_game_data, clicked_btn)

  def _get_button_commands(self, clicked_btn, flags: pg.GameFlags):
    if clicked_btn == co.BTN_PICK_UP:
      return {"delete": [self.SPOTLIGHT]}

    return None

  def _get_updated_drawing_objects(
      self,
      game_env,
      flags: pg.GameFlags,
      best_score: int = 9999,
      game_updated: bool = True) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_updated_drawing_objects(game_env, flags,
                                                     best_score, game_updated)
    if co.BTN_PICK_UP not in flags.clicked_btn:
      objs = self._get_btn_actions(True, True, True, True, True, False, True,
                                   True)
      for obj in objs:
        dict_objs[obj.name] = obj

    return dict_objs


class CanvasPageGoToGoal(CanvasPageTutorialBase):
  def __init__(self, is_movers, game_map) -> None:
    super().__init__(is_movers, False, game_map, False, False)

  def _init_user_data(self, user_game_data: pg.UserGameData):
    super()._init_user_data(user_game_data)

    game = user_game_data.game
    agent1 = InteractiveAgent()
    if self._IS_MOVERS:
      agent2 = BoxPushSimpleAgent(self._AGENT2, self._GAME_MAP["x_grid"],
                                  self._GAME_MAP["y_grid"],
                                  self._GAME_MAP["boxes"],
                                  self._GAME_MAP["goals"],
                                  self._GAME_MAP["walls"],
                                  self._GAME_MAP["drops"])
    else:
      agent2 = InteractiveAgent()

    game.set_autonomous_agent(agent1, agent2)
    game.current_step = user_game_data.score
    PICKUP_BOX = 2
    if self._IS_MOVERS:
      game.a1_pos = game.boxes[PICKUP_BOX]
      game.a2_pos = game.boxes[PICKUP_BOX]
      game.box_states[PICKUP_BOX] = conv_box_state_2_idx(
          (pg.BoxState.WithBoth, None), len(game.drops))
      game.event_input(self._AGENT1, pgg.EventType.SET_LATENT, ("goal", 0))
      game.event_input(self._AGENT2, pgg.EventType.SET_LATENT, ("goal", 0))
    else:
      game.a1_pos = game.boxes[PICKUP_BOX]
      game.box_states[PICKUP_BOX] = conv_box_state_2_idx(
          (pg.BoxState.WithAgent1, None), len(game.drops))
      game.event_input(self._AGENT1, pgg.EventType.SET_LATENT, ("goal", 0))

  def _on_action_taken(self, user_cur_game_data: pg.UserGameData,
                       dict_prev_game: Mapping[str, Any],
                       tuple_actions: Sequence[Any]):
    super()._on_action_taken(user_cur_game_data, dict_prev_game, tuple_actions)

    user_cur_game_data.score = user_cur_game_data.game.current_step

    game_env = user_cur_game_data.game.get_env_info()
    num_drops = len(game_env["drops"])
    num_goals = len(game_env["goals"])
    a1_box, _ = pg.get_holding_box_idx(game_env["box_states"], num_drops,
                                       num_goals)
    if a1_box < 0:
      user_cur_game_data.go_to_next_page()

  def _get_instruction(self, flags: pg.GameFlags):
    object_type = "box" if self._IS_MOVERS else "trash bag"

    return ("After picking up the " + object_type +
            ", you need to drop it at the flag. " + "Please carry the " +
            object_type + " to the flag and drop it there.")

  def _get_init_drawing_objects(
      self, game_env, flags: pg.GameFlags, score: int,
      best_score: int) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(game_env, flags, score,
                                                  best_score)

    dict_objs[co.BTN_NEXT].disable = True

    return dict_objs


class CanvasPageScore(CanvasPageTutorialBase):
  def __init__(self, is_movers, game_map) -> None:
    super().__init__(is_movers, False, game_map, False, False)

  def _init_user_data(self, user_game_data: pg.UserGameData):
    # game no need to be initialized
    user_game_data.flags.action_count = 0
    user_game_data.flags.select = False
    user_game_data.flags.aligned_a2_action = False

  def _get_init_drawing_objects(
      self, game_env, flags: pg.GameFlags, score: int,
      best_score: int) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(game_env, flags, score,
                                                  best_score)

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

  def _get_instruction(self, flags: pg.GameFlags):
    return ("Well done! You might have noticed that as you were doing the" +
            " task the \"Time Taken\" counter (shown below) was increasing. " +
            "Your goal is to complete the task as fast as possible " +
            "(i.e., with the least amount of time taken).")


class CanvasPageTrapped(CanvasPageTutorialBase):
  def __init__(self, is_movers, game_map) -> None:
    super().__init__(is_movers, False, game_map, False, False)

  def _init_user_data(self, user_game_data: pg.UserGameData):
    super()._init_user_data(user_game_data)

    user_game_data.flags.aligned_a2_action = True if self._IS_MOVERS else False

    game = user_game_data.game

    agent1 = InteractiveAgent()
    agent2 = InteractiveAgent()
    game.set_autonomous_agent(agent1, agent2)
    # make scenario
    TRAP_BOX = 1
    if self._IS_MOVERS:
      game.box_states[TRAP_BOX] = conv_box_state_2_idx(
          (pg.BoxState.WithBoth, None), len(game.drops))
      game.a1_pos = game.boxes[TRAP_BOX]
      game.a2_pos = game.boxes[TRAP_BOX]
    else:
      game.a1_pos = game.boxes[TRAP_BOX]
      game.box_states[TRAP_BOX] = conv_box_state_2_idx(
          (pg.BoxState.WithAgent1, None), len(game.drops))
      A2_HOLDING_BOX = 0
      game.box_states[A2_HOLDING_BOX] = conv_box_state_2_idx(
          (pg.BoxState.WithAgent2, None), len(game.drops))
      game.a2_pos = game.boxes[A2_HOLDING_BOX]

    game.event_input(self._AGENT1, pgg.EventType.SET_LATENT, ("goal", 0))

  def _get_instruction(self, flags: pg.GameFlags):
    object_type = "box" if self._IS_MOVERS else "trash bag"
    return ("Let's look at some other aspects of the task. When you are " +
            "holding a" + object_type + ", you cannot move on top of another " +
            object_type + ". Try moving to the goal. You will notice that you" +
            "are stuck! Please click on the \"Next\" button to continue.")


class CanvasPageTargetHint(CanvasPageTutorialBase):
  def __init__(self, is_movers, game_map) -> None:
    super().__init__(is_movers, False, game_map, False, False)

  def _get_init_drawing_objects(
      self, game_env, flags: pg.GameFlags, score: int,
      best_score: int) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(game_env, flags, score,
                                                  best_score)

    obj = dict_objs[self.TEXT_INSTRUCTION]  # type: co.TextObject
    x_cen = int(obj.pos[0] + 0.5 * obj.width)
    y_cen = int(self.GAME_HEIGHT / 5)
    radius = int(y_cen * 0.1)
    dict_objs[self.SPOTLIGHT] = self._get_spotlight(x_cen, y_cen, radius)

    objs = self._get_btn_actions(True, True, True, True, True, True, True, True)
    for obj in objs:
      dict_objs[obj.name] = obj

    return dict_objs

  def _get_instruction(self, flags: pg.GameFlags):
    return ("During the PRACTICE sessions, you will be given hints on where " +
            "to move next. This will be done using the red circles shown " +
            "earlier. Please click on the \"Next\" button to continue.")


class CanvasPageTargetNoHint(CanvasPageTargetHint):
  def __init__(self, is_movers, game_map) -> None:
    super().__init__(is_movers, game_map)

  def _get_instruction(self, flags: pg.GameFlags):
    return ("In the TEST sessions, you will no longer be given hints. Instead" +
            ", you will have to select your next target using the \"Select" +
            "Destination\" button. Let's see how to do this! Please click on " +
            "the \"Next\" button to continue.")


class CanvasPageLatent(CanvasPageTutorialBase):
  def __init__(self, is_movers, game_map) -> None:
    super().__init__(is_movers, True, game_map, True, True)

  def _get_init_drawing_objects(
      self, game_env, flags: pg.GameFlags, score: int,
      best_score: int) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(game_env, flags, score,
                                                  best_score)
    obj = dict_objs[co.BTN_SELECT]  # type: co.ButtonRect
    obj = self._get_spotlight(*obj.pos, int(obj.size[0] * 0.6))
    dict_objs[obj.name] = obj

    dict_objs[co.BTN_NEXT].disable = True

    objs = self._get_btn_actions(True, True, True, True, True, True, True,
                                 False)
    for obj in objs:
      dict_objs[obj.name] = obj

    return dict_objs

  def _get_instruction(self, flags: pg.GameFlags):
    object_type = "box" if self._IS_MOVERS else "trash bag"
    return ("First, click the “Select Destination” button. " +
            "Possible destinations are numbered and shown as an overlay. " +
            "Please click on your current destination (i.e., the " +
            object_type + " which you are planning to pick next).")

  def _get_button_commands(self, clicked_btn, flags: pg.GameFlags):
    if clicked_btn == co.BTN_SELECT:
      return {"delete": [self.SPOTLIGHT]}

    return None

  def button_clicked(self, user_game_data: pg.UserGameData, clicked_btn):
    if co.is_sel_latent_btn(clicked_btn):
      user_game_data.go_to_next_page()

    return super().button_clicked(user_game_data, clicked_btn)


class CanvasPageSelResult(CanvasPageTutorialBase):
  def __init__(self, is_movers, game_map, is_2nd) -> None:
    super().__init__(is_movers, False, game_map, True, True)
    self._IS_2ND = is_2nd

  def _init_user_data(self, user_game_data: pg.UserGameData):
    # game no need to be initialized
    user_game_data.flags.action_count = 0
    user_game_data.flags.select = False
    user_game_data.flags.aligned_a2_action = False

  def _get_instruction(self, flags: pg.GameFlags):
    if self._IS_2ND:
      return ("Great! As before, your choice is marked with the red circle " +
              "and you have selected your next destination.")
    else:
      return ("Well done! Now you can see your choice is marked with the red" +
              " circle and you have selected your next destination.")

  def _get_init_drawing_objects(
      self, game_env, flags: pg.GameFlags, score: int,
      best_score: int) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(game_env, flags, score,
                                                  best_score)

    objs = self._get_btn_actions(True, True, True, True, True, True, True, True)
    for obj in objs:
      dict_objs[obj.name] = obj

    return dict_objs


class CanvasPageSelPrompt(CanvasPageTutorialBase):
  def __init__(self, is_movers, game_map) -> None:
    super().__init__(is_movers, False, game_map, True, True)

  def _get_instruction(self, flags: pg.GameFlags):
    return ("We will also prompt the destination selection auto-matically  " +
            "and periodically during the TEST sessions. Please move the human" +
            "player several steps. When the destination selection is prompted" +
            ", please click on your current destination.")

  def button_clicked(self, user_game_data: pg.UserGameData, clicked_btn):
    if co.is_sel_latent_btn(clicked_btn):
      user_game_data.go_to_next_page()

    return super().button_clicked(user_game_data, clicked_btn)

  def _get_init_drawing_objects(
      self, game_env, flags: pg.GameFlags, score: int,
      best_score: int) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(game_env, flags, score,
                                                  best_score)

    dict_objs[co.BTN_NEXT].disable = True

    return dict_objs


class CanvasPageMiniGame(CanvasPageTutorialBase):
  def __init__(self, is_movers, game_map) -> None:
    super().__init__(is_movers, True, game_map, True, True)

  def _get_instruction(self, flags: pg.GameFlags):
    return ("Now, we are at the final step of the tutorial. Feel free to " +
            "interact with the interface and get familiar with the task. You" +
            " can also press the back button to revisit any of the previous " +
            "prompts.\n Once you are ready, please proceed to the PRACTICE " +
            "sessions (using the button at the bottom of this page).")

  def _init_user_data(self, user_game_data: pg.UserGameData):
    super()._init_user_data(user_game_data)

    game = user_game_data.game
    agent1 = InteractiveAgent()
    agent2 = BoxPushSimpleAgent(self._AGENT2, self._GAME_MAP["x_grid"],
                                self._GAME_MAP["y_grid"],
                                self._GAME_MAP["boxes"],
                                self._GAME_MAP["goals"],
                                self._GAME_MAP["walls"],
                                self._GAME_MAP["drops"])
    game.set_autonomous_agent(agent1, agent2)
    game.event_input(self._AGENT1, pgg.EventType.SET_LATENT, ("pickup", 0))

    # set task done
    if not getattr(user_game_data.user, user_game_data.session_name):
      user_game_data.user = User.query.filter_by(
          userid=user_game_data.user.userid).first()
      setattr(user_game_data.user, user_game_data.session_name, True)
      db.session.commit()

  def _get_init_drawing_objects(
      self, game_env, flags: pg.GameFlags, score: int,
      best_score: int) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(game_env, flags, score,
                                                  best_score)

    dict_objs[co.BTN_NEXT].disable = True

    return dict_objs
