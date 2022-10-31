from typing import Mapping, Any, Sequence
from web_experiment.exp_common.page_base import Exp1UserData
from web_experiment.exp_common.page_rescue_v2_game import RescueV2GamePage
import web_experiment.exp_common.canvas_objects as co
from web_experiment.define import ExpType
from web_experiment.models import ExpDataCollection, ExpIntervention, db
from ai_coach_domain.agent import InteractiveAgent
from ai_coach_domain.rescue_v2 import E_EventType
from ai_coach_domain.rescue_v2.agent import AIAgent_Rescue_PartialObs

MAX_STEP = str(15)


class RescueV2TutorialBase(RescueV2GamePage):
  CLICKED_BTNS = "clicked_btn"

  def __init__(self,
               manual_latent_selection,
               auto_prompt: bool = True,
               prompt_on_change: bool = True) -> None:
    super().__init__(manual_latent_selection, auto_prompt, prompt_on_change, 5)

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    game = user_game_data.get_game_ref()
    agent1 = InteractiveAgent()
    agent2 = InteractiveAgent()
    agent3 = InteractiveAgent()
    game.set_autonomous_agent(agent1, agent2, agent3)

    TARGET = 0
    game.event_input(self._AGENT1, E_EventType.Set_Latent, TARGET)
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
    drawing_order = drawing_order + self.ACTION_BUTTONS
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


class RescueV2TutorialActions(RescueV2TutorialBase):

  def __init__(self) -> None:
    super().__init__(False, False, False)

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    user_game_data.data[self.CLICKED_BTNS] = set()
    user_game_data.data[Exp1UserData.SELECT] = False
    user_game_data.data[Exp1UserData.PARTIAL_OBS] = False
    game = user_game_data.get_game_ref()
    game.event_input(self._AGENT1, E_EventType.Set_Latent, None)

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_game_data)

    dict_objs[co.BTN_NEXT].disable = True

    obj = dict_objs[self.STAY]  # type: co.Circle
    pos = (obj.pos[0], obj.pos[1])
    obj = self._get_spotlight(*pos, int(obj.radius * 3))
    dict_objs[obj.name] = obj

    return dict_objs

  def _get_instruction(self, user_game_data: Exp1UserData):
    return (
        "During the task, you control the police car. You can move the " +
        "police car by clicking the move buttons on the right. Please " +
        "investigate how each move buttons work. You will see the direction " +
        "of the move button changes according to your location. Once you " +
        "get familiar with them, please click on the \"Next\" button.")

  def button_clicked(self, user_game_data: Exp1UserData, clicked_btn: str):
    if clicked_btn in self.ACTION_BUTTONS:
      user_game_data.data[self.CLICKED_BTNS].add(clicked_btn)

    return super().button_clicked(user_game_data, clicked_btn)

  def _get_button_commands(self, clicked_btn, user_data: Exp1UserData):
    commands = super()._get_button_commands(clicked_btn, user_data)
    if clicked_btn in self.ACTION_BUTTONS:
      if commands is not None:
        commands["delete"] = commands.get("delete", []) + [self.SPOTLIGHT]
      else:
        commands = {"delete": [self.SPOTLIGHT]}

    return commands

  def _get_updated_drawing_objects(
      self,
      user_data: Exp1UserData,
      dict_prev_game: Mapping[str,
                              Any] = None) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_updated_drawing_objects(user_data, dict_prev_game)

    _, btn_next = self._get_btn_prev_next(False, True)
    dict_objs[btn_next.name] = btn_next

    clicked_btns = user_data.data[self.CLICKED_BTNS]  # type: set
    if len(clicked_btns) != 0:
      obj = dict_objs[btn_next.name]  # type: co.ButtonRect
      obj.disable = False

    return dict_objs


class RescueV2TutorialPlain(RescueV2TutorialBase):

  def __init__(self) -> None:
    super().__init__(False, False, False)

  def init_user_data(self, user_game_data: Exp1UserData):
    user_game_data.data[Exp1UserData.GAME_DONE] = False
    user_game_data.data[Exp1UserData.SELECT] = False
    user_game_data.data[Exp1UserData.ACTION_COUNT] = 0
    user_game_data.data[Exp1UserData.SHOW_LATENT] = True
    user_game_data.data[Exp1UserData.PARTIAL_OBS] = False

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ""

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_game_data)
    objs = self._get_btn_actions(user_game_data.get_game_ref().get_env_info(),
                                 True, True, True)
    for obj in objs:
      dict_objs[obj.name] = obj
    return dict_objs


class RescueV2TutorialOverallGoal(RescueV2TutorialPlain):

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("Your goal is to rescue people as many as possible in " + MAX_STEP +
            " steps. " +
            "You can rescue each group of people by resolving the situations " +
            "at the places marked with \"yellow sign\".")


class RescueV2TutorialOnlyHuman(RescueV2TutorialPlain):

  def _get_instruction(self, user_game_data: Exp1UserData):
    return (
        "While the success of the task depends on both you (police car) " +
        "and the fire engine, you cannot control the fire engine. You can " +
        "only control the police car. The fire engine moves autonomously.")


class RescueV2TutorialSimpleTarget(RescueV2TutorialBase):

  def __init__(self) -> None:
    super().__init__(False, False, False)

  def _get_instruction(self, user_game_data: Exp1UserData):
    return (
        "The red circle indicates your current destination. Please move to " +
        "the circled place and try to rescue people. The \"Rescue\" button will"
        + " be available only when you are at the correct destination.")

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
    if a1_latent is not None:
      wstate = game_env["work_states"][a1_latent]
      if wstate == 0:
        user_game_data.go_to_next_page()


class RescueV2TutorialResolvedAlone(RescueV2TutorialPlain):

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("Well done! You will see the \"yellow sign\" at City Hall" +
            " is now disappeared and one person is rescued. Similarly, " +
            "people at the Campsite can also be rescued by just one player.")


class RescueV2TutorialComplexTarget(RescueV2TutorialBase):

  def __init__(self) -> None:
    super().__init__(False, False, False)

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    game = user_game_data.get_game_ref()

    TARGET_BOTH = 1
    game.event_input(self._AGENT1, E_EventType.Set_Latent, TARGET_BOTH)

    user_game_data.data[self.CLICKED_BTNS] = set()

  def _get_instruction(self, user_game_data: Exp1UserData):
    return (
        "To rescue people in the mall, you should repair one of the bridges. " +
        "However, to repair the bridge, both you (police car) and the fire " +
        "engine should work together. Please move to the bridge circled in " +
        "red and click \"Rescue\" button. You will see you can't resolve it " +
        "on your own.")

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_game_data)
    dict_objs[co.BTN_NEXT].disable = True

    return dict_objs

  def _get_updated_drawing_objects(
      self,
      user_data: Exp1UserData,
      dict_prev_game: Mapping[str,
                              Any] = None) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_updated_drawing_objects(user_data, dict_prev_game)

    _, btn_next = self._get_btn_prev_next(False, True)
    dict_objs[btn_next.name] = btn_next

    clicked_btns = user_data.data[self.CLICKED_BTNS]  # type: set
    if len(clicked_btns) != 0:
      obj = dict_objs[btn_next.name]  # type: co.ButtonRect
      obj.disable = False

    return dict_objs

  def button_clicked(self, user_game_data: Exp1UserData, clicked_btn: str):
    if clicked_btn == self.RESCUE:
      user_game_data.data[self.CLICKED_BTNS].add(clicked_btn)

    return super().button_clicked(user_game_data, clicked_btn)


class RescueV2TutorialComplexTargetTogether(RescueV2TutorialBase):

  def __init__(self) -> None:
    super().__init__(False, False, False)

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    game = user_game_data.get_game_ref()
    init_states = ([1] * len(self._GAME_MAP["work_locations"]),
                   self._GAME_MAP["a1_init"], self._GAME_MAP["a2_init"],
                   self._GAME_MAP["a3_init"])
    agent2 = AIAgent_Rescue_PartialObs(init_states, self._AGENT2,
                                       self._TEAMMATE_POLICY_1)
    agent3 = AIAgent_Rescue_PartialObs(init_states, self._AGENT3,
                                       self._TEAMMATE_POLICY_2)
    game.set_autonomous_agent(agent2=agent2, agent3=agent3)

    TARGET_BOTH = 1
    game.event_input(self._AGENT1, E_EventType.Set_Latent, TARGET_BOTH)
    game.event_input(self._AGENT2, E_EventType.Set_Latent, TARGET_BOTH)

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("Please move to the red circle and try to resolve it again. " +
            "This time the fire engine will also come to there. Wait for the" +
            " fire engine at the circled location and rescue people together.")

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
    if a1_latent is not None:
      wstate = game_env["work_states"][a1_latent]
      if wstate == 0:
        user_game_data.go_to_next_page()


class RescueV2TutorialResolvedTogether(RescueV2TutorialPlain):

  def _get_instruction(self, user_game_data: Exp1UserData):
    return (
        "Well done! You will see the bridge is repaired and the \"yellow sign\""
        + " is now disappeared. Note that if one of the bridges is repaired, " +
        "you don't need to repair the other one.")


class RescueV2TutorialScore(RescueV2TutorialPlain):

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_game_data)

    obj = dict_objs[self.TEXT_SCORE]  # type: co.TextObject
    x_cen = int(obj.pos[0] + 0.95 * obj.width)
    y_cen = int(obj.pos[1] + 1.5 * obj.font_size)
    radius = int(obj.font_size * 2)
    obj = self._get_spotlight(x_cen, y_cen, radius)
    dict_objs[obj.name] = obj

    return dict_objs

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("You might have noticed that whenever you rescue people, the " +
            "\"Score\" (shown below) was increasing. Your goal is to receive " +
            "points as many as possible in " + MAX_STEP + " steps. Please " +
            "note that people rescued by your teammate (fire engine) " +
            "will also be counted as the same points and added to your score.")


class RescueV2TutorialPartialObs(RescueV2TutorialBase):

  def __init__(self) -> None:
    super().__init__(False, False, False)

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)
    init_states = ([1] * len(self._GAME_MAP["work_locations"]),
                   self._GAME_MAP["a1_init"], self._GAME_MAP["a2_init"],
                   self._GAME_MAP["a3_init"])
    agent2 = AIAgent_Rescue_PartialObs(init_states, self._AGENT2,
                                       self._TEAMMATE_POLICY_1)
    agent3 = AIAgent_Rescue_PartialObs(init_states, self._AGENT3,
                                       self._TEAMMATE_POLICY_2)
    game = user_game_data.get_game_ref()
    game.set_autonomous_agent(agent2=agent2, agent3=agent3)
    game.event_input(self._AGENT1, E_EventType.Set_Latent, None)

    user_game_data.data[Exp1UserData.PARTIAL_OBS] = True

  def _get_instruction(self, user_game_data: Exp1UserData):
    return (
        "During the TASK sessions, You cannot " +
        "observe the fire engine unless it is at each landmark or at the " +
        "same location as you (police car). Therefore, to do the task " +
        "efficiently you will need to guess where the fire engine will go. " +
        "Similarly, the fire engine CANNOT fully observe you.")


class RescueV2TutorialDestination(RescueV2TutorialBase):

  def __init__(self) -> None:
    super().__init__(False, False, False)

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    user_game_data.data[Exp1UserData.PARTIAL_OBS] = True

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_game_data)

    obj = dict_objs[self.TEXT_INSTRUCTION]  # type: co.TextObject
    x_cen = int(obj.pos[0] + 0.5 * obj.width)
    y_cen = int(self.GAME_HEIGHT / 5)
    radius = int(y_cen * 0.1)
    dict_objs[self.SPOTLIGHT] = self._get_spotlight(x_cen, y_cen, radius)

    objs = self._get_btn_actions(user_game_data.get_game_ref().get_env_info(),
                                 True, True, True)
    for obj in objs:
      dict_objs[obj.name] = obj

    return dict_objs

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("In the TASK sessions, you will have to select your next target" +
            " using the \"Select Destination\" button. Again, it is very " +
            "important to provide your destination instantly whenever you " +
            "change your target in your mind.")


class RescueV2TutorialLatent(RescueV2TutorialBase):

  def __init__(self) -> None:
    super().__init__(True, True, True)

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

    objs = self._get_btn_actions(user_game_data.get_game_ref().get_env_info(),
                                 True, True, True)
    for obj in objs:
      dict_objs[obj.name] = obj

    return dict_objs

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("In the TASK sessions, you will have to select your next target" +
            " using the \"Select Destination\" button. Again, it is very " +
            "important to provide your destination instantly whenever you " +
            "change your target in your mind.")

  def _get_button_commands(self, clicked_btn, user_data: Exp1UserData):
    if clicked_btn == co.BTN_SELECT:
      return {"delete": [self.SPOTLIGHT]}

    return None

  def button_clicked(self, user_game_data: Exp1UserData, clicked_btn: str):
    if self.is_sel_latent_btn(clicked_btn):
      user_game_data.go_to_next_page()

    return super().button_clicked(user_game_data, clicked_btn)


class RescueV2TutorialSelResult(RescueV2TutorialPlain):

  def init_user_data(self, user_game_data: Exp1UserData):
    # game no need to be initialized
    user_game_data.data[Exp1UserData.ACTION_COUNT] = 0
    user_game_data.data[Exp1UserData.SELECT] = False
    user_game_data.data[Exp1UserData.PARTIAL_OBS] = True

  def _get_instruction(self, user_game_data: Exp1UserData):
    return (
        "Well done! You can see your choice is marked with the flashing " +
        "red circle and you have selected your current destination. Also note" +
        " that we will prompt the destination selection automatically and " +
        "periodically to track the change in your mind as seamlessly" +
        " as possible.")

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_game_data)

    objs = self._get_btn_actions(user_game_data.get_game_ref().get_env_info(),
                                 True, True, True)
    for obj in objs:
      dict_objs[obj.name] = obj

    return dict_objs


class RescueV2TutorialMiniGame(RescueV2TutorialBase):

  def __init__(self) -> None:
    super().__init__(True, True, True)

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("Now, we are at the final step of the tutorial. Feel free to " +
            "interact with the interface and get familiar with the task. You" +
            " can also press the back button to revisit any of the previous " +
            "prompts.\n Once you are ready, please proceed to the TASK " +
            "sessions (using the button at the bottom of this page).")

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    TARGET = 0
    game = user_game_data.get_game_ref()
    agent1 = InteractiveAgent()
    init_states = ([1] * len(self._GAME_MAP["work_locations"]),
                   self._GAME_MAP["a1_init"], self._GAME_MAP["a2_init"],
                   self._GAME_MAP["a3_init"])
    agent2 = AIAgent_Rescue_PartialObs(init_states, self._AGENT2,
                                       self._TEAMMATE_POLICY_1)
    agent3 = AIAgent_Rescue_PartialObs(init_states, self._AGENT3,
                                       self._TEAMMATE_POLICY_2)
    game.set_autonomous_agent(agent1, agent2, agent3)
    game.event_input(self._AGENT1, E_EventType.Set_Latent, TARGET)

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
