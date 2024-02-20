from typing import Mapping, Any, Sequence
from web_experiment.exp_common.canvas_objects import DrawingObject
from web_experiment.exp_common.page_base import Exp1UserData
import web_experiment.exp_common.canvas_objects as co
from web_experiment.exp_common.page_tutorial_base import MixinTutorialBase
from web_experiment.define import ExpType
from aic_domain.agent import InteractiveAgent
from aic_domain.rescue import E_EventType
from aic_domain.box_push_v2 import EventType
from web_experiment.exp_intervention.page_intervention_movers import (
    BoxPushV2InterventionPage)
from web_experiment.exp_intervention.page_intervention_rescue import (
    RescueV2InterventionPage)
from web_experiment.exp_common.page_tutorial import CanvasPageTutorialPlain
from web_experiment.exp_common.page_tutorial_rescue import RescueTutorialPlain
import json
from flask_socketio import emit


class BoxPushTutorialInterventionIntro(CanvasPageTutorialPlain):
  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("Above this game screen is Tim, your AI team coach. " +
            "Tim will monitor your and your teammate's actions and " +
            "give you suggestions if he estimates they could be beneficial " +
            "for the team.")


class BoxPushTutorialInterventionUI(MixinTutorialBase,
                                    BoxPushV2InterventionPage):
  def __init__(self, domain_type) -> None:
    super().__init__(domain_type)
    self._MANUAL_SELECTION = self._AUTO_PROMPT = self._PROMPT_ON_CHANGE = False

  def init_user_data(self, user_game_data: Exp1UserData):
    self._base_init_user_data(user_game_data)

    intervention_latent = ("pickup", 0)
    user_game_data.data[Exp1UserData.INTERVENTION] = intervention_latent

    text_latent = self._conv_latent_to_advice(intervention_latent)
    txt_advice = (
        "Beep beep -! I\'ve spotted a potential opportunity to enhance our teamwork: "
        + text_latent)
    objs = {}
    objs["advice"] = txt_advice
    objs_json = json.dumps(objs)
    emit("intervention", objs_json)

  def _get_instruction(self, user_game_data: Exp1UserData):
    return (
        "This is how the UI looks like when Tim tells you advice. " +
        "You will see Tim's suggestion on the screen as a red flashing circle, "
        + "and a \"Confirm\" button will appear below this panel. " +
        "You can resume the task by clicking the \"Confirm\" button.")

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, DrawingObject]:
    dict_objs = self._base_get_init_drawing_objects(user_game_data)
    dict_objs[co.BTN_NEXT].disable = True
    return dict_objs

  def button_clicked(self, user_game_data: Exp1UserData, clicked_btn: str):
    basic_text = ("Beep- beep. I'm Tim, your AI team coach. " +
                  "I'm here to help you achieve better task results.")
    if clicked_btn == self.CONFIRM_BUTTON:
      objs = {}
      objs["advice"] = basic_text
      objs_json = json.dumps(objs)
      emit("intervention", objs_json)
      user_game_data.go_to_next_page()
      return
    elif clicked_btn == co.BTN_PREV:
      objs = {}
      objs["advice"] = basic_text
      objs_json = json.dumps(objs)
      emit("intervention", objs_json)
      user_game_data.go_to_prev_page()
      return

    return super().button_clicked(user_game_data, clicked_btn)


class RescueTutorialInterventionIntro(RescueTutorialPlain):
  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)
    user_game_data.data[Exp1UserData.PARTIAL_OBS] = True

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("Again, for some sessions, your AI team coach, Tim, " +
            "will help you achieve better task results. " +
            "Please confirm Tim's suggestions and follow them once received.")
