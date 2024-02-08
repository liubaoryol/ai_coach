from typing import Mapping, Any, Sequence
from web_experiment.exp_common.page_base import Exp1UserData
from web_experiment.exp_common.page_rescue_v2_game import RescueV2GamePage
import web_experiment.exp_common.canvas_objects as co
from web_experiment.define import ExpType
from web_experiment.models import ExpDataCollection, ExpIntervention, db
from aic_domain.agent import InteractiveAgent
from aic_domain.rescue_v2 import E_EventType
from aic_domain.rescue_v2.agent import AIAgent_Rescue_PartialObs

MAX_STEP = str(15)


class RescueV2TutorialBase(RescueV2GamePage):
  CLICKED_BTNS = "clicked_btn"

  def __init__(self, latent_collection: bool = True) -> None:
    super().__init__(latent_collection)

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
    user_game_data.data[Exp1UserData.PARTIAL_OBS] = False

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_game_data)

    btn_prev, btn_next = self._get_btn_prev_next(False, False)
    dict_objs[btn_prev.name] = btn_prev
    dict_objs[btn_next.name] = btn_next

    return dict_objs

  def _get_drawing_order(self, user_game_data: Exp1UserData):
    drawing_order = super()._get_drawing_order(user_game_data)

    drawing_order.append(self.SPOTLIGHT)
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


class RescueV2TutorialMiniGame(RescueV2TutorialBase):
  def __init__(self, latent_collection=True) -> None:
    super().__init__(latent_collection)

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
