from aic_domain.rescue_v2.simulator import RescueSimulatorV2
from aic_domain.agent import InteractiveAgent
from aic_domain.rescue_v2.agent import AIAgent_Rescue_PartialObs
from aic_domain.rescue_v2.mdp import MDP_Rescue_Task, MDP_Rescue_Agent
from aic_domain.rescue_v2.policy import Policy_Rescue
from aic_domain.rescue_v2.maps import MAP_RESCUE
from web_experiment.models import db, User
from web_experiment.exp_common.page_base import Exp1UserData
from web_experiment.exp_common.helper import get_file_name
from web_experiment.exp_common.page_rescue_v2_base import RescueV2GamePageBase

TEMPERATURE = 0.3
RESCUE_MAX_STEP = 15


class RescueV2GamePage(RescueV2GamePageBase):
  RESCUE_TEAMMATE1_POLICY = Policy_Rescue(MDP_Rescue_Task(**MAP_RESCUE),
                                          MDP_Rescue_Agent(**MAP_RESCUE),
                                          TEMPERATURE, RescueSimulatorV2.AGENT2)
  RESCUE_TEAMMATE2_POLICY = Policy_Rescue(MDP_Rescue_Task(**MAP_RESCUE),
                                          MDP_Rescue_Agent(**MAP_RESCUE),
                                          TEMPERATURE, RescueSimulatorV2.AGENT3)

  def __init__(self, latent_collection: bool = True) -> None:
    super().__init__(MAP_RESCUE, latent_collection)

    self._TEAMMATE_POLICY_1 = RescueV2GamePage.RESCUE_TEAMMATE1_POLICY
    self._TEAMMATE_POLICY_2 = RescueV2GamePage.RESCUE_TEAMMATE2_POLICY

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    game = user_game_data.get_game_ref()
    if game is None:
      game = RescueSimulatorV2()
      game.max_steps = RESCUE_MAX_STEP

      user_game_data.set_game(game)

    game.init_game(**self._GAME_MAP)
    game.set_autonomous_agent()

    user_game_data.data[Exp1UserData.ACTION_COUNT] = 0

  def _on_game_finished(self, user_game_data: Exp1UserData):
    '''
    user_game_data: NOTE - values will be updated
    '''
    super()._on_game_finished(user_game_data)

    game = user_game_data.get_game_ref()  # type: RescueSimulatorV2
    user = user_game_data.data[Exp1UserData.USER]
    user_id = user.userid

    # save trajectory
    save_path = user_game_data.data[Exp1UserData.SAVE_PATH]
    session_name = user_game_data.data[Exp1UserData.SESSION_NAME]
    file_name = get_file_name(save_path, user_id, session_name)
    header = game.__class__.__name__ + "-" + session_name + "\n"
    header += "User ID: %s\n" % (str(user_id), )
    header += str(self._GAME_MAP)
    game.save_history(file_name, header)

    # update score
    best_score = user.best_c

    if best_score < game.score:
      user = User.query.filter_by(userid=user_id).first()
      user.best_c = game.score
      db.session.commit()
      user_game_data.data[Exp1UserData.USER] = user

    # move to next page
    user_game_data.go_to_next_page()

  def _get_score_text(self, user_data):
    game = user_data.get_game_ref()
    if game is None:
      score = 0
      time_taken = 0
    else:
      score = user_data.get_game_ref().score
      time_taken = user_data.get_game_ref().current_step

    best_score = user_data.data[Exp1UserData.USER].best_c

    text_score = "Time Taken: " + str(time_taken) + "\n"
    text_score += "Score: " + str(score) + "\n"
    text_score += "(Your Best: " + str(best_score) + ")"

    return text_score


class RescueV2GameUserRandom(RescueV2GamePage):
  def __init__(self, partial_obs, latent_collection=True) -> None:
    super().__init__(latent_collection)
    self._PARTIAL_OBS = partial_obs

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    agent1 = InteractiveAgent()
    init_states = ([1] * len(self._GAME_MAP["work_locations"]),
                   self._GAME_MAP["a1_init"], self._GAME_MAP["a2_init"],
                   self._GAME_MAP["a3_init"])
    agent2 = AIAgent_Rescue_PartialObs(init_states, self._AGENT2,
                                       self._TEAMMATE_POLICY_1)
    agent3 = AIAgent_Rescue_PartialObs(init_states, self._AGENT3,
                                       self._TEAMMATE_POLICY_2)

    game = user_game_data.get_game_ref()
    game.set_autonomous_agent(agent1, agent2, agent3)
    user_game_data.data[Exp1UserData.SELECT] = self._LATENT_COLLECTION

    user_game_data.data[Exp1UserData.PARTIAL_OBS] = self._PARTIAL_OBS
