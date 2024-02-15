from aic_domain.box_push_v2.simulator import BoxPushSimulatorV2
from aic_domain.box_push_v2.mdp import (MDP_Movers_Task, MDP_Movers_Agent,
                                        MDP_Cleanup_Task, MDP_Cleanup_Agent)
from aic_domain.box_push_v2.policy import Policy_Movers, Policy_Cleanup
from aic_domain.box_push_v2.maps import MAP_MOVERS
from aic_domain.box_push_v2.maps import MAP_CLEANUP_V3 as MAP_CLEANUP
from aic_domain.agent import InteractiveAgent
from aic_domain.box_push_v2.agent import (BoxPushAIAgent_PO_Indv,
                                          BoxPushAIAgent_PO_Team)
from web_experiment.models import db, User
from web_experiment.define import EDomainType
from web_experiment.exp_common.page_base import Exp1UserData
from web_experiment.exp_common.page_exp1_game_base import BoxPushGamePageBase
from web_experiment.exp_common.helper import (get_file_name,
                                              store_user_label_locally)

TEMPERATURE = 0.3


class BoxPushV2GamePage(BoxPushGamePageBase):
  MOVERS_TEAMMATE_POLICY = Policy_Movers(MDP_Movers_Task(**MAP_MOVERS),
                                         MDP_Movers_Agent(**MAP_MOVERS),
                                         TEMPERATURE, BoxPushSimulatorV2.AGENT2)
  CLEANUP_TEAMMATE_POLICY = Policy_Cleanup(MDP_Cleanup_Task(**MAP_CLEANUP),
                                           MDP_Cleanup_Agent(**MAP_CLEANUP),
                                           TEMPERATURE,
                                           BoxPushSimulatorV2.AGENT2)

  def __init__(self,
               domain_type,
               partial_obs,
               latent_collection: bool = True) -> None:
    game_map = MAP_MOVERS if domain_type == EDomainType.Movers else MAP_CLEANUP
    super().__init__(domain_type, game_map, latent_collection)

    if self._DOMAIN_TYPE == EDomainType.Movers:
      self._TEAMMATE_POLICY = BoxPushV2GamePage.MOVERS_TEAMMATE_POLICY
    else:
      self._TEAMMATE_POLICY = BoxPushV2GamePage.CLEANUP_TEAMMATE_POLICY

    self._PARTIAL_OBS = partial_obs

  def init_user_data(self, user_game_data: Exp1UserData):
    user_game_data.data[Exp1UserData.PAGE_DONE] = False
    user_game_data.data[Exp1UserData.GAME_DONE] = False

    game = user_game_data.get_game_ref()
    if game is None:
      game = BoxPushSimulatorV2(None)

      user_game_data.set_game(game)

    game.init_game(**self._GAME_MAP)

    user_game_data.data[Exp1UserData.ACTION_COUNT] = 0
    user_game_data.data[Exp1UserData.USER_LABELS] = []

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

    game = user_game_data.get_game_ref()
    game.set_autonomous_agent(agent1, agent2)

    user_game_data.data[Exp1UserData.SELECT] = self._LATENT_COLLECTION
    user_game_data.data[Exp1UserData.PARTIAL_OBS] = self._PARTIAL_OBS

  def _on_game_finished(self, user_game_data: Exp1UserData):
    '''
    user_game_data: NOTE - values will be updated
    '''
    super()._on_game_finished(user_game_data)

    game = user_game_data.get_game_ref()
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

    if self._LATENT_COLLECTION:
      user_label_path = user_game_data.data[Exp1UserData.USER_LABEL_PATH]
      user_labels = user_game_data.data[Exp1UserData.USER_LABELS]
      store_user_label_locally(user_label_path, user_id, session_name,
                               user_labels)

    # update score
    if self._DOMAIN_TYPE == EDomainType.Movers:
      best_score = user.best_a
    elif self._DOMAIN_TYPE == EDomainType.Cleanup:
      best_score = user.best_b
    else:
      raise ValueError("Domain should be either Movers or Cleanup." +
                       f"{self._DOMAIN_TYPE} cannot not be handled")

    if best_score > game.current_step:
      user = User.query.filter_by(userid=user_id).first()
      if self._DOMAIN_TYPE == EDomainType.Movers:
        user.best_a = game.current_step
      elif self._DOMAIN_TYPE == EDomainType.Cleanup:
        user.best_b = game.current_step
      else:
        raise ValueError("Domain should be either Movers or Cleanup." +
                         f"{self._DOMAIN_TYPE} cannot not be handled")

      db.session.commit()
      user_game_data.data[Exp1UserData.USER] = user

    # move to next page
    user_game_data.go_to_next_page()

  def _get_score_text(self, user_data: Exp1UserData):
    game = user_data.get_game_ref()
    if game is None:
      score = 0
    else:
      score = user_data.get_game_ref().current_step

    if self._DOMAIN_TYPE == EDomainType.Movers:
      best_score = user_data.data[Exp1UserData.USER].best_a
    else:
      best_score = user_data.data[Exp1UserData.USER].best_b

    text_score = "Time Taken: " + str(score) + "\n"
    if best_score == 999:
      text_score += "(Your Best: - )"
    else:
      text_score += "(Your Best: " + str(best_score) + ")"

    return text_score
