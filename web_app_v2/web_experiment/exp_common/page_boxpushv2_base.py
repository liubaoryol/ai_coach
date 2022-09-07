from ai_coach_domain.box_push_v2.simulator import BoxPushSimulatorV2
from ai_coach_domain.box_push_v2.mdp import (MDP_Movers_Task, MDP_Movers_Agent,
                                             MDP_Cleanup_Task,
                                             MDP_Cleanup_Agent)
from ai_coach_domain.box_push_v2.policy import Policy_Movers, Policy_Cleanup
from ai_coach_domain.box_push.agent import (BoxPushAIAgent_Team2,
                                            BoxPushAIAgent_Indv2,
                                            InteractiveAgent)
from web_experiment.exp_common.page_base import Exp1UserData
from web_experiment.exp_common.page_exp1_game_base import BoxPushGamePageBase
from web_experiment.define import EDomainType


class BoxPushV2GamePage(BoxPushGamePageBase):
  def __init__(self,
               domain_type,
               manual_latent_selection,
               game_map,
               auto_prompt: bool = True,
               prompt_on_change: bool = True,
               prompt_freq: int = 5) -> None:
    super().__init__(domain_type, manual_latent_selection, game_map,
                     auto_prompt, prompt_on_change, prompt_freq)

    TEMPERATURE = 0.3
    if self._DOMAIN_TYPE == EDomainType.Movers:
      task_mdp = MDP_Movers_Task(**self._GAME_MAP)
      agent_mdp = MDP_Movers_Agent(**self._GAME_MAP)
      self._TEAMMATE_POLICY = Policy_Movers(task_mdp, agent_mdp, TEMPERATURE,
                                            self._AGENT2)
    else:
      task_mdp = MDP_Cleanup_Task(**self._GAME_MAP)
      agent_mdp = MDP_Cleanup_Agent(**self._GAME_MAP)
      self._TEAMMATE_POLICY = Policy_Cleanup(task_mdp, agent_mdp, TEMPERATURE,
                                             self._AGENT2)

  def init_user_data(self, user_game_data: Exp1UserData):
    user_game_data.data[Exp1UserData.GAME_DONE] = False
    user_game_data.data[Exp1UserData.SELECT] = False

    game = user_game_data.get_game_ref()
    if game is None:
      game = BoxPushSimulatorV2(None)

      user_game_data.set_game(game)

    game.init_game(**self._GAME_MAP)

    user_game_data.data[Exp1UserData.ACTION_COUNT] = 0
    user_game_data.data[Exp1UserData.SELECT] = False


class BoxPushV2UserRandom(BoxPushV2GamePage):
  def __init__(self, domain_type, game_map, partial_obs) -> None:
    super().__init__(domain_type, True, game_map, True, True, 5)

    self._PARTIAL_OBS = partial_obs

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    agent1 = InteractiveAgent()
    if self._DOMAIN_TYPE == EDomainType.Movers:
      agent2 = BoxPushAIAgent_Team2(self._TEAMMATE_POLICY)
    else:
      agent2 = BoxPushAIAgent_Indv2(self._TEAMMATE_POLICY)

    game = user_game_data.get_game_ref()
    game.set_autonomous_agent(agent1, agent2)
    user_game_data.data[Exp1UserData.SELECT] = True
    user_game_data.data[Exp1UserData.SHOW_LATENT] = True

    user_game_data.data[Exp1UserData.PARTIAL_OBS] = self._PARTIAL_OBS
