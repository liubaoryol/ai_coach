from ai_coach_domain.box_push_v2.simulator import BoxPushSimulatorV2
from ai_coach_domain.box_push_v2.mdp import (MDP_Movers_Task, MDP_Movers_Agent,
                                             MDP_Cleanup_Task,
                                             MDP_Cleanup_Agent)
from ai_coach_domain.box_push_v2.policy import Policy_Movers, Policy_Cleanup
from ai_coach_domain.box_push.agent import (BoxPushAIAgent_Team2,
                                            BoxPushAIAgent_Indv2,
                                            BoxPushInteractiveAgent)
from web_experiment.exp_common.page_exp1_base import Exp1UserData
from web_experiment.exp_common.page_exp1_game_base import Exp1PageGame


class BoxPushV2GamePage(Exp1PageGame):
  def __init__(self,
               is_movers,
               manual_latent_selection,
               game_map,
               auto_prompt: bool = True,
               prompt_on_change: bool = True,
               prompt_freq: int = 5) -> None:
    super().__init__(is_movers, manual_latent_selection, game_map, auto_prompt,
                     prompt_on_change, prompt_freq)

    TEMPERATURE = 0.3
    if self._IS_MOVERS:
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
  def __init__(self, is_movers, game_map, partial_obs) -> None:
    super().__init__(is_movers, True, game_map, True, True, 5)

    self._PARTIAL_OBS = partial_obs

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    agent1 = BoxPushInteractiveAgent()
    if self._IS_MOVERS:
      agent2 = BoxPushAIAgent_Team2(self._TEAMMATE_POLICY)
    else:
      agent2 = BoxPushAIAgent_Indv2(self._TEAMMATE_POLICY)

    game = user_game_data.get_game_ref()
    game.set_autonomous_agent(agent1, agent2)
    user_game_data.data[Exp1UserData.SELECT] = True
    user_game_data.data[Exp1UserData.SHOW_LATENT] = True

    user_game_data.data[Exp1UserData.PARTIAL_OBS] = self._PARTIAL_OBS
