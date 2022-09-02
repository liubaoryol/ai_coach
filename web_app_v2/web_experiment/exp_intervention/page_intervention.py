from typing import Mapping, Sequence, Any
from web_experiment.exp_common.page_boxpushv2_base import BoxPushV2UserRandom
from web_experiment.exp_common.page_exp1_game import Exp1UserData
from web_experiment.exp_intervention.helper import task_intervention
from web_experiment.define import EDomainType


class BoxPushV2Intervention(BoxPushV2UserRandom):
  def _on_action_taken(self, user_game_data: Exp1UserData,
                       dict_prev_game: Mapping[str, Any],
                       tuple_actions: Sequence[Any]):
    super()._on_action_taken(user_game_data, dict_prev_game, tuple_actions)

    game = user_game_data.get_game_ref()
    domain = EDomainType.Movers if self._IS_MOVERS else EDomainType.Cleanup
    task_intervention(game.history, game, domain)
