from typing import Mapping, Sequence, Any
from web_experiment.exp_common.page_boxpushv2_base import BoxPushV2UserRandom
from web_experiment.exp_common.page_exp1_game_base import Exp1UserData
from web_experiment.exp_intervention.helper import task_intervention


class BoxPushV2Intervention(BoxPushV2UserRandom):
  def _on_action_taken(self, user_game_data: Exp1UserData,
                       dict_prev_game: Mapping[str, Any],
                       tuple_actions: Sequence[Any]):
    super()._on_action_taken(user_game_data, dict_prev_game, tuple_actions)

    game = user_game_data.get_game_ref()
    task_intervention(game.history, game, self._DOMAIN_TYPE)
