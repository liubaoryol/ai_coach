from web_experiment.exp_common.page_boxpushv2_base import BoxPushV2UserRandom
from web_experiment.exp_common.page_exp1_game_base import Exp1UserData
from aic_core.intervention.feedback_strategy import (InterventionValueBased,
                                                     E_CertaintyHandling)
from web_experiment.exp_intervention.page_intervention_base import (
    MixinInterventionBase)


class BoxPushV2Intervention(MixinInterventionBase, BoxPushV2UserRandom):
  def __init__(self, domain_type, partial_obs) -> None:
    super().__init__(domain_type, partial_obs, latent_collection=False)
    self._V_VALUES = MixinInterventionBase.MOVERS_V_VALUES
    self._LIST_POLICIES = MixinInterventionBase.MOVERS_LIST_PI
    self._LIST_TXS = MixinInterventionBase.MOVERS_LIST_TX

    self.intervention_strategy = InterventionValueBased(
        self._V_VALUES,
        E_CertaintyHandling.Average,
        inference_threshold=0,
        intervention_threshold=3,
        intervention_cost=1)

  def _get_action_btn_disabled(self, user_data: Exp1UserData):

    if user_data.data[Exp1UserData.INTERVENTION] is not None:
      return True, True, True, True, True, True, True

    return super()._get_action_btn_disabled(user_data)

  def _conv_latent_to_advice(self, latent):
    if latent is not None:
      if latent[0] == "pickup":
        if latent[1] == 0:
          return "Please pick up the TOP LEFT box."
        elif latent[1] == 1:
          return "Please pick up the BOTTOM CENTER box."
        elif latent[1] == 2:
          return "Please pick up the BOTTOM RIGHT box."
        else:
          raise ValueError(f"Invalid latent: {latent}")
      elif latent[0] == "goal":
        return "Please drop the box at the TRUCK."
      else:
        raise ValueError(f"Invalid latent: {latent}")
    else:
      return None
