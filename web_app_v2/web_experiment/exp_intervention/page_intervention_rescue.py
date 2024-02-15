from web_experiment.exp_common.page_rescue_game import RescueGamePage
from web_experiment.exp_common.page_exp1_game_base import Exp1UserData
from aic_core.intervention.feedback_strategy import (InterventionValueBased,
                                                     E_CertaintyHandling)
from web_experiment.exp_intervention.page_intervention_base import (
    MixinInterventionBase)
from aic_domain.rescue import PlaceName


class RescueV2InterventionPage(MixinInterventionBase, RescueGamePage):
  def __init__(self) -> None:
    super().__init__(True, latent_collection=False)
    self._V_VALUES = MixinInterventionBase.RESCUE_V_VALUES
    self._LIST_POLICIES = MixinInterventionBase.RESCUE_LIST_PI
    self._LIST_TXS = MixinInterventionBase.RESCUE_LIST_TX

    self.intervention_strategy = InterventionValueBased(
        self._V_VALUES,
        E_CertaintyHandling.Average,
        inference_threshold=0,
        intervention_threshold=0.1,
        intervention_cost=0)

  def _get_action_btn_disabled(self, user_data: Exp1UserData):

    if user_data.data[Exp1UserData.INTERVENTION] is not None:
      return True, True, True

    return super()._get_action_btn_disabled(user_data)

  def _conv_latent_to_advice(self, latent):
    if latent is not None:
      task_mdp = self._TEAMMATE_POLICY.mdp
      id = task_mdp.work_locations[latent].id
      place_name = task_mdp.places[id].name

      if place_name == PlaceName.City_hall:
        return "Please rescue the person at the CITY HALL."
      if place_name == PlaceName.Campsite:
        return "Please rescue people at the CAMPSITE."
      elif place_name == PlaceName.Bridge_1:
        return "Please repair the BOTTOM BRIDGE."
      elif place_name == PlaceName.Bridge_2:
        return "Please repair the UPPER BRIDGE."
      else:
        raise ValueError(f"Invalid latent: {latent}")
    else:
      return None
