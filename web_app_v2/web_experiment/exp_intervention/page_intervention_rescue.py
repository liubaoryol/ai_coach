import os
from typing import Mapping, Sequence, Any
from web_experiment.define import EDomainType
from web_experiment.exp_common.page_rescue_game import RescueGameUserRandom
from web_experiment.exp_common.page_exp1_game_base import Exp1UserData
from web_experiment.exp_intervention.helper import task_intervention
from aic_core.intervention.feedback_strategy import (InterventionValueBased,
                                                     E_CertaintyHandling)
import pickle
import numpy as np

# TODO: encapsulate these variables.
np_v_values_rescue = None
list_np_policy = []
list_np_tx = []


class RescueV2Intervention(RescueGameUserRandom):
  def __init__(self, partial_obs) -> None:
    super().__init__(partial_obs, latent_collection=False)

    model_dir = os.path.join(os.path.dirname(__file__), "model_data/")
    v_value_file = "rescue_2_160_0,30_30_merged_v_values_learned.pickle"

    tx1_file = "rescue_2_btil_dec_tx_human_FTTT_160_0,30_a1.npy"
    tx2_file = "rescue_2_btil_dec_tx_human_FTTT_160_0,30_a2.npy"

    policy1_file = "rescue_2_btil_dec_policy_human_woTx_FTTT_160_0,30_a1.npy"
    policy2_file = "rescue_2_btil_dec_policy_human_woTx_FTTT_160_0,30_a2.npy"

    with open(model_dir + v_value_file, 'rb') as handle:
      np_v_values_rescue = pickle.load(handle)

    np_policy1 = np.load(model_dir + policy1_file)
    np_tx1 = np.load(model_dir + tx1_file)
    np_policy2 = np.load(model_dir + policy2_file)
    np_tx2 = np.load(model_dir + tx2_file)
    list_np_policy.append(np_policy1)
    list_np_policy.append(np_policy2)
    list_np_tx.append(np_tx1)
    list_np_tx.append(np_tx2)

    self.intervention_strategy = InterventionValueBased(
        np_v_values_rescue, E_CertaintyHandling.Average, 0, 0.1, 0)

  def _on_action_taken(self, user_game_data: Exp1UserData,
                       dict_prev_game: Mapping[str, Any],
                       tuple_actions: Sequence[Any]):
    super()._on_action_taken(user_game_data, dict_prev_game, tuple_actions)

    def policy_nxsa(nidx, xidx, sidx, tuple_aidx):
      return list_np_policy[nidx][xidx, sidx, tuple_aidx[nidx]]

    def Tx_nxsasx(nidx, xidx, sidx, tuple_aidx, sidx_n, xidx_n):
      np_idx = tuple([xidx, *tuple_aidx, sidx_n])
      np_dist = list_np_tx[nidx][np_idx]

      # for illegal states or states that haven't appeared during the training,
      # we assume mental model was maintained.
      if np.all(np_dist == np_dist[0]):
        np_dist = np.zeros_like(np_dist)
        np_dist[xidx] = 1

      return np_dist[xidx_n]

    game = user_game_data.get_game_ref()
    prev_inference = user_game_data.data[Exp1UserData.PREV_INFERENCE]
    inf_res = task_intervention(game.history, game, self._DOMAIN_TYPE,
                                self.intervention_strategy, prev_inference,
                                policy_nxsa, Tx_nxsasx)
    user_game_data.data[Exp1UserData.PREV_INFERENCE] = inf_res

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("Please choose your next action.")
