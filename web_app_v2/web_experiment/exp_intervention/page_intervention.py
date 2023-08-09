from typing import Mapping, Sequence, Any
import pickle
import numpy as np
from web_experiment.exp_common.page_boxpushv2_base import BoxPushV2UserRandom
from web_experiment.exp_common.page_exp1_game_base import Exp1UserData
from web_experiment.exp_intervention.helper import task_intervention
from ai_coach_core.intervention.feedback_strategy import (
    InterventionValueBased, E_CertaintyHandling)
import web_experiment.exp_common.canvas_objects as co

# TODO: encapsulate these variables.
np_v_values_movers = None
list_np_policy = []
list_np_tx = []


class BoxPushV2Intervention(BoxPushV2UserRandom):

  def __init__(self, domain_type, partial_obs) -> None:
    super().__init__(domain_type, partial_obs, latent_collection=False)

    data_dir = "../misc/TIC_results/data/"
    model_dir = data_dir + "learned_models/"
    v_value_movers_file = "movers_500_0,30_500_merged_v_values_learned.pickle"

    tx1_file = "movers_btil2_tx_synth_FTTT_500_0,30_a1.npy"
    tx2_file = "movers_btil2_tx_synth_FTTT_500_0,30_a2.npy"

    policy1_file = "movers_btil2_policy_synth_woTx_FTTT_500_0,30_a1.npy"
    policy2_file = "movers_btil2_policy_synth_woTx_FTTT_500_0,30_a2.npy"

    with open(data_dir + v_value_movers_file, 'rb') as handle:
      np_v_values_movers = pickle.load(handle)

    np_policy1 = np.load(model_dir + policy1_file)
    np_tx1 = np.load(model_dir + tx1_file)
    np_policy2 = np.load(model_dir + policy2_file)
    np_tx2 = np.load(model_dir + tx2_file)
    list_np_policy.append(np_policy1)
    list_np_policy.append(np_policy2)
    list_np_tx.append(np_tx1)
    list_np_tx.append(np_tx2)

    self.intervention_strategy = InterventionValueBased(
        np_v_values_movers, E_CertaintyHandling.Average, 0, 15, 0)

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
    inf_res, cur_inference, require_intervention = task_intervention(
        game.history, game, self._DOMAIN_TYPE, self.intervention_strategy,
        prev_inference, policy_nxsa, Tx_nxsasx)
    if require_intervention:
      user_game_data.data[Exp1UserData.DURING_INTERVENTION] = True
      user_game_data.data[Exp1UserData.CUR_INFERENCE] = cur_inference
    else:
      user_game_data.data[Exp1UserData.DURING_INTERVENTION] = False
      user_game_data.data[Exp1UserData.CUR_INFERENCE] = None

    user_game_data.data[Exp1UserData.PREV_INFERENCE] = inf_res

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("Please choose your next action.")

  def _get_drawing_order(self, user_game_data: Exp1UserData):
    dict_game = user_game_data.get_game_ref().get_env_info()
    drawing_order = []
    drawing_order.append(self.GAME_BORDER)

    drawing_order = (drawing_order +
                     self._game_scene_names(dict_game, user_game_data))
    drawing_order = (drawing_order +
                     self._game_overlay_names(dict_game, user_game_data))
    drawing_order = drawing_order + co.ACTION_BUTTONS

    drawing_order.append(self.TEXT_SCORE)

    drawing_order.append(self.TEXT_INSTRUCTION)

    return drawing_order