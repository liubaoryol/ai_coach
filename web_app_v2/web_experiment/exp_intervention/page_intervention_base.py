import os
from typing import List, Mapping, Sequence, Any
import pickle
import numpy as np
from web_experiment.exp_common.canvas_objects import DrawingObject
from web_experiment.exp_common.page_exp1_game_base import Exp1UserData
from web_experiment.exp_intervention.helper import (task_intervention,
                                                    store_intervention_history)
import web_experiment.exp_common.canvas_objects as co
from web_experiment.define import EDomainType
import json
from flask_socketio import emit


def load_learned_models(domain_type):
  model_dir = os.path.join(os.path.dirname(__file__), "model_data/")
  if domain_type == EDomainType.Movers:
    v_value_file = "movers_160_0,30_150_merged_v_values_learned.pickle"
    policy1_file = "movers_btil_dec_policy_human_woTx_FTTT_160_0,30_a1.npy"
    policy2_file = "movers_btil_dec_policy_human_woTx_FTTT_160_0,30_a2.npy"
    tx1_file = "movers_btil_dec_tx_human_FTTT_160_0,30_a1.npy"
    tx2_file = "movers_btil_dec_tx_human_FTTT_160_0,30_a2.npy"
  elif domain_type == EDomainType.Rescue:
    v_value_file = "rescue_2_160_0,30_30_merged_v_values_learned.pickle"
    policy1_file = "rescue_2_btil_dec_policy_human_woTx_FTTT_160_0,30_a1.npy"
    policy2_file = "rescue_2_btil_dec_policy_human_woTx_FTTT_160_0,30_a2.npy"
    tx1_file = "rescue_2_btil_dec_tx_human_FTTT_160_0,30_a1.npy"
    tx2_file = "rescue_2_btil_dec_tx_human_FTTT_160_0,30_a2.npy"
  else:
    raise ValueError(f"Intervention page for {domain_type} is not implemented.")

  with open(model_dir + v_value_file, 'rb') as handle:
    v_values = pickle.load(handle)

  np_policy1 = np.load(model_dir + policy1_file)
  np_policy2 = np.load(model_dir + policy2_file)

  list_policies = [np_policy1, np_policy2]

  np_tx1 = np.load(model_dir + tx1_file)
  np_tx2 = np.load(model_dir + tx2_file)

  list_x_transitions = [np_tx1, np_tx2]

  return v_values, list_policies, list_x_transitions


class MixinInterventionBase:
  MOVERS_V_VALUES, MOVERS_LIST_PI, MOVERS_LIST_TX = load_learned_models(
      EDomainType.Movers)
  RESCUE_V_VALUES, RESCUE_LIST_PI, RESCUE_LIST_TX = load_learned_models(
      EDomainType.Rescue)

  CONFIRM_BUTTON = "Confirm Intervention"
  CONFIRM_MARK = "Confirm Mark"

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)
    user_game_data.data[Exp1UserData.INTERVENTION_HISTORY] = []

  def button_clicked(self, user_game_data: Exp1UserData, clicked_btn: str):

    if clicked_btn == MixinInterventionBase.CONFIRM_BUTTON:
      user_game_data.data[Exp1UserData.INTERVENTION] = None
      return

    return super().button_clicked(user_game_data, clicked_btn)

  def _game_overlay_names(self, game_env, user_data: Exp1UserData) -> List:
    overlay_names = super()._game_overlay_names(game_env, user_data)

    if user_data.data[Exp1UserData.INTERVENTION] is not None:
      overlay_names.append(co.SEL_LAYER)
      overlay_names.append(MixinInterventionBase.CONFIRM_MARK)
      overlay_names.append(MixinInterventionBase.CONFIRM_BUTTON)

    return overlay_names

  def _get_interv_confirm_objs(self, game_env, x_interv):

    pos, radius = self._get_latent_pos_size(game_env, x_interv)
    obj_mark = co.BlinkCircle(MixinInterventionBase.CONFIRM_MARK,
                              pos,
                              radius,
                              line_color="red",
                              fill=False,
                              border=True,
                              linewidth=3)

    x_ctrl_cen = int(self.GAME_RIGHT + (co.CANVAS_WIDTH - self.GAME_RIGHT) / 2)
    y_ctrl_cen = int(co.CANVAS_HEIGHT * 0.35)
    buttonsize = (int(self.GAME_WIDTH / 4), int(self.GAME_WIDTH / 15))
    font_size = 20

    obj_button = co.ButtonRect(MixinInterventionBase.CONFIRM_BUTTON,
                               (x_ctrl_cen, y_ctrl_cen),
                               buttonsize,
                               font_size,
                               "Confirm",
                               disable=False,
                               linewidth=3)

    return [obj_mark, obj_button]

  def _game_overlay(self, game_env,
                    user_data: Exp1UserData) -> List[DrawingObject]:
    overlay_obs = super()._game_overlay(game_env, user_data)

    x_intervention = user_data.data[Exp1UserData.INTERVENTION]
    if x_intervention is not None:
      obj = co.Rectangle(co.SEL_LAYER, (self.GAME_LEFT, self.GAME_TOP),
                         (self.GAME_WIDTH, self.GAME_HEIGHT),
                         fill_color="white",
                         alpha=0.8)
      overlay_obs.append(obj)

      objs = self._get_interv_confirm_objs(game_env, x_intervention)
      overlay_obs = overlay_obs + objs

    return overlay_obs

  def _conv_latent_to_advice(self, latent):
    return latent

  def _on_action_taken(self, user_game_data: Exp1UserData,
                       dict_prev_game: Mapping[str, Any],
                       tuple_actions: Sequence[Any]):
    super()._on_action_taken(user_game_data, dict_prev_game, tuple_actions)
    game = user_game_data.get_game_ref()
    if game.is_finished():
      return

    def policy_nxsa(nidx, xidx, sidx, tuple_aidx):
      return self._LIST_POLICIES[nidx][xidx, sidx, tuple_aidx[nidx]]

    def Tx_nxsasx(nidx, xidx, sidx, tuple_aidx, sidx_n, xidx_n):
      np_idx = tuple([xidx, *tuple_aidx, sidx_n])
      np_dist = self._LIST_TXS[nidx][np_idx]

      # for illegal states or states that haven't appeared during the training,
      # we assume mental model was maintained.
      if np.all(np_dist == np_dist[0]):
        np_dist = np.zeros_like(np_dist)
        np_dist[xidx] = 1

      return np_dist[xidx_n]

    prev_inference = user_game_data.data[Exp1UserData.PREV_INFERENCE]
    inf_res, intervention_latent, robot_latent = task_intervention(
        game.history[-1], game, self._TEAMMATE_POLICY, self._DOMAIN_TYPE,
        self.intervention_strategy, prev_inference, policy_nxsa, Tx_nxsasx)

    user_game_data.data[Exp1UserData.INTERVENTION] = intervention_latent
    user_game_data.data[Exp1UserData.PREV_INFERENCE] = inf_res

    objs = {}

    text_latent = self._conv_latent_to_advice(intervention_latent)
    if text_latent is None:
      txt_advice = "Beep- . Keep up the good work. "
    else:
      user_game_data.data[Exp1UserData.INTERVENTION_HISTORY].append(
          (game.current_step, intervention_latent, robot_latent))
      txt_advice = (
          "Beep beep -! A potential improvement in teamwork is identified: " +
          text_latent)

    objs["advice"] = txt_advice
    objs_json = json.dumps(objs)
    emit("intervention", objs_json)

  def _get_instruction(self, user_data: Exp1UserData):
    if user_data.data[Exp1UserData.INTERVENTION] is not None:
      return ("Tim suggested a better target for you " +
              "estimating the current situation. " +
              "The target is marked with a flashing red circle. " +
              "Please click the \"Confirm\" button once confirmed.")

    return ("Please choose your next action.")

  def _on_game_finished(self, user_game_data: Exp1UserData):
    super()._on_game_finished(user_game_data)

    user = user_game_data.data[Exp1UserData.USER]
    user_id = user.userid
    session_name = user_game_data.data[Exp1UserData.SESSION_NAME]

    user_path = user_game_data.data[Exp1UserData.USER_LABEL_PATH]
    interv_history = user_game_data.data[Exp1UserData.INTERVENTION_HISTORY]

    store_intervention_history(user_path, interv_history, user_id, session_name)
