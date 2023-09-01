import numpy as np
import os
import pickle
from aic_core.utils.exceptions import InvalidTransitionError
from tests.examples.environment import MMDPPolicy
from tests.examples.tooldelivery_v3_mdp import ToolDeliveryMDP_V3
import tests.examples.tooldelivery_v3_state_action as T3SA

RATIONALITY_SCALAR = 1


class ToolDeliveryPolicy_V3(MMDPPolicy):
  def __init__(self, mmdp):
    super().__init__()
    assert isinstance(mmdp, ToolDeliveryMDP_V3), "Wrong MMDP"
    self.mmdp = mmdp

    list_num_cn_state = [
        self.mmdp.dict_sTools_space[T3SA.ToolNames.SCALPEL_S].num_states,
        self.mmdp.dict_sTools_space[T3SA.ToolNames.SUTURE_S].num_states,
        self.mmdp.sCNPos_space.num_states
    ]
    self.num_cn_states = np.prod(list_num_cn_state)
    np_list_idx = np.arange(self.num_cn_states, dtype=np.int32)
    self.np_cn_state_to_idx = np_list_idx.reshape(list_num_cn_state)
    np_cn_idx_to_state = np.zeros((self.num_cn_states, len(list_num_cn_state)),
                                  dtype=np.int32)
    for state, idx in np.ndenumerate(self.np_cn_state_to_idx):
      np_cn_idx_to_state[idx] = state
    self.np_cn_idx_to_state = np_cn_idx_to_state

    self.value_table = {}
    cur_dir = os.path.dirname(__file__)
    pickle_name_scalpel_vtable = os.path.join(cur_dir,
                                              "scalpel_vtable_v3.pickle")
    pickle_name_suture_vtable = os.path.join(cur_dir,
                                             "forceps_vtable_v3.pickle")
    if os.path.exists(pickle_name_scalpel_vtable):
      with open(pickle_name_scalpel_vtable, 'rb') as handle:
        self.value_table[T3SA.LatentState.SCALPEL] = (pickle.load(handle))
      print("Value table for scalpel loaded from pickle")
    else:
      print("Generating value table for scalpel")
      vtable_scalpel = self.value_iteration(T3SA.LatentState.SCALPEL)
      self.value_table[T3SA.LatentState.SCALPEL] = vtable_scalpel
      with open(pickle_name_scalpel_vtable, 'wb') as handle:
        pickle.dump(vtable_scalpel, handle, protocol=pickle.HIGHEST_PROTOCOL)
      print("Done")

    if os.path.exists(pickle_name_suture_vtable):
      with open(pickle_name_suture_vtable, 'rb') as handle:
        self.value_table[T3SA.LatentState.SUTURE] = (pickle.load(handle))
      print("Value table for suture loaded from pickle")
    else:
      print("Generating value table for suture")
      vtable_suture = self.value_iteration(T3SA.LatentState.SUTURE)
      self.value_table[T3SA.LatentState.SUTURE] = vtable_suture
      with open(pickle_name_suture_vtable, 'wb') as handle:
        pickle.dump(vtable_suture, handle, protocol=pickle.HIGHEST_PROTOCOL)
      print("Done")

  def get_possible_latstate_indices(self):
    return tuple([i.value for i in T3SA.LatentState])

  # TODO: make this method generic
  def reward(self, cn_state_idx, cn_action_idx, latstate):
    sScal, sSut, sC = self.np_cn_idx_to_state[cn_state_idx]
    s_scalpel = self.mmdp.dict_sTools_space[
        T3SA.ToolNames.SCALPEL_S].idx_to_state[sScal]
    s_suture = self.mmdp.dict_sTools_space[
        T3SA.ToolNames.SUTURE_S].idx_to_state[sSut]
    act_cn = self.mmdp.aCN_space.idx_to_action[cn_action_idx]

    SCALEPL_REWARD = 10.
    FORCEPS_REWARD = 10.
    if (latstate == T3SA.LatentState.SCALPEL and s_scalpel == T3SA.ToolLoc.CN
        and act_cn == T3SA.ActionCN.HANDOVER):
      return SCALEPL_REWARD
    elif (latstate == T3SA.LatentState.SUTURE and s_suture == T3SA.ToolLoc.CN
          and act_cn == T3SA.ActionCN.HANDOVER):
      return FORCEPS_REWARD

    return 0.

  # TODO: make this method generic
  def q_values(self, value_table, cn_state_idx, latstate):
    GAMMA = 0.9
    q_table = np.zeros(self.mmdp.aCN_space.num_actions)
    for a_idx in range(self.mmdp.aCN_space.num_actions):
      sScal, sSut, sC = self.np_cn_idx_to_state[cn_state_idx]
      try:
        np_next_p_state_idx = self.mmdp.T_CN(sScal, sSut, sC, a_idx)
        val_sum = self.reward(cn_state_idx, a_idx, latstate)
        for p_sn, sn_tuple in np_next_p_state_idx:
          sn_idx = self.np_cn_state_to_idx[sn_tuple]
          val_sum += (p_sn * GAMMA * value_table[sn_idx])
      except InvalidTransitionError:
        val_sum = float("-inf")

      q_table[a_idx] = val_sum

    return q_table

  def value_iteration(self, latstate, eps=0.01):
    v_table = np.zeros(self.num_cn_states)
    while True:
      v_table_old = np.copy(v_table)
      delta = 0
      for s_idx in range(self.num_cn_states):
        q_table = self.q_values(v_table_old, s_idx, latstate)
        v_table[s_idx] = np.max(q_table)
        delta = max(delta, abs(v_table[s_idx] - v_table_old[s_idx]))
      if delta <= eps:
        return v_table

  def pi(self, obstate_idx, latstate_idx):
    state_vector = self.mmdp.np_idx_to_state[obstate_idx]
    dict_sTools = {}
    for i, tool in enumerate(T3SA.ToolNames):
      dict_sTools[tool] = state_vector[i]

    sPatient = state_vector[self.mmdp.num_tools]
    sCNPos = state_vector[self.mmdp.num_tools + 1]
    sAsked = state_vector[self.mmdp.num_tools + 2]

    latstate = None
    if latstate_idx is not None:
      latstate = T3SA.LatentState(latstate_idx)

    np_next_p_CN = self._pi_CN(dict_sTools=dict_sTools,
                               sCNPos=sCNPos,
                               sAsked=sAsked,
                               latstate=latstate)
    np_next_p_SN = self._pi_SN(dict_sTools=dict_sTools,
                               sPatient=sPatient,
                               sAsked=sAsked,
                               latstate=latstate)
    np_next_p_AS = self._pi_AS(dict_sTools=dict_sTools,
                               sPatient=sPatient,
                               sAsked=sAsked,
                               latstate=latstate)

    dict_next_p_action_idx = {}
    for p_aCN, aCN in np_next_p_CN:
      if p_aCN <= 0.0:
        continue
      for p_aSN, aSN in np_next_p_SN:
        if p_aSN <= 0.0:
          continue
        for p_aAS, aAS in np_next_p_AS:
          if p_aAS <= 0.0:
            continue
          p_action = p_aCN * p_aSN * p_aAS
          action_idx = self.mmdp.np_action_to_idx[aCN.astype(np.int32),
                                                  aSN.astype(np.int32),
                                                  aAS.astype(np.int32)]
          dict_next_p_action_idx[action_idx] = p_action

    np_next_p_action_idx = np.zeros((len(dict_next_p_action_idx), 2))
    iter_idx = 0
    for action_idx in dict_next_p_action_idx:
      np_next_p = dict_next_p_action_idx.get(action_idx)
      np_next_p_action_idx[iter_idx] = np_next_p, action_idx
      iter_idx += 1

    return np_next_p_action_idx

  def _pi_CN(self, dict_sTools, sCNPos, sAsked, latstate):
    action_to_idx = self.mmdp.aCN_space.action_to_idx
    s_asked = self.mmdp.sAsked_space.idx_to_state[sAsked]
    cn_state_idx = self.np_cn_state_to_idx[
        dict_sTools[T3SA.ToolNames.SCALPEL_S],
        dict_sTools[T3SA.ToolNames.SUTURE_S], sCNPos]

    list_next_p_aCN = []
    if s_asked == T3SA.StateAsked.NOT_ASKED:  # manual policy
      list_next_p_aCN.append([0.9, action_to_idx[T3SA.ActionCN.STAY]])
      list_next_p_aCN.append([0.025, action_to_idx[T3SA.ActionCN.MOVE_UP]])
      list_next_p_aCN.append([0.025, action_to_idx[T3SA.ActionCN.MOVE_DOWN]])
      list_next_p_aCN.append([0.025, action_to_idx[T3SA.ActionCN.MOVE_LEFT]])
      list_next_p_aCN.append([0.025, action_to_idx[T3SA.ActionCN.MOVE_RIGHT]])
    else:  # value based policy

      def get_probability_from_vtable(cn_state_idx_local, lat_state):
        q_table = self.q_values(self.value_table[lat_state], cn_state_idx_local,
                                lat_state)
        positive_indices = np.argwhere(q_table >= 0.)

        num_possible_actions = len(positive_indices)
        np_p = np.zeros(num_possible_actions)
        denom = 0.
        for idx, i_a in enumerate(positive_indices):
          numer = np.exp(RATIONALITY_SCALAR *
                         q_table[i_a])  # maxent distribution
          np_p[idx] = numer
          denom += numer
        np_p = np_p / denom

        dict_aCn_p = {}
        for idx in range(num_possible_actions):
          dict_aCn_p[positive_indices[idx][0]] = np_p[idx]

        return dict_aCn_p

      dict_aCN_p_from_vtable = get_probability_from_vtable(
          cn_state_idx, latstate)

      for aidx in dict_aCN_p_from_vtable:
        list_next_p_aCN.append([dict_aCN_p_from_vtable[aidx], aidx])

    return np.array(list_next_p_aCN)

  def _pi_SN(self, dict_sTools, sPatient, sAsked, latstate):
    action_to_idx = self.mmdp.aSN_space.action_to_idx
    s_asked = self.mmdp.sAsked_space.idx_to_state[sAsked]
    s_patient = self.mmdp.sPatient_space.idx_to_state[sPatient]

    sScalpel_p = dict_sTools[T3SA.ToolNames.SCALPEL_P]
    sSuture_p = dict_sTools[T3SA.ToolNames.SUTURE_P]
    s_scal_p = self.mmdp.dict_sTools_space[
        T3SA.ToolNames.SCALPEL_P].idx_to_state[sScalpel_p]
    s_sut_p = self.mmdp.dict_sTools_space[
        T3SA.ToolNames.SUTURE_P].idx_to_state[sSuture_p]

    if s_asked == T3SA.StateAsked.NOT_ASKED:
      if T3SA.ToolLoc.FLOOR in [s_scal_p, s_sut_p]:
        list_next_p_aSN = []
        # need to change the mental state here as well
        list_next_p_aSN.append([1.0, action_to_idx[T3SA.ActionSN.ASKTOOL]])
        # list_next_p_aSN.append(
        #     [0.01, action_to_idx[T3SA.ActionSN.STAY]])
        return np.array(list_next_p_aSN)
      elif (s_scal_p == T3SA.ToolLoc.SN
            and s_patient == T3SA.StatePatient.NO_INCISION):
        return np.array([[1., action_to_idx[T3SA.ActionSN.HO_SCALPEL]]])
      elif (s_sut_p == T3SA.ToolLoc.SN
            and s_patient == T3SA.StatePatient.INCISION):
        return np.array([[1., action_to_idx[T3SA.ActionSN.HO_SUTURE]]])
      elif T3SA.ToolLoc.AS in [s_scal_p, s_sut_p]:
        list_next_p_aSN = []
        p_sum = 0.
        if (s_patient == T3SA.StatePatient.NO_INCISION
            and s_scal_p == T3SA.ToolLoc.AS):
          p_tmp = 0.3  # 0.2 for scalpel, 0.1 for suture
          p_sum += p_tmp
          list_next_p_aSN.append([p_tmp, action_to_idx[T3SA.ActionSN.ASKTOOL]])
        elif (s_patient == T3SA.StatePatient.INCISION
              and s_sut_p == T3SA.ToolLoc.AS):
          p_tmp = 0.3  # 0.1 for scalpel, 0.2 for suture
          p_sum += p_tmp
          list_next_p_aSN.append([p_tmp, action_to_idx[T3SA.ActionSN.ASKTOOL]])

        list_next_p_aSN.append([1.0 - p_sum, action_to_idx[T3SA.ActionSN.STAY]])
        return np.array(list_next_p_aSN)
      assert False
    else:
      # need policies differentiated by latent states
      if latstate == T3SA.LatentState.SCALPEL:
        list_next_p_aSN = []
        list_next_p_aSN.append([0.2, action_to_idx[T3SA.ActionSN.STAY]])
        list_next_p_aSN.append(
            [0.6, action_to_idx[T3SA.ActionSN.SCALPEL_RELATED]])
        list_next_p_aSN.append(
            [0.2, action_to_idx[T3SA.ActionSN.SUTURE_RELATED]])
        return np.array(list_next_p_aSN)
      elif latstate == T3SA.LatentState.SUTURE:
        list_next_p_aSN = []
        list_next_p_aSN.append([0.2, action_to_idx[T3SA.ActionSN.STAY]])
        list_next_p_aSN.append(
            [0.2, action_to_idx[T3SA.ActionSN.SCALPEL_RELATED]])
        list_next_p_aSN.append(
            [0.6, action_to_idx[T3SA.ActionSN.SUTURE_RELATED]])
        return np.array(list_next_p_aSN)
      assert False

  def _pi_AS(self, dict_sTools, sPatient, sAsked, latstate):
    action_to_idx = self.mmdp.aAS_space.action_to_idx
    s_patient = self.mmdp.sPatient_space.idx_to_state[sPatient]
    s_asked = self.mmdp.sAsked_space.idx_to_state[sAsked]

    sScalpel_p = dict_sTools[T3SA.ToolNames.SCALPEL_P]
    sSuture_p = dict_sTools[T3SA.ToolNames.SUTURE_P]
    s_scal_p = self.mmdp.dict_sTools_space[
        T3SA.ToolNames.SCALPEL_P].idx_to_state[sScalpel_p]
    s_sut_p = self.mmdp.dict_sTools_space[
        T3SA.ToolNames.SUTURE_P].idx_to_state[sSuture_p]

    if s_asked == T3SA.StateAsked.NOT_ASKED:
      if (s_patient == T3SA.StatePatient.NO_INCISION
          and s_scal_p == T3SA.ToolLoc.AS):
        return np.array([[1., action_to_idx[T3SA.ActionAS.USE_SCALPEL]]])
      elif (s_patient == T3SA.StatePatient.INCISION
            and s_scal_p == T3SA.ToolLoc.AS):
        return np.array([[1., action_to_idx[T3SA.ActionAS.HO_SCALPEL]]])
      elif (s_patient == T3SA.StatePatient.INCISION
            and s_sut_p == T3SA.ToolLoc.AS):
        return np.array([[1., action_to_idx[T3SA.ActionAS.USE_SUTURE]]])
      else:
        return np.array([[1., action_to_idx[T3SA.ActionAS.STAY]]])
    else:
      return np.array([[1., action_to_idx[T3SA.ActionAS.STAY]]])


if __name__ == "__main__":
  task_mdp = ToolDeliveryMDP_V3()
  policy = ToolDeliveryPolicy_V3(task_mdp)
