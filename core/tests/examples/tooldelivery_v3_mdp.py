import numpy as np
from ai_coach_core.models.mdp import MDP
from ai_coach_core.utils.mdp_utils import StateSpace, ActionSpace
from ai_coach_core.utils.exceptions import InvalidTransitionError
import tests.examples.tooldelivery_v3_state_action as T3SA


class ToolDeliveryMDP_V3(MDP):
  def __init__(self):
    self.num_tools = len(T3SA.ToolNames)
    self.num_x_grid = 5
    self.num_y_grid = 5
    self.cabinet_loc = (2, 4)
    self.storage_loc = (0, 4)
    self.handover_loc = (4, 1)
    self.walls = [(1.5, i) for i in range(self.num_y_grid)]
    self.doors = [(1.5, 1)]
    super().__init__()
    self.num_agents = 3

  def init_statespace(self):
    '''
        We use a factored structure for the state defined as follows:
        sTools[tool_name] : state vars associated with location of tool_name
        sPatient : state vars associated with patient status
        sCNPos : state vars associated with CN positions
        sAsked : state vars associated with whether SN asked a tool
        s = [sTools[too1], sTools[tool2], ..., sPatient, sCNPos, sAsked]
            these factored statespaces are stored in a dictionary
        '''

    self.dict_factored_statespace = {}

    self.dict_sTools_space = {}
    for tool in T3SA.ToolNames:
      self.dict_sTools_space[tool] = StateSpace(statespace=T3SA.ToolLoc)

    self.sPatient_space = StateSpace(statespace=T3SA.StatePatient)
    self.sCNPos_space = StateSpace(
        statespace=T3SA.generate_CNPos_states(self.num_x_grid, self.num_y_grid))
    self.sAsked_space = StateSpace(statespace=T3SA.StateAsked)

    for i, tool in enumerate(T3SA.ToolNames):
      self.dict_factored_statespace[i] = self.dict_sTools_space[tool]

    idx_patient = self.num_tools
    idx_cnpos = self.num_tools + 1
    idx_asked = self.num_tools + 2
    self.dict_factored_statespace[idx_patient] = self.sPatient_space
    self.dict_factored_statespace[idx_cnpos] = self.sCNPos_space
    self.dict_factored_statespace[idx_asked] = self.sAsked_space

    self.dummy_states = None

  def init_actionspace(self):
    '''
        We use a factored representation for the joint action:
        aCN : circulating nurse action
        aSN : scrub nurse action
        aAS : surgeon action
        '''

    self.dict_factored_actionspace = {}

    self.aCN_space = ActionSpace(actionspace=T3SA.ActionCN)
    self.aSN_space = ActionSpace(actionspace=T3SA.ActionSN)
    self.aAS_space = ActionSpace(actionspace=T3SA.ActionAS)

    self.dict_factored_actionspace = {
        0: self.aCN_space,
        1: self.aSN_space,
        2: self.aAS_space
    }

  def transition_model(self, state_idx, action_idx):
    '''
        Returns a np array with two columns and at least one row.
        The first column corresponds to the probability for the next state.
        The second column corresponds to the index of the next state.
        '''

    # unpack the input
    state_vector = self.np_idx_to_state[state_idx]
    sScal_p, sSut_p, sScal_s, sSut_s, sPat, sCNPos, sAsk = state_vector

    action_vector = self.np_idx_to_action[action_idx]
    aCN, aSN, aAS = action_vector

    self._raise_exception_if_illegal_T(sScal_p, sSut_p, sScal_s, sSut_s, sPat,
                                       sCNPos, sAsk, aCN, aSN, aAS)

    np_next_p_sScal_p = self._T_sTool(tool_nm=T3SA.ToolNames.SCALPEL_P,
                                      sTool=sScal_p,
                                      sCNPos=sCNPos,
                                      aCN=aCN,
                                      aSN=aSN,
                                      aAS=aAS)
    np_next_p_sSut_p = self._T_sTool(tool_nm=T3SA.ToolNames.SUTURE_P,
                                     sTool=sSut_p,
                                     sCNPos=sCNPos,
                                     aCN=aCN,
                                     aSN=aSN,
                                     aAS=aAS)
    np_next_p_sScal_s = self._T_sTool(tool_nm=T3SA.ToolNames.SCALPEL_S,
                                      sTool=sScal_s,
                                      sCNPos=sCNPos,
                                      aCN=aCN,
                                      aSN=aSN,
                                      aAS=aAS)
    np_next_p_sSut_s = self._T_sTool(tool_nm=T3SA.ToolNames.SUTURE_S,
                                     sTool=sSut_s,
                                     sCNPos=sCNPos,
                                     aCN=aCN,
                                     aSN=aSN,
                                     aAS=aAS)

    np_next_p_sPatinet = self._T_sPatient(sPat=sPat, aAS=aAS)

    # deterministic
    np_next_p_sCNPos = self._T_sCNPos(sCNPos=sCNPos, aCN=aCN)
    np_next_p_sAsked = self._T_sAsked(sAsked=sAsk, aSN=aSN)
    sPos_prime = np_next_p_sCNPos[0][1].astype(np.int32)
    sAsk_prime = np_next_p_sAsked[0][1].astype(np.int32)

    # Merge np_next_p_s* to get np_next_p_state_idx
    dict_next_p_state_idx = {}
    for p_ScP, sScP in np_next_p_sScal_p:
      for p_SuP, sSuP in np_next_p_sSut_p:
        for p_ScS, sScS in np_next_p_sScal_s:
          for p_SuS, sSuS in np_next_p_sSut_s:
            for p_pat, sPat in np_next_p_sPatinet:
              next_state_idx = self.np_state_to_idx[sScP.astype(np.int32),
                                                    sSuP.astype(np.int32),
                                                    sScS.astype(np.int32),
                                                    sSuS.astype(np.int32),
                                                    sPat.astype(np.int32),
                                                    sPos_prime, sAsk_prime]
              next_p = p_ScP * p_SuP * p_ScS * p_SuS * p_pat
              dict_next_p_state_idx[next_state_idx] = next_p

    num_next_states = len(dict_next_p_state_idx)
    np_next_p_state_idx = np.zeros((num_next_states, 2))
    iter_idx = 0
    for next_state_idx in dict_next_p_state_idx:
      next_p = dict_next_p_state_idx.get(next_state_idx)
      np_next_p_state_idx[iter_idx] = next_p, next_state_idx
      iter_idx += 1

    # assert (np.sum(np_next_p_state_idx[:, 0]) == 1.)

    return np_next_p_state_idx

  def _T_sPatient(self, sPat, aAS):
    state = self.sPatient_space.idx_to_state[sPat]
    action = self.aAS_space.idx_to_action[aAS]

    list_next_p_sPatient = []
    if (state == T3SA.StatePatient.NO_INCISION
        and action == T3SA.ActionAS.USE_SCALPEL):
      sIncision = self.sPatient_space.state_to_idx[T3SA.StatePatient.INCISION]
      list_next_p_sPatient.append([0.2, sIncision])
      list_next_p_sPatient.append([0.8, sPat])

      return np.array(list_next_p_sPatient)
    elif (state == T3SA.StatePatient.INCISION
          and action == T3SA.ActionAS.USE_SUTURE):
      sNoIncision = self.sPatient_space.state_to_idx[
          T3SA.StatePatient.NO_INCISION]
      list_next_p_sPatient.append([0.2, sNoIncision])
      list_next_p_sPatient.append([0.8, sPat])

      return np.array(list_next_p_sPatient)

    # otherwise, patient status doesn't change
    return np.array([[1., sPat]])

  def _raise_exception_if_illegal_T(self, sScal_p, sSut_p, sScal_s, sSut_s,
                                    sPatient, sCNPos, sAsked, aCN, aSN, aAS):
    coord_cn = self.sCNPos_space.idx_to_state[sCNPos]
    state_asked = self.sAsked_space.idx_to_state[sAsked]
    act_cn = self.aCN_space.idx_to_action[aCN]
    act_sn = self.aSN_space.idx_to_action[aSN]
    act_as = self.aAS_space.idx_to_action[aAS]

    scalpel_p_loc = self.dict_sTools_space[
        T3SA.ToolNames.SCALPEL_P].idx_to_state[sScal_p]
    suture_p_loc = self.dict_sTools_space[
        T3SA.ToolNames.SUTURE_P].idx_to_state[sSut_p]
    scalpel_s_loc = self.dict_sTools_space[
        T3SA.ToolNames.SCALPEL_S].idx_to_state[sScal_s]
    suture_s_loc = self.dict_sTools_space[
        T3SA.ToolNames.SUTURE_S].idx_to_state[sSut_s]
    list_spare_tool_loc = [scalpel_s_loc, suture_s_loc]

    if ((coord_cn not in [self.cabinet_loc, self.storage_loc])
        and act_cn == T3SA.ActionCN.PICKUP):
      raise InvalidTransitionError

    if coord_cn != self.handover_loc and act_cn == T3SA.ActionCN.HANDOVER:
      raise InvalidTransitionError

    if (coord_cn == self.cabinet_loc and act_cn == T3SA.ActionCN.PICKUP
        and (T3SA.ToolLoc.CABINET not in list_spare_tool_loc)):
      raise InvalidTransitionError

    if (coord_cn == self.storage_loc and act_cn == T3SA.ActionCN.PICKUP
        and (T3SA.ToolLoc.STORAGE not in list_spare_tool_loc)):
      raise InvalidTransitionError

    if (act_cn == T3SA.ActionCN.HANDOVER
        and (T3SA.ToolLoc.CN not in list_spare_tool_loc)):
      raise InvalidTransitionError

    if (act_sn == T3SA.ActionSN.HO_SCALPEL
        and scalpel_p_loc != T3SA.ToolLoc.SN):
      raise InvalidTransitionError

    if (act_sn == T3SA.ActionSN.HO_SUTURE and suture_p_loc != T3SA.ToolLoc.SN):
      raise InvalidTransitionError

    if (state_asked == T3SA.StateAsked.ASKED
        and act_sn == T3SA.ActionSN.ASKTOOL):
      raise InvalidTransitionError

    if (act_as in [T3SA.ActionAS.HO_SCALPEL, T3SA.ActionAS.USE_SCALPEL]
        and scalpel_p_loc != T3SA.ToolLoc.AS):
      raise InvalidTransitionError

    if (act_as == T3SA.ActionAS.USE_SUTURE and suture_p_loc != T3SA.ToolLoc.AS):
      raise InvalidTransitionError

  def _T_sTool(self, tool_nm, sTool, sCNPos, aCN, aSN, aAS):
    tool_loc = self.dict_sTools_space[tool_nm].idx_to_state[sTool]
    coord_cn = self.sCNPos_space.idx_to_state[sCNPos]
    act_cn = self.aCN_space.idx_to_action[aCN]
    act_sn = self.aSN_space.idx_to_action[aSN]
    act_as = self.aAS_space.idx_to_action[aAS]

    sTool_cn = (self.dict_sTools_space[tool_nm].state_to_idx[T3SA.ToolLoc.CN])
    sTool_sn = (self.dict_sTools_space[tool_nm].state_to_idx[T3SA.ToolLoc.SN])
    sTool_as = (self.dict_sTools_space[tool_nm].state_to_idx[T3SA.ToolLoc.AS])
    sTool_flr = (
        self.dict_sTools_space[tool_nm].state_to_idx[T3SA.ToolLoc.FLOOR])

    if (act_cn == T3SA.ActionCN.PICKUP and coord_cn == self.cabinet_loc
        and tool_loc == T3SA.ToolLoc.CABINET):
      return np.array([[1.0, sTool_cn]])
    elif (act_cn == T3SA.ActionCN.PICKUP and coord_cn == self.storage_loc
          and tool_loc == T3SA.ToolLoc.STORAGE):
      return np.array([[1.0, sTool_cn]])
    elif (act_cn == T3SA.ActionCN.HANDOVER and coord_cn == self.handover_loc
          and tool_loc == T3SA.ToolLoc.CN):
      return np.array([[1.0, sTool_sn]])
    elif (act_sn == T3SA.ActionSN.HO_SCALPEL
          and tool_nm == T3SA.ToolNames.SCALPEL_P
          and tool_loc == T3SA.ToolLoc.SN):
      list_next_p_sTool = []
      list_next_p_sTool.append([0.9, sTool_as])
      list_next_p_sTool.append([0.1, sTool_flr])
      return np.array(list_next_p_sTool)
    elif (act_sn == T3SA.ActionSN.HO_SUTURE
          and tool_nm == T3SA.ToolNames.SUTURE_P
          and tool_loc == T3SA.ToolLoc.SN):
      list_next_p_sTool = []
      list_next_p_sTool.append([0.9, sTool_as])
      list_next_p_sTool.append([0.1, sTool_flr])
      return np.array(list_next_p_sTool)
    elif (act_as == T3SA.ActionAS.HO_SCALPEL
          and tool_nm == T3SA.ToolNames.SCALPEL_P
          and tool_loc == T3SA.ToolLoc.AS):
      list_next_p_sTool = []
      list_next_p_sTool.append([0.9, sTool_sn])
      list_next_p_sTool.append([0.1, sTool_flr])
      return np.array(list_next_p_sTool)
    else:
      return np.array([[1.0, sTool]])

  def _T_sCNPos(self, sCNPos, aCN):
    coord_cn = self.sCNPos_space.idx_to_state[sCNPos]
    action_cn = self.aCN_space.idx_to_action[aCN]

    def bound(coord):
      x, y = coord
      if x < 0:
        x = 0
      elif x >= self.num_x_grid:
        x = self.num_x_grid - 1

      if y < 0:
        y = 0
      elif y >= self.num_y_grid:
        y = self.num_y_grid - 1

      return (x, y)

    def is_valid_move(coord, coord_new):
      'assuem sum of coord increment is 0 or 1'
      x_a, y_a = coord
      x_b, y_b = coord_new
      # make a always smaller than or equal to b
      if x_a > x_b:
        x_a, x_b = x_b, x_a
      if y_a > y_b:
        y_a, y_b = y_b, y_a

      for x_d, y_d in self.doors:
        if y_a == y_b and y_a == y_d and x_a < x_d and x_b > x_d:
          return True
        if x_a == x_b and x_a == x_d and y_a < y_d and y_b > y_d:
          return True

      for x_w, y_w in self.walls:
        if y_a == y_b and y_a == y_w and x_a < x_w and x_b > x_w:
          return False
        if x_a == x_b and x_a == x_w and y_a < y_w and y_b > y_w:
          return False

      return True

    cn_move_actions = [
        T3SA.ActionCN.MOVE_DOWN, T3SA.ActionCN.MOVE_UP, T3SA.ActionCN.MOVE_LEFT,
        T3SA.ActionCN.MOVE_RIGHT
    ]

    if action_cn not in cn_move_actions:
      return np.array([[1., sCNPos]])
    else:
      new_coord = coord_cn
      if action_cn == T3SA.ActionCN.MOVE_DOWN:
        new_coord = bound((coord_cn[0], coord_cn[1] + 1))
      elif action_cn == T3SA.ActionCN.MOVE_UP:
        new_coord = bound((coord_cn[0], coord_cn[1] - 1))
      elif action_cn == T3SA.ActionCN.MOVE_LEFT:
        new_coord = bound((coord_cn[0] - 1, coord_cn[1]))
      elif action_cn == T3SA.ActionCN.MOVE_RIGHT:
        new_coord = bound((coord_cn[0] + 1, coord_cn[1]))

      if not is_valid_move(coord_cn, new_coord):
        new_coord = coord_cn

      return np.array([[1., self.sCNPos_space.state_to_idx[new_coord]]])

  def _T_sAsked(self, sAsked, aSN):
    state = self.sAsked_space.idx_to_state[sAsked]
    action = self.aSN_space.idx_to_action[aSN]

    if (action == T3SA.ActionSN.ASKTOOL and state == T3SA.StateAsked.NOT_ASKED):
      return np.array(
          [[1., self.sAsked_space.state_to_idx[T3SA.StateAsked.ASKED]]])
    else:
      return np.array([[1., sAsked]])

  # TODO: make this method generic
  def T_CN(self, sScalpel_s, sSuture_s, sCNPos, aCN):
    fixed_sScalpel_p = self.dict_sTools_space[
        T3SA.ToolNames.SCALPEL_P].state_to_idx[T3SA.ToolLoc.SN]
    fixed_sSuture_p = self.dict_sTools_space[
        T3SA.ToolNames.SUTURE_P].state_to_idx[T3SA.ToolLoc.SN]

    fixed_sPatient = self.sPatient_space.state_to_idx[
        T3SA.StatePatient.NO_INCISION]
    fixed_sAsked = self.sAsked_space.state_to_idx[T3SA.StateAsked.ASKED]
    fixed_aSN = self.aSN_space.action_to_idx[T3SA.ActionSN.STAY]
    fixed_aAS = self.aAS_space.action_to_idx[T3SA.ActionAS.STAY]

    # to check invalid actions
    self._raise_exception_if_illegal_T(fixed_sScalpel_p, fixed_sSuture_p,
                                       sScalpel_s, sSuture_s, fixed_sPatient,
                                       sCNPos, fixed_sAsked, aCN, fixed_aSN,
                                       fixed_aAS)

    np_next_p_sScalpel_s = self._T_sTool(tool_nm=T3SA.ToolNames.SCALPEL_S,
                                         sTool=sScalpel_s,
                                         sCNPos=sCNPos,
                                         aCN=aCN,
                                         aSN=fixed_aSN,
                                         aAS=fixed_aAS)
    np_next_p_sSuture_s = self._T_sTool(tool_nm=T3SA.ToolNames.SUTURE_S,
                                        sTool=sSuture_s,
                                        sCNPos=sCNPos,
                                        aCN=aCN,
                                        aSN=fixed_aSN,
                                        aAS=fixed_aAS)
    np_next_p_sCNPos = self._T_sCNPos(sCNPos=sCNPos, aCN=aCN)

    dict_next_p_state_idx = {}
    for p_sS, sS_next in np_next_p_sScalpel_s:
      for p_sF, sF_next in np_next_p_sSuture_s:
        for p_sC, sC_next in np_next_p_sCNPos:
          next_p = p_sS * p_sF * p_sC
          next_state = (sS_next.astype(np.int32), sF_next.astype(np.int32),
                        sC_next.astype(np.int32))
          dict_next_p_state_idx[next_state] = next_p

    list_next_p_state_idx = []
    for next_state_idx in dict_next_p_state_idx:
      next_p = dict_next_p_state_idx.get(next_state_idx)
      list_next_p_state_idx.append((next_p, next_state_idx))

    return list_next_p_state_idx


if __name__ == "__main__":
  task_mdp = ToolDeliveryMDP_V3()
  sCoord = task_mdp.sCNPos_space.state_to_idx[(1, 3)]
  aCN = task_mdp.aCN_space.action_to_idx[T3SA.ActionCN.MOVE_RIGHT]
  np_new = task_mdp._T_sCNPos(sCoord, aCN)
  sCoord_n = int(np_new[0][1])
  print(task_mdp.sCNPos_space.idx_to_state[sCoord_n])
