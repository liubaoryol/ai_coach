import numpy as np
from tests.examples.environment import RequestEnvironment
from tests.examples.tooldelivery_v3_mdp import ToolDeliveryMDP_V3
from tests.examples.tooldelivery_v3_policy import ToolDeliveryPolicy_V3
import tests.examples.tooldelivery_v3_state_action as T3SA


class ToolDeliveryEnv_V3(RequestEnvironment):
  def __init__(self, use_policy=True):
    td_mmdp = ToolDeliveryMDP_V3()
    policy = ToolDeliveryPolicy_V3(td_mmdp) if use_policy else None
    super().__init__(td_mmdp, policy, 2)

    self.p_CN_latent_scalpel = 0.5
    self.map_agent_2_brain[0] = 0
    self.map_agent_2_brain[1] = 1  # SN and AS maps to the same brain
    self.map_agent_2_brain[2] = 1  # because they share their mental model

  def get_initial_state_dist(self):
    np_init_p_sScalpel_p = np.array([[
        1., self.mmdp.dict_sTools_space[T3SA.ToolNames.SCALPEL_P].state_to_idx[
            T3SA.ToolLoc.SN]
    ]])
    np_init_p_sSuture_p = np.array([[
        1., self.mmdp.dict_sTools_space[T3SA.ToolNames.SUTURE_P].state_to_idx[
            T3SA.ToolLoc.SN]
    ]])
    np_init_p_sScalpel_s = np.array([[
        1., self.mmdp.dict_sTools_space[T3SA.ToolNames.SCALPEL_S].state_to_idx[
            T3SA.ToolLoc.STORAGE]
    ]])
    np_init_p_sSuture_s = np.array([[
        1., self.mmdp.dict_sTools_space[T3SA.ToolNames.SUTURE_S].state_to_idx[
            T3SA.ToolLoc.CABINET]
    ]])
    np_init_p_sPatient = np.array([[
        1., self.mmdp.sPatient_space.state_to_idx[T3SA.StatePatient.NO_INCISION]
    ]])
    np_init_p_sCNPos = np.array(
        [[1., self.mmdp.sCNPos_space.state_to_idx[self.mmdp.handover_loc]]])
    np_init_p_sAsked = np.array(
        [[1., self.mmdp.sAsked_space.state_to_idx[T3SA.StateAsked.NOT_ASKED]]])

    dict_init_p_state_idx = {}
    for p_sSa, sSa in np_init_p_sScalpel_p:
      for p_sFa, sFa in np_init_p_sSuture_p:
        for p_sSb, sSb in np_init_p_sScalpel_s:
          for p_sFb, sFb in np_init_p_sSuture_s:
            for p_sPat, sPat in np_init_p_sPatient:
              for p_sPos, sPos in np_init_p_sCNPos:
                for p_sAsk, sAsk in np_init_p_sAsked:
                  init_p = (p_sSa * p_sFa * p_sSb * p_sFb * p_sPat * p_sPos *
                            p_sAsk)
                  state_idx = self.mmdp.np_state_to_idx[sSa.astype(np.int32),
                                                        sFa.astype(np.int32),
                                                        sSb.astype(np.int32),
                                                        sFb.astype(np.int32),
                                                        sPat.astype(np.int32),
                                                        sPos.astype(np.int32),
                                                        sAsk.astype(np.int32)]
                  dict_init_p_state_idx[state_idx] = init_p

    np_init_p_state_idx = np.zeros((len(dict_init_p_state_idx), 2))
    iter_idx = 0
    for state_idx in dict_init_p_state_idx:
      np_next_p = dict_init_p_state_idx.get(state_idx)
      np_init_p_state_idx[iter_idx] = np_next_p, state_idx
      iter_idx += 1

    return np_init_p_state_idx

  def is_terminal_state(self, obstate_idx):
    state_vector = self.mmdp.np_idx_to_state[obstate_idx]
    sScal_p, sSut_p, sScal_s, sSut_s, sPat, sPos, sAsk = state_vector

    s_scal_s = self.mmdp.dict_sTools_space[
        T3SA.ToolNames.SCALPEL_S].idx_to_state[sScal_s]
    s_sut_s = self.mmdp.dict_sTools_space[
        T3SA.ToolNames.SUTURE_S].idx_to_state[sSut_s]
    s_sut_p = self.mmdp.dict_sTools_space[
        T3SA.ToolNames.SUTURE_P].idx_to_state[sSut_p]
    s_patient = self.mmdp.sPatient_space.idx_to_state[sPat]

    if (s_scal_s == T3SA.ToolLoc.SN or s_sut_s == T3SA.ToolLoc.SN):
      return True

    if (s_sut_p == T3SA.ToolLoc.AS
        and s_patient == T3SA.StatePatient.NO_INCISION):
      return True

    return False

  def get_latentstate(self, obstate_idx):
    def choice_latent_state(np_prior_p_latent):
      lat_state = None
      if np_prior_p_latent[0][0] == 1.0:
        lat_state = np_prior_p_latent[0][1]
      else:
        lat_choice = np.random.choice(np_prior_p_latent[:, 1],
                                      1,
                                      p=np_prior_p_latent[:, 0])
        lat_state = lat_choice[0].astype(np.int32)
      return lat_state.astype(np.int32)

    np_cn_prior, np_sn_prior = self.get_latentstate_prior(obstate_idx)
    if np_cn_prior is None or np_sn_prior is None:
      return None
    else:
      cn_lat = choice_latent_state(np_cn_prior)
      sn_lat = choice_latent_state(np_sn_prior)
      return cn_lat, sn_lat

  def set_CN_latent_scalpel_probability(self, p_scalpel=0.5):
    self.p_CN_latent_scalpel = p_scalpel

  def get_latentstate_prior(self, obstate_idx):
    '''
        this method is valid only at the onset of StateAsked changing to 1.
        please do not use for other time steps
        '''
    np_cn_p_latent = None
    np_sn_p_latent = None
    if self.is_initiated_state(obstate_idx):
      state_vector = self.mmdp.np_idx_to_state[obstate_idx]
      sScal_p, sSut_p, sScal_s, sSut_s, sPat, sPos, sAsk = state_vector

      s_scal_p = self.mmdp.dict_sTools_space[
          T3SA.ToolNames.SCALPEL_P].idx_to_state[sScal_p]
      s_sut_p = self.mmdp.dict_sTools_space[
          T3SA.ToolNames.SUTURE_P].idx_to_state[sSut_p]

      if s_scal_p == T3SA.ToolLoc.FLOOR:
        np_sn_p_latent = np.array([
            [1.0, T3SA.LatentState.SCALPEL.value],
        ])
      elif s_sut_p == T3SA.ToolLoc.FLOOR:
        np_sn_p_latent = np.array([[1.0, T3SA.LatentState.SUTURE.value]])
      elif s_scal_p == T3SA.ToolLoc.AS:
        np_sn_p_latent = np.array([[2.0 / 3.0, T3SA.LatentState.SCALPEL.value],
                                   [1.0 / 3.0, T3SA.LatentState.SUTURE.value]])
      elif s_sut_p == T3SA.ToolLoc.AS:
        np_sn_p_latent = np.array([[1.0 / 3.0, T3SA.LatentState.SCALPEL.value],
                                   [2.0 / 3.0, T3SA.LatentState.SUTURE.value]])

      if self.p_CN_latent_scalpel == 1.0:
        np_cn_p_latent = np.array([[1.0, T3SA.LatentState.SCALPEL.value]])
      elif self.p_CN_latent_scalpel == 0.0:
        np_cn_p_latent = np.array([[1.0, T3SA.LatentState.SUTURE.value]])
      else:
        np_cn_p_latent = np.array(
            [[self.p_CN_latent_scalpel, T3SA.LatentState.SCALPEL.value],
             [1.0 - self.p_CN_latent_scalpel, T3SA.LatentState.SUTURE.value]])

    return np_cn_p_latent, np_sn_p_latent

  def is_initiated_state(self, state_idx):
    state_vector = self.mmdp.np_idx_to_state[state_idx]
    sAsked = state_vector[self.mmdp.num_tools + 2]
    idx_to_state = self.mmdp.sAsked_space.idx_to_state

    return idx_to_state[sAsked] == T3SA.StateAsked.ASKED


if __name__ == "__main__":
  env = ToolDeliveryEnv_V3()
  np_init_p_state = env.get_initial_state_dist()
  start_idx = np_init_p_state[0][1]
  list_sequence, list_latent_state = env.generate_sequence(
      start_idx.astype(np.int32),
      timeout=1000,
      save=True,
      file_name='./examples/sequence4.txt')

  print(list_latent_state)
