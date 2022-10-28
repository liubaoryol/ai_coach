import copy
from typing import Callable, Sequence
import numpy as np
from ai_coach_core.utils.mdp_utils import StateSpace
from ai_coach_core.models.mdp import MDP


class FullMDP(MDP):

  def __init__(self, mmdp: MDP, cb_tx: Callable,
               tup_lstate: Sequence[StateSpace]):
    self.mmdp = mmdp
    self.cb_tx = cb_tx
    self.tup_lstate = tup_lstate
    super().__init__(False, use_sparse=True)

  def is_terminal(self, state_idx: int):
    state_vec = self.conv_idx_to_state(state_idx)
    obs_vec = state_vec[:self.mmdp.num_state_factors]
    obs_idx = self.mmdp.conv_state_to_idx(tuple(obs_vec))
    return self.mmdp.is_terminal(obs_idx)

  def legal_actions(self, state_idx: int):
    if self.is_terminal(state_idx):
      return []

    # return super().legal_actions(state_idx)

    state_vec = self.conv_idx_to_state(state_idx)
    obs_vec = state_vec[:self.mmdp.num_state_factors]
    obs_idx = self.mmdp.conv_state_to_idx(tuple(obs_vec))
    return self.mmdp.legal_actions(obs_idx)

  def init_statespace(self):
    self.dict_factored_statespace = copy.copy(
        self.mmdp.dict_factored_statespace)
    num_obstate_factors = len(self.dict_factored_statespace)
    for idx in range(len(self.tup_lstate)):
      idx_s = num_obstate_factors + idx
      self.dict_factored_statespace[idx_s] = self.tup_lstate[idx]

    self.dummy_states = None

  def init_actionspace(self):
    self.dict_factored_actionspace = self.mmdp.dict_factored_actionspace

  def conv_sim_states_to_mdp_sidx(self, tup_states):
    return super().conv_sim_states_to_mdp_sidx(tup_states)

  def conv_mdp_sidx_to_sim_states(self, state_idx):
    return super().conv_mdp_sidx_to_sim_states(state_idx)

  def conv_mdp_aidx_to_sim_actions(self, action_idx):
    return self.mmdp.conv_mdp_aidx_to_sim_actions(action_idx)

  def conv_sim_actions_to_mdp_aidx(self, tuple_actions):
    return self.mmdp.conv_sim_actions_to_mdp_aidx(tuple_actions)

  def transition_model(self, state_idx: int, action_idx: int) -> np.ndarray:
    num_obs_dim = self.mmdp.num_state_factors

    vec_factored_sidx = self.conv_idx_to_state(state_idx)
    vec_obs_idx = vec_factored_sidx[:num_obs_dim]

    vec_factored_aidx = self.conv_idx_to_action(action_idx)

    obs_idx = self.mmdp.conv_state_to_idx(tuple(vec_obs_idx))
    np_next_p_obs_idx = self.mmdp.transition_model(obs_idx, action_idx)

    list_next_p_sidx = []
    for p_obs_n, oidx_n in np_next_p_obs_idx:
      np_next_p_xidx = np.ones(self.list_num_states[num_obs_dim:])
      for agent_idx in range(len(self.tup_lstate)):
        idx_s = num_obs_dim + agent_idx
        xidx = vec_factored_sidx[idx_s]

        np_next_xidx_dist = self.cb_tx(agent_idx, xidx, obs_idx,
                                       vec_factored_aidx, int(oidx_n))
        shape = [1] * len(self.tup_lstate)
        shape[agent_idx] = self.list_num_states[idx_s]
        np_next_p_xidx = np_next_p_xidx * np_next_xidx_dist.reshape(shape)

      for vec_lat_n, p_lat_n in np.ndenumerate(np_next_p_xidx):
        p_next = p_obs_n * p_lat_n
        if p_next != 0:
          vec_obs_idx_n = self.mmdp.conv_idx_to_state(int(oidx_n))
          vec_factored_n = tuple(vec_obs_idx_n) + vec_lat_n
          sidx_n = self.conv_state_to_idx(vec_factored_n)
          list_next_p_sidx.append((p_next, sidx_n))

    return np.array(list_next_p_sidx)

  def reward(self, state_idx: int, action_idx: int) -> float:
    if self.is_terminal(state_idx):
      return 0

    return -1
