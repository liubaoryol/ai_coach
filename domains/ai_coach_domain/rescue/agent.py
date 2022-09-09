from typing import Sequence, Optional
import numpy as np
from ai_coach_core.models.agent_model import AgentModel
from ai_coach_core.models.policy import PolicyInterface, CachedPolicyInterface
from ai_coach_domain.agent import SimulatorAgent
from ai_coach_domain.rescue import AGENT_ACTIONSPACE, E_EventType, is_work_done
from ai_coach_domain.rescue.mdp import MDP_Rescue


def assumed_initial_mental_distribution(agent_idx: int, obstate_idx: int,
                                        mdp: MDP_Rescue):
  '''
      obstate_idx: absolute (task-perspective) state representation.
  '''
  work_states, _, _ = mdp.conv_mdp_sidx_to_sim_states(obstate_idx)

  np_work_states = np.array(work_states)
  np_bx = np_work_states / np.sum(np_work_states)

  return np_bx


class RescueAM(AgentModel):
  def __init__(self,
               agent_idx: int,
               policy_model: Optional[PolicyInterface] = None) -> None:
    super().__init__(policy_model)
    self.agent_idx = agent_idx

  def initial_mental_distribution(self, obstate_idx: int) -> np.ndarray:
    '''
        assume agent1 and 2 has the same policy
        state_idx: absolute (task-perspective) state representation.
                    For here, we assume agent1 state and task state is the same
        '''
    mdp = self.get_reference_mdp()  # type: MDP_Rescue
    return assumed_initial_mental_distribution(self.agent_idx, obstate_idx, mdp)

  def transition_mental_state(self, latstate_idx: int, obstate_idx: int,
                              tuple_action_idx: Sequence[int],
                              obstate_next_idx: int) -> np.ndarray:
    '''
    obstate_idx: absolute (task-perspective) state representation.
          Here, we assume agent1 state space and task state space is the same
    '''
    mdp = self.get_reference_mdp()  # type: MDP_Rescue

    work_states_cur, a1_pos, a2_pos = mdp.conv_mdp_sidx_to_sim_states(
        obstate_idx)
    work_states_nxt, _, _ = mdp.conv_mdp_sidx_to_sim_states(obstate_next_idx)

    my_loc = a1_pos if self.agent_idx == 0 else a2_pos
    mate_loc = a2_pos if self.agent_idx == 0 else a1_pos

    my_aidx = tuple_action_idx[self.agent_idx]
    my_act = AGENT_ACTIONSPACE.idx_to_action[my_aidx]

    if work_states_cur != work_states_nxt:
      if is_work_done(latstate_idx, work_states_nxt,
                      mdp.work_info[latstate_idx].coupled_works):
        np_work_states_nxt = np.array(work_states_nxt)
        for idx, _ in enumerate(work_states_nxt):
          if is_work_done(idx, work_states_nxt,
                          mdp.work_info[idx].coupled_works):
            np_work_states_nxt[idx] = 0

        return np_work_states_nxt / np.sum(np_work_states_nxt)

    P_CHANGE = 0.1
    if mate_loc in mdp.work_locations:
      widx = mdp.work_locations.index(mate_loc)
      wstate_n = work_states_nxt[widx]
      workload = mdp.work_info[widx].workload
      if wstate_n != 0 and workload > 1:
        np_Tx = np.zeros(len(work_states_cur))
        np_Tx[widx] += P_CHANGE
        np_Tx[latstate_idx] += 1 - P_CHANGE
        return np_Tx

    np_Tx = np.zeros(len(work_states_cur))
    np_Tx[latstate_idx] = 1
    return np_Tx


class AIAgent_Rescue(SimulatorAgent):
  def __init__(self,
               agent_idx,
               policy_model: CachedPolicyInterface,
               has_mind: bool = True) -> None:
    super().__init__(has_mind=has_mind, has_policy=True)
    self.agent_idx = agent_idx
    self.agent_model = self._create_agent_model(policy_model)
    self.manual_action = None

  def _create_agent_model(self,
                          policy_model: CachedPolicyInterface) -> RescueAM:
    'Should be implemented at inherited method'
    return RescueAM(agent_idx=self.agent_idx, policy_model=policy_model)

  def init_latent(self, tup_states):
    mdp = self.agent_model.get_reference_mdp()
    sidx = mdp.conv_sim_states_to_mdp_sidx(tup_states)

    self.agent_model.set_init_mental_state_idx(sidx)

  def get_current_latent(self):
    if self.agent_model.is_current_latent_valid():
      return self.agent_model.policy_model.conv_idx_to_latent(
          self.agent_model.current_latent)
    else:
      return None

  def get_action(self, tup_states):
    if self.manual_action is not None:
      next_action = self.manual_action
      self.manual_action = None
      return next_action

    mdp = self.agent_model.get_reference_mdp()
    sidx = mdp.conv_sim_states_to_mdp_sidx(tup_states)
    tup_aidx = self.agent_model.get_action_idx(sidx)
    return self.agent_model.policy_model.conv_idx_to_action(tup_aidx)[0]

  def update_mental_state(self, tup_cur_state, tup_actions, tup_nxt_state):
    'tup_actions: tuple of actions'

    mdp = self.agent_model.get_reference_mdp()
    sidx_cur = mdp.conv_sim_states_to_mdp_sidx(tup_cur_state)
    sidx_nxt = mdp.conv_sim_states_to_mdp_sidx(tup_nxt_state)

    list_aidx = []
    for idx, act in enumerate(tup_actions):
      list_aidx.append(mdp.dict_factored_actionspace[idx].action_to_idx[act])

    self.agent_model.update_mental_state_idx(sidx_cur, tuple(list_aidx),
                                             sidx_nxt)

  def set_latent(self, latent):
    xidx = self.agent_model.policy_model.conv_latent_to_idx(latent)
    self.agent_model.set_init_mental_state_idx(None, xidx)

  def set_action(self, action):
    self.manual_action = action
