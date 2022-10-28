from typing import Sequence
import numpy as np
from ai_coach_core.models.policy import CachedPolicyInterface
from ai_coach_domain.agent import (AIAgent_PartialObs, AIAgent_Abstract,
                                   BTILCachedAgentModel)
from ai_coach_domain.box_push_v2 import (conv_box_idx_2_state,
                                         conv_box_state_2_idx, BoxState)
from ai_coach_domain.box_push_v2.mdp import MDP_BoxPushV2
from ai_coach_domain.box_push_v2.agent_model import (
    AM_BoxPushV2, AM_BoxPushV2_Cleanup, AM_BoxPushV2_Movers,
    assumed_initial_mental_distribution)


class BoxPushAIAgent_PartialObs(AIAgent_PartialObs):
  def observed_states(self, tup_states):
    box_states, a1_pos, a2_pos = tup_states

    mdp = self.agent_model.get_reference_mdp()  # type: MDP_BoxPushV2
    num_drops = len(mdp.drops)
    num_goals = len(mdp.goals)
    assert num_goals == 1
    prev_box_states, prev_a1_pos, prev_a2_pos = self.assumed_tup_states

    def max_dist(my_pos, mate_pos):
      return max(abs(my_pos[0] - mate_pos[0]), abs(my_pos[1] - mate_pos[1]))

    def assumed_state(prev_box_states, prev_my_pos, prev_mate_pos, box_states,
                      my_pos, mate_pos, e_boxstate_with_me: BoxState,
                      e_boxstate_with_mate: BoxState):
      agent_dist = max_dist(my_pos, mate_pos)

      assumed_box_states = list(prev_box_states)

      assumed_my_pos = my_pos
      assumed_mate_pos = prev_mate_pos

      if agent_dist <= 1:
        assumed_mate_pos = mate_pos
      else:
        prev_agent_dist = max_dist(my_pos, prev_mate_pos)
        if prev_agent_dist <= 1:
          possible_coords = [(prev_mate_pos[0], prev_mate_pos[1]),
                             (prev_mate_pos[0] - 1, prev_mate_pos[1]),
                             (prev_mate_pos[0] + 1, prev_mate_pos[1]),
                             (prev_mate_pos[0], prev_mate_pos[1] - 1),
                             (prev_mate_pos[0], prev_mate_pos[1] + 1)]
          for crd in possible_coords:
            if crd[0] < 0 or crd[0] >= mdp.x_grid:
              continue
            if crd[1] < 0 or crd[1] >= mdp.y_grid:
              continue
            if crd in mdp.walls:
              continue
            dist_tmp = max_dist(my_pos, crd)
            if dist_tmp > 1:
              assumed_mate_pos = crd
              break

      for idx, coord in enumerate(mdp.boxes):
        if max_dist(my_pos, coord) <= 1:
          bstate = conv_box_idx_2_state(box_states[idx], num_drops, num_goals)
          if bstate[0] != BoxState.Original:
            assumed_box_states[idx] = conv_box_state_2_idx(
                (BoxState.OnGoalLoc, 0), num_drops)

      for idx, bidx in enumerate(box_states):
        bpos = None
        bstate = conv_box_idx_2_state(bidx, num_drops, num_goals)
        if bstate[0] == BoxState.Original:
          bpos = mdp.boxes[idx]
        elif bstate[0] == e_boxstate_with_me:
          bpos = my_pos
        elif bstate[0] == e_boxstate_with_mate:
          bpos = mate_pos
        elif bstate[0] == BoxState.WithBoth:
          bpos = my_pos
        elif bstate[0] == BoxState.OnDropLoc:
          bpos = mdp.drops[bstate[1]]

        prev_bstate = conv_box_idx_2_state(prev_box_states[idx], num_drops,
                                           num_goals)
        if (prev_bstate[0] in [e_boxstate_with_me, BoxState.WithBoth]
            and bpos is None):
          assumed_box_states[idx] = conv_box_state_2_idx(
              (BoxState.OnGoalLoc, 0), num_drops)
        elif (agent_dist <= 1 and prev_mate_pos == mdp.goals[0]
              and prev_bstate[0] == e_boxstate_with_mate and bpos is None):
          assumed_box_states[idx] = conv_box_state_2_idx(
              (BoxState.OnGoalLoc, 0), num_drops)
        elif bpos is not None:
          if max_dist(my_pos, bpos) <= 1:
            assumed_box_states[idx] = bidx

      return tuple(assumed_box_states), assumed_my_pos, assumed_mate_pos

    if self.agent_idx == 0:
      assumed_box_states, assumed_a1_pos, assumed_a2_pos = assumed_state(
          prev_box_states, prev_a1_pos, prev_a2_pos, box_states, a1_pos, a2_pos,
          BoxState.WithAgent1, BoxState.WithAgent2)
    else:
      assumed_box_states, assumed_a2_pos, assumed_a1_pos = assumed_state(
          prev_box_states, prev_a2_pos, prev_a1_pos, box_states, a2_pos, a1_pos,
          BoxState.WithAgent2, BoxState.WithAgent1)

    return assumed_box_states, assumed_a1_pos, assumed_a2_pos

  def observed_actions(self, tup_actions, tup_nxt_state) -> tuple:
    bstate, a1_pos, a2_pos = self.assumed_tup_states
    observed_actions = [None, None]
    if self.agent_idx == 0:
      observed_actions[0] = tup_actions[0]
      if max(abs(a1_pos[0] - a2_pos[0]), abs(a1_pos[1] - a2_pos[1])) <= 1:
        observed_actions[1] = tup_actions[1]
    else:
      observed_actions[1] = tup_actions[1]
      if max(abs(a1_pos[0] - a2_pos[0]), abs(a1_pos[1] - a2_pos[1])) <= 1:
        observed_actions[0] = tup_actions[0]

    return tuple(observed_actions)


class BoxPushAIAgent_PO_Team(BoxPushAIAgent_PartialObs):
  def __init__(self,
               init_tup_states,
               policy_model: CachedPolicyInterface,
               has_mind: bool = True,
               agent_idx: int = 0) -> None:
    super().__init__(init_tup_states, policy_model, has_mind, agent_idx)

  def _create_agent_model(self,
                          policy_model: CachedPolicyInterface) -> AM_BoxPushV2:
    return AM_BoxPushV2_Movers(agent_idx=self.agent_idx,
                               policy_model=policy_model)


class BoxPushAIAgent_PO_Indv(BoxPushAIAgent_PartialObs):
  def __init__(self,
               init_tup_states,
               policy_model: CachedPolicyInterface,
               has_mind: bool = True,
               agent_idx: int = 0) -> None:
    super().__init__(init_tup_states, policy_model, has_mind, agent_idx)

  def _create_agent_model(self,
                          policy_model: CachedPolicyInterface) -> AM_BoxPushV2:
    return AM_BoxPushV2_Cleanup(self.agent_idx, policy_model)


class BoxPushAIAgent_Team(AIAgent_Abstract):
  def __init__(self,
               policy_model: CachedPolicyInterface,
               has_mind: bool = True,
               agent_idx: int = 0) -> None:
    super().__init__(policy_model, has_mind, agent_idx)

  def _create_agent_model(self,
                          policy_model: CachedPolicyInterface) -> AM_BoxPushV2:
    return AM_BoxPushV2_Movers(agent_idx=self.agent_idx,
                               policy_model=policy_model)


class BoxPushAIAgent_Indv(AIAgent_Abstract):
  def __init__(self,
               policy_model: CachedPolicyInterface,
               has_mind: bool = True,
               agent_idx: int = 0) -> None:
    super().__init__(policy_model, has_mind, agent_idx)

  def _create_agent_model(self,
                          policy_model: CachedPolicyInterface) -> AM_BoxPushV2:
    return AM_BoxPushV2_Cleanup(self.agent_idx, policy_model)


class BoxPushAIAgent_BTIL(AIAgent_Abstract):
  def __init__(self,
               np_tx: np.ndarray,
               mask_sas: Sequence[bool],
               policy_model: CachedPolicyInterface,
               agent_idx: int = 0) -> None:
    self.np_tx = np_tx
    self.mask_sas = mask_sas
    super().__init__(policy_model, True, agent_idx)

  def _create_agent_model(self, policy_model: CachedPolicyInterface):
    def init_latents(obstate_idx):
      return assumed_initial_mental_distribution(self.agent_idx, obstate_idx,
                                                 policy_model.mdp)

    return BTILCachedAgentModel(init_latents, self.np_tx, self.mask_sas,
                                policy_model)
