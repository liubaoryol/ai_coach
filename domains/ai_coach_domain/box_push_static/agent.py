from typing import Sequence
import os
import numpy as np
from ai_coach_domain.box_push.mdpagent import BoxPushMDPAgent
from ai_coach_domain.box_push.mdppolicy import BoxPushPolicyInterface
from ai_coach_domain.box_push_static.mdp import StaticBoxPushMDP
from ai_coach_domain.box_push.agent import BoxPushAIAgent_Abstract
from ai_coach_domain.box_push.maps import TUTORIAL_MAP

GAME_MAP = TUTORIAL_MAP

policy_static_list = []


class StaticBoxPushPolicy(BoxPushPolicyInterface):
  def __init__(self, mdp: StaticBoxPushMDP, temperature: float,
               agent_idx: int) -> None:
    cur_dir = os.path.dirname(__file__)
    str_fileprefix = os.path.join(cur_dir, "data/box_push_np_q_value_static_")
    super().__init__(mdp,
                     str_fileprefix,
                     policy_static_list,
                     temperature,
                     queried_agent_indices=(agent_idx, ))


class StaticBoxPushMDPAgent(BoxPushMDPAgent):
  def initial_mental_distribution(self, obstate_idx: int) -> np.ndarray:
    mdp = self.get_reference_mdp()  # type: StaticBoxPushMDP

    np_bx = np.ones(mdp.num_latents) / mdp.num_latents
    return np_bx

  def transition_mental_state(self, latstate_idx: int, obstate_idx: int,
                              tuple_action_idx: Sequence[int],
                              obstate_next_idx: int) -> np.ndarray:
    'should not be called'
    raise NotImplementedError


class StaticBoxPushAgent(BoxPushAIAgent_Abstract):
  def __init__(self, policy_model: BoxPushPolicyInterface, agent_idx) -> None:
    self.agent_idx = agent_idx
    super().__init__(policy_model, use_flipped_state_space=False, has_mind=True)

  def _create_agent_model(self, policy_model: BoxPushPolicyInterface):
    return StaticBoxPushMDPAgent(self.agent_idx, policy_model=policy_model)

  def update_mental_state(self, tup_cur_state, tup_actions, tup_nxt_state):
    'do nothing'
    pass