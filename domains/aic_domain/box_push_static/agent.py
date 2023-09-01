from typing import Sequence
import os
import numpy as np
from aic_core.models.policy import CachedPolicyInterface
from aic_domain.box_push.agent_model import BoxPushAM
from aic_domain.box_push_static.mdp import StaticBoxPushMDP
from aic_domain.agent import AIAgent_Abstract
from aic_domain.box_push.maps import TUTORIAL_MAP

GAME_MAP = TUTORIAL_MAP

policy_static_list = []


class StaticBoxPushPolicy(CachedPolicyInterface):

  def __init__(self, mdp: StaticBoxPushMDP, temperature: float,
               agent_idx: int) -> None:
    cur_dir = os.path.dirname(__file__)
    str_fileprefix = os.path.join(cur_dir, "data/box_push_np_q_value_static_")
    super().__init__(mdp,
                     str_fileprefix,
                     policy_static_list,
                     temperature,
                     queried_agent_indices=(agent_idx, ))


class StaticBoxPushAM(BoxPushAM):

  def initial_mental_distribution(self, obstate_idx: int) -> np.ndarray:
    mdp = self.get_reference_mdp()  # type: StaticBoxPushMDP

    np_bx = np.ones(mdp.num_latents) / mdp.num_latents
    return np_bx

  def transition_mental_state(self, latstate_idx: int, obstate_idx: int,
                              tuple_action_idx: Sequence[int],
                              obstate_next_idx: int) -> np.ndarray:
    'should not be called'
    raise NotImplementedError


class StaticBoxPushAgent(AIAgent_Abstract):

  def __init__(self, policy_model: CachedPolicyInterface, agent_idx) -> None:
    self.agent_idx = agent_idx
    super().__init__(policy_model, has_mind=True)

  def _create_agent_model(self, policy_model: CachedPolicyInterface):
    return StaticBoxPushAM(self.agent_idx, policy_model=policy_model)

  def update_mental_state(self, tup_cur_state, tup_actions, tup_nxt_state):
    'do nothing'
    pass
