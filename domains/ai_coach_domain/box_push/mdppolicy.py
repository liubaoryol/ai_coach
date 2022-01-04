import os
import pickle
from typing import Sequence
import numpy as np
import ai_coach_core.models.mdp as mdp_lib
from ai_coach_core.models.agent_model import PolicyInterface
import ai_coach_core.RL.planning as plan_lib
from ai_coach_domain.box_push.mdp import (BoxPushMDP, BoxPushTeamMDP,
                                          BoxPushAgentMDP,
                                          BoxPushTeamMDP_AlwaysTogether,
                                          BoxPushAgentMDP_AlwaysAlone)

policy_exp1_list = []
policy_indv_list = []
policy_test_agent_list = []
policy_test_team_list = []


class BoxPushPolicyInterface(PolicyInterface):
  def __init__(
      self,
      mdp: BoxPushMDP,
      file_prefix: str,
      list_policy: list,
      temperature: float,
      queried_agent_indices: Sequence[int] = (0, )) -> None:
    super().__init__(mdp)
    self.list_policy = list_policy
    self.file_prefix = file_prefix
    self.temperature = temperature
    self.queried_agent_indices = queried_agent_indices

  def prepare_policy(self):
    if (self.list_policy is None) or len(self.list_policy) == 0:
      # cur_dir = os.path.dirname(__file__)
      for idx in range(self.mdp.num_latents):
        # str_q_val = os.path.join(
        #     cur_dir, "data/" + self.file_prefix + "%d.pickle" % (idx, ))
        str_q_val = self.file_prefix + "%d.pickle" % (idx, )
        np_q_value = None
        GAMMA = 0.95
        if self.file_prefix == "" or (not os.path.exists(str_q_val)):
          _, _, np_q_value = plan_lib.value_iteration(
              self.mdp.np_transition_model,
              self.mdp.np_reward_model[idx],
              discount_factor=GAMMA,
              max_iteration=500,
              epsilon=0.01)

          if self.file_prefix != "":
            with open(str_q_val, "wb") as f:
              pickle.dump(np_q_value, f, pickle.HIGHEST_PROTOCOL)
        else:
          with open(str_q_val, "rb") as f:
            np_q_value = pickle.load(f)

        self.list_policy.append(
            mdp_lib.softmax_policy_from_q_value(np_q_value, self.temperature))

  # TODO: not tested yet - test this method
  def policy(self, obstate_idx: int, latstate_idx: int) -> np.ndarray:
    'slow'
    self.prepare_policy()

    np_action_dist = np.reshape(self.list_policy[latstate_idx][obstate_idx, :],
                                self.mdp.list_num_actions)

    axis2sum = [
        idx for idx in range(self.mdp.num_action_factors)
        if idx not in self.queried_agent_indices
    ]
    np_action_dist = np_action_dist.sum(
        axis=tuple(axis2sum))  # type: np.ndarray
    return np_action_dist.ravel()

  def get_action(self, obstate_idx: int, latstate_idx: int):
    self.prepare_policy()

    joint_aidx = np.random.choice(
        range(self.mdp.num_actions),
        size=1,
        replace=False,
        p=self.list_policy[latstate_idx][obstate_idx, :])[0]
    vector_indv_aidx = self.mdp.conv_idx_to_action(joint_aidx)
    return vector_indv_aidx[list(self.queried_agent_indices)]

  def conv_idx_to_action(self, tuple_aidx: Sequence[int]):
    list_actions = []
    for idx, fidx in enumerate(self.queried_agent_indices):
      list_actions.append(
          self.mdp.dict_factored_actionspace[fidx].idx_to_action[
              tuple_aidx[idx]])

    return list_actions

  def conv_action_to_idx(self, tuple_actions: Sequence) -> Sequence[int]:
    list_aidx = []
    for idx, fidx in enumerate(self.queried_agent_indices):
      list_aidx.append(self.mdp.dict_factored_actionspace[fidx].action_to_idx[
          tuple_actions[idx]])

    return list_aidx


class BoxPushPolicyTeamExp1(BoxPushPolicyInterface):
  def __init__(self, mdp: BoxPushTeamMDP_AlwaysTogether, temperature: float,
               agent_idx: int) -> None:
    cur_dir = os.path.dirname(__file__)
    str_fileprefix = os.path.join(cur_dir, "data/box_push_np_q_value_exp1_")
    super().__init__(mdp, str_fileprefix, policy_exp1_list, temperature,
                     (agent_idx, ))
    # TODO: check if mdp has the same configuration as EXP1_MAP


class BoxPushPolicyIndvExp1(BoxPushPolicyInterface):
  def __init__(self, mdp: BoxPushAgentMDP_AlwaysAlone,
               temperature: float) -> None:
    cur_dir = os.path.dirname(__file__)
    str_fileprefix = os.path.join(cur_dir, "data/box_push_np_q_value_indv_")
    super().__init__(mdp, str_fileprefix, policy_indv_list, temperature)


class BoxPushPolicyTeamTest(BoxPushPolicyInterface):
  def __init__(self, mdp: BoxPushTeamMDP, temperature: float,
               agent_idx: int) -> None:
    super().__init__(mdp, "", policy_test_team_list, temperature, (agent_idx, ))


class BoxPushPolicyIndvTest(BoxPushPolicyInterface):
  def __init__(self, mdp: BoxPushAgentMDP, temperature: float) -> None:
    super().__init__(mdp, "", policy_test_agent_list, temperature)
