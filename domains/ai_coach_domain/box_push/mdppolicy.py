import os
from ai_coach_core.models.policy import CachedPolicyInterface
from ai_coach_domain.box_push.mdp import (BoxPushTeamMDP, BoxPushAgentMDP,
                                          BoxPushTeamMDP_AlwaysTogether,
                                          BoxPushAgentMDP_AlwaysAlone)

policy_exp1_list = []
policy_indv_list = []
policy_test_agent_list = []
policy_test_team_list = []


class BoxPushPolicyTeamExp1(CachedPolicyInterface):
  def __init__(self, mdp: BoxPushTeamMDP_AlwaysTogether, temperature: float,
               agent_idx: int) -> None:
    cur_dir = os.path.dirname(__file__)
    str_fileprefix = os.path.join(cur_dir, "data/box_push_np_q_value_exp1_")
    super().__init__(mdp, str_fileprefix, policy_exp1_list, temperature,
                     (agent_idx, ))
    # TODO: check if mdp has the same configuration as EXP1_MAP


class BoxPushPolicyIndvExp1(CachedPolicyInterface):
  def __init__(self, mdp: BoxPushAgentMDP_AlwaysAlone,
               temperature: float) -> None:
    cur_dir = os.path.dirname(__file__)
    str_fileprefix = os.path.join(cur_dir, "data/box_push_np_q_value_indv_")
    super().__init__(mdp, str_fileprefix, policy_indv_list, temperature)


class BoxPushPolicyTeamTest(CachedPolicyInterface):
  def __init__(self, mdp: BoxPushTeamMDP, temperature: float,
               agent_idx: int) -> None:
    super().__init__(mdp, "", policy_test_team_list, temperature, (agent_idx, ))


class BoxPushPolicyIndvTest(CachedPolicyInterface):
  def __init__(self, mdp: BoxPushAgentMDP, temperature: float) -> None:
    super().__init__(mdp, "", policy_test_agent_list, temperature)
