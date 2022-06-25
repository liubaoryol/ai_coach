from ai_coach_core.utils.mdp_utils import StateSpace
from ai_coach_domain.box_push.mdp import BoxPushTeamMDP_AloneOrTogether
from ai_coach_domain.box_push.defines import BoxState, EventType


class StaticBoxPushMDP(BoxPushTeamMDP_AloneOrTogether):
  def init_latentspace(self):
    latent_states = []
    latent_states.append(("alone", 0))
    latent_states.append(("together", 0))
    self.latent_space = StateSpace(latent_states)

  def reward(self, latent_idx: int, state_idx: int, action_idx: int) -> float:
    if self.is_terminal(state_idx):
      return 0

    len_s_space = len(self.dict_factored_statespace)
    state_vec = self.conv_idx_to_state(state_idx)

    a1_pos = self.pos1_space.idx_to_state[state_vec[0]]
    a2_pos = self.pos2_space.idx_to_state[state_vec[1]]

    box_states = []
    for idx in range(2, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.dict_factored_statespace[idx].idx_to_state[box_sidx]
      box_states.append(box_state)

    act1, act2 = self.conv_mdp_aidx_to_sim_actions(action_idx)
    latent = self.latent_space.idx_to_state[latent_idx]

    with_a1 = -1
    with_a2 = -1
    with_both = -1
    for idx, bstate in enumerate(box_states):
      if bstate[0] == BoxState.WithAgent1:
        with_a1 = idx
      elif bstate[0] == BoxState.WithAgent2:
        with_a2 = idx
      elif bstate[0] == BoxState.WithBoth:
        with_both = idx

    move_actions = [
        EventType.UP, EventType.DOWN, EventType.LEFT, EventType.RIGHT
    ]

    panelty = -1
    if latent[0] == "together":
      if with_a1 >= 0 and act1 in move_actions:
        panelty += -5

      if with_a2 >= 0 and act2 in move_actions:
        panelty += -5

    if (with_a1 >= 0 and a1_pos in self.goals and act1 == EventType.UNHOLD
        and (a2_pos != a1_pos or act2 != EventType.HOLD)):
      return 100

    if (with_a2 >= 0 and a2_pos in self.goals and act2 == EventType.UNHOLD
        and (a1_pos != a2_pos or act1 != EventType.HOLD)):
      return 100

    if (with_both >= 0 and a1_pos in self.goals and act1 == EventType.UNHOLD
        and act2 == EventType.UNHOLD):
      return 100

    return panelty
