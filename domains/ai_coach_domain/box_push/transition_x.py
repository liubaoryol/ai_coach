from typing import Union
import numpy as np
from ai_coach_domain.box_push import conv_box_idx_2_state, BoxState
from ai_coach_domain.box_push.mdp import (BoxPushAgentMDP_AlwaysAlone,
                                          BoxPushTeamMDP_AlwaysTogether)


def get_holding_box_and_floor_boxes(box_states, num_drops, num_goals):
  a1_box = -1
  a2_box = -1
  floor_boxes = []
  for idx in range(len(box_states)):
    state = conv_box_idx_2_state(box_states[idx], num_drops, num_goals)
    if state[0] == BoxState.WithAgent1:
      a1_box = idx
    elif state[0] == BoxState.WithAgent2:
      a2_box = idx
    elif state[0] == BoxState.WithBoth:
      a1_box = idx
      a2_box = idx
    elif state[0] in [BoxState.Original, BoxState.OnDropLoc]:
      floor_boxes.append(idx)

  return a1_box, a2_box, floor_boxes


def get_np_Tx_indv(mdp_agent: BoxPushAgentMDP_AlwaysAlone, agent_idx,
                   latent_idx, state_idx, tuple_action_idx, next_state_idx):
  '''
  state_idx: absolute (task-perspective) state representation.
            Here, we assume agent1 state space and task state space is the same
  '''
  num_drops = len(mdp_agent.drops)
  num_goals = len(mdp_agent.goals)

  _, _, box_states_cur = mdp_agent.conv_mdp_sidx_to_sim_states(state_idx)
  _, _, box_states_nxt = mdp_agent.conv_mdp_sidx_to_sim_states(next_state_idx)

  a1_box_cur, a2_box_cur, _ = get_holding_box_and_floor_boxes(
      box_states_cur, num_drops, num_goals)
  a1_box_nxt, a2_box_nxt, valid_boxes = get_holding_box_and_floor_boxes(
      box_states_nxt, num_drops, num_goals)

  def get_np_Tx_indv_impl(my_box_cur, mate_box_cur, my_box_nxt, mate_box_nxt):
    my_pickup = my_box_cur < 0 and my_box_nxt >= 0
    my_drop = my_box_cur >= 0 and my_box_nxt < 0
    mate_pickup = mate_box_cur < 0 and mate_box_nxt >= 0

    num_valid_box = len(valid_boxes)
    np_Tx = np.zeros(mdp_agent.num_latents)
    if my_pickup:
      xidx = mdp_agent.latent_space.state_to_idx[("goal", 0)]
      np_Tx[xidx] = 1
      return np_Tx
    elif my_drop:
      if num_valid_box > 0:
        for idx in valid_boxes:
          xidx = mdp_agent.latent_space.state_to_idx[("pickup", idx)]
          np_Tx[xidx] = 1 / num_valid_box
        return np_Tx
      else:
        if mate_box_nxt > 0:
          xidx = mdp_agent.latent_space.state_to_idx[("pickup", mate_box_nxt)]
          np_Tx[xidx] = 1
          return np_Tx
    elif mate_pickup:
      latent = mdp_agent.latent_space.idx_to_state[latent_idx]
      if my_box_nxt < 0 and latent[0] == "pickup" and latent[1] == mate_box_nxt:
        if num_valid_box > 0:
          for idx in valid_boxes:
            xidx = mdp_agent.latent_space.state_to_idx[("pickup", idx)]
            np_Tx[xidx] = 1 / num_valid_box
          return np_Tx

    np_Tx[latent_idx] = 1
    return np_Tx

  if agent_idx == 0:
    return get_np_Tx_indv_impl(a1_box_cur, a2_box_cur, a1_box_nxt, a2_box_nxt)
  else:
    return get_np_Tx_indv_impl(a2_box_cur, a1_box_cur, a2_box_nxt, a1_box_nxt)


def get_np_Tx_team(mdp_agent: BoxPushTeamMDP_AlwaysTogether, agent_idx,
                   latent_idx, state_idx, tuple_action_idx, next_state_idx):
  '''
  state_idx: absolute (task-perspective) state representation.
            Here, we assume agent1 state space and task state space is the same
  '''
  num_drops = len(mdp_agent.drops)
  num_goals = len(mdp_agent.goals)

  _, _, box_states_cur = mdp_agent.conv_mdp_sidx_to_sim_states(state_idx)
  a1_pos, a2_pos, box_states_nxt = mdp_agent.conv_mdp_sidx_to_sim_states(
      next_state_idx)

  a1_box_cur, a2_box_cur, _ = get_holding_box_and_floor_boxes(
      box_states_cur, num_drops, num_goals)
  a1_box_nxt, a2_box_nxt, valid_boxes = get_holding_box_and_floor_boxes(
      box_states_nxt, num_drops, num_goals)

  def get_np_Tx_team_impl(my_box_cur, mate_box_cur, my_box_nxt, mate_box_nxt,
                          mate_pos):
    my_pickup = my_box_cur < 0 and my_box_nxt >= 0
    my_drop = my_box_cur >= 0 and my_box_nxt < 0

    num_valid_box = len(valid_boxes)
    np_Tx = np.zeros(mdp_agent.num_latents)
    if my_pickup:
      xidx = mdp_agent.latent_space.state_to_idx[("goal", 0)]
      np_Tx[xidx] = 1
      return np_Tx
    elif my_drop:
      if num_valid_box > 0:
        for idx in valid_boxes:
          xidx = mdp_agent.latent_space.state_to_idx[("pickup", idx)]
          np_Tx[xidx] = 1 / num_valid_box
        return np_Tx
    elif my_box_nxt < 0:
      latent = mdp_agent.latent_space.idx_to_state[latent_idx]
      # change latent based on teammate position
      if latent[0] == "pickup":
        min_idx = None
        dist_min = mdp_agent.x_grid * mdp_agent.y_grid
        for idx, bidx in enumerate(box_states_nxt):
          bstate = conv_box_idx_2_state(bidx, num_drops, num_goals)
          if bstate[0] == BoxState.Original:
            box_pos = mdp_agent.boxes[idx]
            dist_tmp = abs(mate_pos[0] - box_pos[0]) + abs(mate_pos[1] -
                                                           box_pos[1])
            if dist_min > dist_tmp:
              dist_min = dist_tmp
              min_idx = idx
        if min_idx is not None and dist_min < 2 and latent[1] != min_idx:
          P_CHANGE = 0.1
          xidx = mdp_agent.latent_space.state_to_idx[("pickup", min_idx)]
          np_Tx[xidx] = P_CHANGE
          np_Tx[latent_idx] = 1 - P_CHANGE
          return np_Tx

    np_Tx[latent_idx] = 1
    return np_Tx

  if agent_idx == 0:
    return get_np_Tx_team_impl(a1_box_cur, a2_box_cur, a1_box_nxt, a2_box_nxt,
                               a2_pos)
  else:
    return get_np_Tx_team_impl(a2_box_cur, a1_box_cur, a2_box_nxt, a1_box_nxt,
                               a1_pos)


def get_np_bx_temp(mdp_agent: BoxPushAgentMDP_AlwaysAlone, agent_idx,
                   state_idx):
  '''
  assume agent1 and 2 has the same policy
  state_idx: absolute (task-perspective) state representation.
            For here, we assume agent1 state and task state is the same
  '''

  _, _, box_states = mdp_agent.conv_mdp_sidx_to_sim_states(state_idx)

  num_drops = len(mdp_agent.drops)
  num_goals = len(mdp_agent.goals)
  a1_box, a2_box, valid_box = get_holding_box_and_floor_boxes(
      box_states, num_drops, num_goals)

  def get_np_bx_from_hold_state(my_box, mate_box):
    P_ORIG = 0.1
    P_DROP = 0
    P_GOAL = 1 - P_ORIG - P_DROP

    np_bx = np.zeros(len(mdp_agent.boxes) + 1 + num_drops + num_goals)
    if my_box >= 0:
      xidx = mdp_agent.latent_space.state_to_idx[("origin", 0)]
      np_bx[xidx] = P_ORIG
      for idx in range(num_drops):
        xidx = mdp_agent.latent_space.state_to_idx[("drop", idx)]
        np_bx[xidx] = P_DROP / num_drops
      for idx in range(num_goals):
        xidx = mdp_agent.latent_space.state_to_idx[("goal", idx)]
        np_bx[xidx] = P_GOAL / num_goals
    else:
      num_valid_box = len(valid_box)
      if num_valid_box > 0:
        for idx in valid_box:
          xidx = mdp_agent.latent_space.state_to_idx[("pickup", idx)]
          np_bx[xidx] = 1 / num_valid_box
      else:
        if mate_box > 0:
          xidx = mdp_agent.latent_space.state_to_idx[("pickup", mate_box)]
          np_bx[xidx] = 1
        else:  # game finished, not meaningful state
          xidx = mdp_agent.latent_space.state_to_idx[("goal", 0)]
          np_bx[xidx] = 1

    return np_bx

  if agent_idx == 0:
    return get_np_bx_from_hold_state(a1_box, a2_box)
  else:
    return get_np_bx_from_hold_state(a2_box, a1_box)


def get_indv_np_bx(mdp_agent: BoxPushAgentMDP_AlwaysAlone, agent_idx,
                   state_idx):
  ' state_idx: absolute (task-perspective) state representation.'
  return get_np_bx_temp(mdp_agent, agent_idx, state_idx)


def get_team_np_bx(mdp_agent: BoxPushTeamMDP_AlwaysTogether, agent_idx,
                   state_idx):
  ' state_idx: absolute (task-perspective) state representation.'
  return get_np_bx_temp(mdp_agent, agent_idx, state_idx)


def get_init_x(mdp_agent: Union[BoxPushAgentMDP_AlwaysAlone,
                                BoxPushTeamMDP_AlwaysTogether], box_states,
               a1_pos, a2_pos, is_team):
  A1 = 0
  A2 = 1
  sidx = mdp_agent.conv_sim_states_to_mdp_sidx(a1_pos, a2_pos, box_states)
  a1_np_bx = (get_team_np_bx(mdp_agent, A1, sidx)
              if is_team else get_indv_np_bx(mdp_agent, A1, sidx))
  a2_np_bx = (get_team_np_bx(mdp_agent, A2, sidx)
              if is_team else get_indv_np_bx(mdp_agent, A2, sidx))

  xidx1 = np.random.choice(range(mdp_agent.num_latents),
                           size=1,
                           replace=False,
                           p=a1_np_bx)[0]
  xidx2 = np.random.choice(range(mdp_agent.num_latents),
                           size=1,
                           replace=False,
                           p=a2_np_bx)[0]

  return (mdp_agent.latent_space.idx_to_state[xidx1],
          mdp_agent.latent_space.idx_to_state[xidx2])


def get_Tx_indv(mdp_agent: BoxPushAgentMDP_AlwaysAlone, agent_idx, latent,
                cur_state, a1_action, a2_action, next_state):
  'assume a1 and a2 have the same MDP'
  box_state_cur, a1_pos_cur, a2_pos_cur = cur_state
  box_state_nxt, a1_pos_nxt, a2_pos_nxt = next_state
  sidx_cur = mdp_agent.conv_sim_states_to_mdp_sidx(a1_pos_cur, a2_pos_cur,
                                                   box_state_cur)
  sidx_nxt = mdp_agent.conv_sim_states_to_mdp_sidx(a1_pos_nxt, a2_pos_nxt,
                                                   box_state_nxt)

  xidx = mdp_agent.latent_space.state_to_idx[latent]
  tuple_aidx = (mdp_agent.my_act_space.action_to_idx[a1_action],
                mdp_agent.my_act_space.action_to_idx[a2_action])

  np_Tx = get_np_Tx_indv(mdp_agent, agent_idx, xidx, sidx_cur, tuple_aidx,
                         sidx_nxt)

  xidx_nxt = np.random.choice(range(mdp_agent.num_latents),
                              size=1,
                              replace=False,
                              p=np_Tx)[0]
  return mdp_agent.latent_space.idx_to_state[xidx_nxt]


def get_Tx_team(mdp_agent: BoxPushTeamMDP_AlwaysTogether, agent_idx, latent,
                cur_state, a1_action, a2_action, next_state):
  'assume a1 and a2 have the same MDP'
  box_state_cur, a1_pos_cur, a2_pos_cur = cur_state
  box_state_nxt, a1_pos_nxt, a2_pos_nxt = next_state
  sidx_cur = mdp_agent.conv_sim_states_to_mdp_sidx(a1_pos_cur, a2_pos_cur,
                                                   box_state_cur)
  sidx_nxt = mdp_agent.conv_sim_states_to_mdp_sidx(a1_pos_nxt, a2_pos_nxt,
                                                   box_state_nxt)

  xidx = mdp_agent.latent_space.state_to_idx[latent]
  tuple_aidx = (mdp_agent.a1_a_space.action_to_idx[a1_action],
                mdp_agent.a2_a_space.action_to_idx[a2_action])

  np_Tx = get_np_Tx_team(mdp_agent, agent_idx, xidx, sidx_cur, tuple_aidx,
                         sidx_nxt)
  xidx_nxt = np.random.choice(range(mdp_agent.num_latents),
                              size=1,
                              replace=False,
                              p=np_Tx)[0]
  return mdp_agent.latent_space.idx_to_state[xidx_nxt]


if __name__ == "__main__":
  import ai_coach_domain.box_push.maps as bp_maps
  import ai_coach_domain.box_push.mdp as bp_mdp

  IS_TEAM = True
  IS_TEST = False

  if IS_TEAM:
    BoxPushAgentMDP = bp_mdp.BoxPushTeamMDP_AlwaysTogether
  else:
    BoxPushAgentMDP = bp_mdp.BoxPushAgentMDP_AlwaysAlone

  if IS_TEST:
    GAME_MAP = bp_maps.TEST_MAP
  else:
    GAME_MAP = bp_maps.EXP1_MAP

  MDP_AGENT = BoxPushAgentMDP(**GAME_MAP)  # MDP for agent policy

  for a_i in [0, 1]:
    for xidx in range(MDP_AGENT.num_latents):
      print(xidx)
      for sidx in range(MDP_AGENT.num_states):
        for aidx1 in range(MDP_AGENT.a1_a_space.num_actions):
          for aidx2 in range(MDP_AGENT.a2_a_space.num_actions):
            for sidx_n in range(MDP_AGENT.num_states):
              np_Tx = get_np_Tx_team(MDP_AGENT, a_i, xidx, sidx, (aidx1, aidx2),
                                     sidx_n)
              assert (np.sum(np_Tx) == 1)
