from typing import Union
import numpy as np
from ai_coach_domain.box_push import conv_box_idx_2_state, BoxState
from ai_coach_domain.box_push.mdp import (BoxPushAgentMDP_AlwaysAlone,
                                          BoxPushTeamMDP_AlwaysTogether)

# def are_agent_states_changed(box_states, box_states_next, num_drops,
#  num_goals):
#   a1_box_prev = -1
#   a2_box_prev = -1
#   for idx in range(len(box_states)):
#     state = conv_box_idx_2_state(box_states[idx], num_drops, num_goals)
#     if state[0] == BoxState.WithAgent1:  # with a1
#       a1_box_prev = idx
#     elif state[0] == BoxState.WithAgent2:  # with a2
#       a2_box_prev = idx
#     elif state[0] == BoxState.WithBoth:  # with both
#       a1_box_prev = idx
#       a2_box_prev = idx

#   a1_box = -1
#   a2_box = -1
#   for idx in range(len(box_states_next)):
#     state = conv_box_idx_2_state(box_states_next[idx], num_drops, num_goals)
#     if state[0] == BoxState.WithAgent1:  # with a1
#       a1_box = idx
#     elif state[0] == BoxState.WithAgent2:  # with a2
#       a2_box = idx
#     elif state[0] == BoxState.WithBoth:  # with both
#       a1_box = idx
#       a2_box = idx

#   a1_hold_changed = False
#   a2_hold_changed = False

#   if a1_box_prev != a1_box:
#     a1_hold_changed = True

#   if a2_box_prev != a2_box:
#     a2_hold_changed = True

#   return a1_hold_changed, a2_hold_changed, a1_box, a2_box

# def get_valid_box_to_pickup(box_states, num_drops, num_goals):
#   valid_box = []

#   for idx in range(len(box_states)):
#     state = conv_box_idx_2_state(box_states[idx], num_drops, num_goals)
#     if state[0] in [BoxState.Original, BoxState.OnDropLoc]:  # with a1
#       valid_box.append(idx)

#   return valid_box

# def change_latent_based_on_teammate(latent, box_states, teammate_pos, boxes,
#                                     num_drops, num_goals):
#   if latent[0] != "pickup":
#     return latent

#   closest_idx = None
#   dist = 100000

#   for idx, bidx in enumerate(box_states):
#     bstate = conv_box_idx_2_state(bidx, num_drops, num_goals)
#     if bstate[0] == BoxState.Original:
#       box_pos = boxes[idx]
#       dist_cur = abs(teammate_pos[0] - box_pos[0]) + abs(teammate_pos[1] -
#                                                          box_pos[1])
#       if dist > dist_cur:
#         dist = dist_cur
#         closest_idx = idx

#   if closest_idx is not None and dist < 2 and latent[1] != closest_idx:
#     prop = 0.1
#     if prop > random.uniform(0, 1):
#       return ("pickup", closest_idx)
#   return latent

# def get_a1_latent_indv(cur_state, a1_action, a2_action, a1_latent, next_state,
#                        boxes, drops, goals):
#   num_drops = len(drops)
#   num_goals = len(goals)

#   bstate_nxt, _, _ = next_state

#   a1_hold_changed, a2_hold_changed, a1_box, a2_box = are_agent_states_changed(
#       cur_state, next_state, num_drops, num_goals)

#   a1_pickup = a1_hold_changed and (a1_box >= 0)
#   a1_drop = a1_hold_changed and not (a1_box >= 0)
#   a2_pickup = a2_hold_changed and (a2_box >= 0)
#   # a2_drop = a2_hold_changed and not (a2_box >= 0)

#   if a1_pickup:
#     return ("goal", 0)

#   elif a1_drop:
#     valid_boxes = get_valid_box_to_pickup(bstate_nxt, num_drops, num_goals)
#     if len(valid_boxes) > 0:
#       box_idx = random.choice(valid_boxes)
#       return ("pickup", box_idx)
#     else:
#       return ("pickup", a2_box)

#   elif a2_pickup:
#     if a1_box < 0 and a2_box == a1_latent[1]:
#       valid_boxes = get_valid_box_to_pickup(bstate_nxt, num_drops, num_goals)
#       if len(valid_boxes) > 0:
#         box_idx = random.choice(valid_boxes)
#         return ("pickup", box_idx)
#       else:
#         return ("pickup", a2_box)

#   return a1_latent

# def get_a2_latent_indv(cur_state, a1_action, a2_action, a2_latent, next_state,
#                        boxes, drops, goals):
#   num_drops = len(drops)
#   num_goals = len(goals)

#   bstate_nxt, _, _ = next_state

#   a1_hold_changed, a2_hold_changed, a1_box, a2_box = are_agent_states_changed(
#       cur_state, next_state, num_drops, num_goals)

#   a1_pickup = a1_hold_changed and (a1_box >= 0)
#   # a1_drop = a1_hold_changed and not (a1_box >= 0)
#   a2_pickup = a2_hold_changed and (a2_box >= 0)
#   a2_drop = a2_hold_changed and not (a2_box >= 0)

#   if a2_pickup:
#     return ("goal", 0)

#   elif a2_drop:
#     valid_boxes = get_valid_box_to_pickup(bstate_nxt, num_drops, num_goals)
#     if len(valid_boxes) > 0:
#       box_idx = random.choice(valid_boxes)
#       return ("pickup", box_idx)
#     else:
#       return ("pickup", a1_box)

#   elif a1_pickup:
#     if a2_box < 0 and a1_box == a2_latent[1]:
#       valid_boxes = get_valid_box_to_pickup(bstate_nxt, num_drops, num_goals)
#       if len(valid_boxes) > 0:
#         box_idx = random.choice(valid_boxes)
#         return ("pickup", box_idx)
#       else:
#         return ("pickup", a1_box)

#   return a2_latent

# def get_a1_latent_team(cur_state, a1_action, a2_action, a1_latent, next_state,
#                        boxes, drops, goals):
#   num_drops = len(drops)
#   num_goals = len(goals)

#   bstate_nxt, a1_pos, a2_pos = next_state

#   a1_hold_changed, a2_hold_changed, a1_box, a2_box = are_agent_states_changed(
#       cur_state, next_state, num_drops, num_goals)
#   if a1_hold_changed:
#     if a1_box >= 0:
#       return ("goal", 0)
#     else:
#       valid_boxes = get_valid_box_to_pickup(bstate_nxt, num_drops, num_goals)
#       if len(valid_boxes) > 0:
#         box_idx = random.choice(valid_boxes)
#         return ("pickup", box_idx)
#   else:
#     if a1_box < 0:
#       return change_latent_based_on_teammate(a1_latent, bstate_nxt, a2_pos,
#                                              boxes, num_drops, num_goals)
#   return a1_latent

# def get_a2_latent_team(cur_state, a1_action, a2_action, a2_latent, next_state,
#                        boxes, drops, goals):
#   num_drops = len(drops)
#   num_goals = len(goals)

#   bstate_nxt, a1_pos, a2_pos = next_state

#   a1_hold_changed, a2_hold_changed, a1_box, a2_box = are_agent_states_changed(
#       cur_state, next_state, num_drops, num_goals)
#   if a2_hold_changed:
#     if a2_box >= 0:
#       return ("goal", 0)
#     else:
#       valid_boxes = get_valid_box_to_pickup(bstate_nxt, num_drops, num_goals)
#       if len(valid_boxes) > 0:
#         box_idx = random.choice(valid_boxes)
#         return ("pickup", box_idx)
#   else:
#     if a2_box < 0:
#       return change_latent_based_on_teammate(a2_latent, bstate_nxt, a1_pos,
#                                              boxes, num_drops, num_goals)
#   return a2_latent


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
