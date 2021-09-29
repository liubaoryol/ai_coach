import random
from ai_coach_domain.box_push import conv_box_idx_2_state, BoxState


def are_agent_states_changed(cur_state, next_state, num_drops, num_goals):
  box_states_prev = cur_state[0]
  a1_box_prev = -1
  a2_box_prev = -1
  for idx in range(len(box_states_prev)):
    state = conv_box_idx_2_state(box_states_prev[idx], num_drops, num_goals)
    if state[0] == BoxState.WithAgent1:  # with a1
      a1_box_prev = idx
    elif state[0] == BoxState.WithAgent2:  # with a2
      a2_box_prev = idx
    elif state[0] == BoxState.WithBoth:  # with both
      a1_box_prev = idx
      a2_box_prev = idx

  box_states = next_state[0]
  a1_box = -1
  a2_box = -1
  for idx in range(len(box_states)):
    state = conv_box_idx_2_state(box_states[idx], num_drops, num_goals)
    if state[0] == BoxState.WithAgent1:  # with a1
      a1_box = idx
    elif state[0] == BoxState.WithAgent2:  # with a2
      a2_box = idx
    elif state[0] == BoxState.WithBoth:  # with both
      a1_box = idx
      a2_box = idx

  a1_hold_changed = False
  a2_hold_changed = False

  if a1_box_prev != a1_box:
    a1_hold_changed = True

  if a2_box_prev != a2_box:
    a2_hold_changed = True

  return a1_hold_changed, a2_hold_changed, a1_box, a2_box


def get_valid_box_to_pickup(box_states, num_drops, num_goals):
  valid_box = []

  for idx in range(len(box_states)):
    state = conv_box_idx_2_state(box_states[idx], num_drops, num_goals)
    if state[0] in [BoxState.Original, BoxState.OnDropLoc]:  # with a1
      valid_box.append(idx)

  return valid_box


def change_latent_based_on_teammate(latent, box_states, teammate_pos, boxes,
                                    num_drops, num_goals):
  if latent[0] != "pickup":
    return latent

  closest_idx = None
  dist = 100000

  for idx, bidx in enumerate(box_states):
    bstate = conv_box_idx_2_state(bidx, num_drops, num_goals)
    if bstate[0] == BoxState.Original:
      box_pos = boxes[idx]
      dist_cur = abs(teammate_pos[0] - box_pos[0]) + abs(teammate_pos[1] -
                                                         box_pos[1])
      if dist > dist_cur:
        dist = dist_cur
        closest_idx = idx

  if closest_idx is not None and dist < 2 and latent[1] != closest_idx:
    prop = 0.1
    if prop > random.uniform(0, 1):
      return ("pickup", closest_idx)
  return latent


def get_a1_latent_indv(cur_state, a1_action, a2_action, a1_latent, next_state,
                       num_drops, num_goals):
  bstate_nxt, _, _ = next_state

  a1_hold_changed, a2_hold_changed, a1_box, a2_box = are_agent_states_changed(
      cur_state, next_state, num_drops, num_goals)

  a1_pickup = a1_hold_changed and (a1_box >= 0)
  a1_drop = a1_hold_changed and not (a1_box >= 0)
  a2_pickup = a2_hold_changed and (a2_box >= 0)
  # a2_drop = a2_hold_changed and not (a2_box >= 0)

  if a1_pickup:
    return ("goal", 0)

  elif a1_drop:
    valid_boxes = get_valid_box_to_pickup(bstate_nxt, num_drops, num_goals)
    if len(valid_boxes) > 0:
      box_idx = random.choice(valid_boxes)
      return ("pickup", box_idx)
    else:
      return ("pickup", a2_box)

  elif a2_pickup:
    if a1_box < 0 and a2_box == a1_latent[1]:
      valid_boxes = get_valid_box_to_pickup(bstate_nxt, num_drops, num_goals)
      if len(valid_boxes) > 0:
        box_idx = random.choice(valid_boxes)
        return ("pickup", box_idx)
      else:
        return ("pickup", a2_box)

  return a1_latent


def get_a2_latent_indv(cur_state, a1_action, a2_action, a2_latent, next_state,
                       num_drops, num_goals):
  bstate_nxt, _, _ = next_state

  a1_hold_changed, a2_hold_changed, a1_box, a2_box = are_agent_states_changed(
      cur_state, next_state, num_drops, num_goals)

  a1_pickup = a1_hold_changed and (a1_box >= 0)
  # a1_drop = a1_hold_changed and not (a1_box >= 0)
  a2_pickup = a2_hold_changed and (a2_box >= 0)
  a2_drop = a2_hold_changed and not (a2_box >= 0)

  if a2_pickup:
    return ("goal", 0)

  elif a2_drop:
    valid_boxes = get_valid_box_to_pickup(bstate_nxt, num_drops, num_goals)
    if len(valid_boxes) > 0:
      box_idx = random.choice(valid_boxes)
      return ("pickup", box_idx)
    else:
      return ("pickup", a1_box)

  elif a1_pickup:
    if a2_box < 0 and a1_box == a2_latent[1]:
      valid_boxes = get_valid_box_to_pickup(bstate_nxt, num_drops, num_goals)
      if len(valid_boxes) > 0:
        box_idx = random.choice(valid_boxes)
        return ("pickup", box_idx)
      else:
        return ("pickup", a1_box)

  return a2_latent


def get_a1_latent_team(cur_state, a1_action, a2_action, a1_latent, next_state,
                       boxes, num_drops, num_goals):
  bstate_nxt, a1_pos, a2_pos = next_state

  a1_hold_changed, a2_hold_changed, a1_box, a2_box = are_agent_states_changed(
      cur_state, next_state, num_drops, num_goals)
  if a1_hold_changed:
    if a1_box >= 0:
      return ("goal", 0)
    else:
      valid_boxes = get_valid_box_to_pickup(bstate_nxt, num_drops, num_goals)
      if len(valid_boxes) > 0:
        box_idx = random.choice(valid_boxes)
        return ("pickup", box_idx)
  else:
    if a1_box < 0:
      return change_latent_based_on_teammate(a1_latent, bstate_nxt, a2_pos,
                                             boxes, num_drops, num_goals)
  return a1_latent


def get_a2_latent_team(cur_state, a1_action, a2_action, a2_latent, next_state,
                       boxes, num_drops, num_goals):
  bstate_nxt, a1_pos, a2_pos = next_state

  a1_hold_changed, a2_hold_changed, a1_box, a2_box = are_agent_states_changed(
      cur_state, next_state, num_drops, num_goals)
  if a2_hold_changed:
    if a2_box >= 0:
      return ("goal", 0)
    else:
      valid_boxes = get_valid_box_to_pickup(bstate_nxt, num_drops, num_goals)
      if len(valid_boxes) > 0:
        box_idx = random.choice(valid_boxes)
        return ("pickup", box_idx)
  else:
    if a2_box < 0:
      return change_latent_based_on_teammate(a2_latent, bstate_nxt, a1_pos,
                                             boxes, num_drops, num_goals)
  return a2_latent
