from typing import Tuple
from enum import Enum


class EventType(Enum):
  UP = 0
  DOWN = 1
  LEFT = 2
  RIGHT = 3
  STAY = 4
  HOLD = 5
  UNHOLD = 5
  SET_LATENT = 6


class BoxState(Enum):
  Original = 0
  WithAgent1 = 1
  WithAgent2 = 2
  WithBoth = 3
  OnDropLoc = 4
  OnGoalLoc = 5


def conv_box_idx_2_state(state_idx, num_drops, num_goals):
  if state_idx >= 0 and state_idx < 4:
    return (BoxState(state_idx), None)
  elif state_idx >= 4 and state_idx < 4 + num_drops:
    return (BoxState(4), state_idx - 4)
  elif state_idx >= 4 + num_drops and state_idx < 4 + num_drops + num_goals:
    return (BoxState(5), state_idx - 4 - num_drops)

  raise ValueError


def conv_box_state_2_idx(state: Tuple[BoxState, int], num_drops):
  if state[0].value < 4:
    return state[0].value
  elif state[0] == BoxState.OnDropLoc:
    return 4 + state[1]
  elif state[0] == BoxState.OnGoalLoc:
    return 4 + num_drops + state[1]

  raise ValueError


def is_opposite_direction(dir1, dir2):
  return ((dir1 == EventType.UP and dir2 == EventType.DOWN)
          or (dir1 == EventType.DOWN and dir2 == EventType.UP)
          or (dir1 == EventType.LEFT and dir2 == EventType.RIGHT)
          or (dir1 == EventType.RIGHT and dir2 == EventType.LEFT))


def transition(box_states: list, a1_pos, a2_pos, a1_act, a2_act,
               box_locations: list, goals: list, walls: list, drops: list,
               x_bound, y_bound):
  num_goals = len(goals)
  num_drops = len(drops)

  # methods
  def get_box_idx(coord):
    for idx, sidx in enumerate(box_states):
      state = conv_box_idx_2_state(sidx, num_drops, num_goals)
      if (state[0] == BoxState.Original) and (coord == box_locations[idx]):
        return idx
      elif ((state[0] in [BoxState.WithAgent1, BoxState.WithBoth])
            and (coord == a1_pos)):
        return idx
      elif (state[0] == BoxState.WithAgent2) and (coord == a2_pos):
        return idx
      elif (state[0] == BoxState.OnDropLoc) and (coord == drops[state[1]]):
        return idx
      # elif (state[0] == BoxState.OnGoalLoc) and (coord == goals[state[1]]):
      #   return idx

    return -1

  def is_wall(coord):
    x, y = coord
    if x < 0 or x >= x_bound or y < 0 or y >= y_bound:
      return True

    if coord in walls:
      return True

    return False

  def get_moved_coord(coord, action, check_box=False):
    x, y = coord
    coord_new = None
    if action == EventType.UP:
      coord_new = (x, y - 1)
    elif action == EventType.DOWN:
      coord_new = (x, y + 1)
    elif action == EventType.LEFT:
      coord_new = (x - 1, y)
    elif action == EventType.RIGHT:
      coord_new = (x + 1, y)
    else:
      return coord

    if is_wall(coord_new):
      return coord
      # coord_new = coord

    if check_box and get_box_idx(coord_new) >= 0:
      return coord
      # coord_new = coord

    return coord_new

  def hold_state():
    a1_hold = False
    a2_hold = False
    both_hold = False
    for sidx in box_states:
      state = conv_box_idx_2_state(sidx, num_drops, num_goals)
      if state[0] == BoxState.WithAgent1:
        a1_hold = True
      elif state[0] == BoxState.WithAgent2:
        a2_hold = True
      elif state[0] == BoxState.WithBoth:
        both_hold = True

    if both_hold:
      return "Both"
    elif a1_hold and a2_hold:
      return "Each"
    elif a1_hold:
      return "A1"
    elif a2_hold:
      return "A2"
    else:
      return "None"

  def update_dropped_box_state(boxidx, coord, box_states_new):
    # if the box is at original location, no problem to drop
    if coord == box_locations[boxidx]:
      # boxes_states_new = list(box_states)
      box_states_new[boxidx] = conv_box_state_2_idx((BoxState.Original, None),
                                                    num_drops)
      return True
      # list_next_env.append((1.0, boxes_states_new, a1_pos, a2_pos))
    # if the box is at one of drop locations, no problem
    elif coord in drops:
      # boxes_states_new = list(box_states)
      box_states_new[boxidx] = conv_box_state_2_idx(
          (BoxState.OnDropLoc, drops.index(coord)), num_drops)
      return True
      # list_next_env.append((1.0, boxes_states_new, a1_pos, a2_pos))
    # if the box is at one of goal locations, no problem
    elif coord in goals:
      # boxes_states_new = list(box_states)
      box_states_new[boxidx] = conv_box_state_2_idx(
          (BoxState.OnGoalLoc, goals.index(coord)), num_drops)
      return True
    else:
      return False
      # list_next_env.append((1.0, boxes_states_new, a1_pos, a2_pos))
    # if the box is where dropping is not allowed, only one can unhold it

  P_MOVE = 0.5
  list_next_env = []
  hold = hold_state()
  # both do not hold anything
  if hold == "None":
    box_states_new = list(box_states)
    if (a1_act == EventType.HOLD and a2_act == EventType.HOLD
        and a1_pos == a2_pos):
      bidx = get_box_idx(a1_pos)
      if bidx >= 0:
        state = (BoxState.WithBoth, None)
        box_states_new[bidx] = conv_box_state_2_idx(state, num_drops)

      list_next_env.append((1.0, box_states_new, a1_pos, a2_pos))
    else:
      a1_pos_new = a1_pos
      if a1_act == EventType.HOLD:
        bidx = get_box_idx(a1_pos)
        if bidx >= 0:
          state = (BoxState.WithAgent1, None)
          box_states_new[bidx] = conv_box_state_2_idx(state, num_drops)
      else:
        a1_pos_new = get_moved_coord(a1_pos, a1_act)

      a2_pos_new = a2_pos
      if a2_act == EventType.HOLD:
        bidx = get_box_idx(a2_pos)
        if bidx >= 0:
          state = (BoxState.WithAgent2, None)
          box_states_new[bidx] = conv_box_state_2_idx(state, num_drops)
      else:
        a2_pos_new = get_moved_coord(a2_pos, a2_act)

      list_next_env.append((1.0, box_states_new, a1_pos_new, a2_pos_new))
  # both hold the same box
  elif hold == "Both":
    bidx = get_box_idx(a1_pos)
    assert bidx >= 0
    assert a1_pos == a2_pos

    box_states_new = list(box_states)
    # both try to drop the box
    if a1_act == EventType.UNHOLD and a2_act == EventType.UNHOLD:
      # if succeeded to drop
      if update_dropped_box_state(bidx, a1_pos, box_states_new):
        list_next_env.append((1.0, box_states_new, a1_pos, a2_pos))
      # if failed to drop due to invalid location
      else:
        box_states_new[bidx] = conv_box_state_2_idx((BoxState.WithAgent1, None),
                                                    num_drops)
        box_states_new2 = list(box_states)
        box_states_new2[bidx] = conv_box_state_2_idx(
            (BoxState.WithAgent2, None), num_drops)

        list_next_env.append((0.5, box_states_new, a1_pos, a2_pos))
        list_next_env.append((0.5, box_states_new2, a1_pos, a2_pos))
    # only agent1 try to unhold
    elif a1_act == EventType.UNHOLD:
      box_states_new[bidx] = conv_box_state_2_idx((BoxState.WithAgent2, None),
                                                  num_drops)
      a2_pos_new = get_moved_coord(a2_pos, a2_act, True)
      if a2_pos_new == a2_pos:
        list_next_env.append((1.0, box_states_new, a1_pos, a2_pos))
      else:
        list_next_env.append((P_MOVE, box_states_new, a1_pos, a2_pos_new))
        list_next_env.append((1.0 - P_MOVE, box_states_new, a1_pos, a2_pos))
    elif a2_act == EventType.UNHOLD:
      box_states_new[bidx] = conv_box_state_2_idx((BoxState.WithAgent1, None),
                                                  num_drops)
      a1_pos_new = get_moved_coord(a1_pos, a1_act, True)
      if a1_pos_new == a1_pos:
        list_next_env.append((1.0, box_states_new, a1_pos, a2_pos))
      else:
        list_next_env.append((P_MOVE, box_states_new, a1_pos_new, a2_pos))
        list_next_env.append((1.0 - P_MOVE, box_states_new, a1_pos, a2_pos))
    else:
      if (is_opposite_direction(a1_act, a2_act)
          or (a1_act == EventType.STAY and a2_act == EventType.STAY)):
        list_next_env.append((1.0, box_states, a1_pos, a2_pos))
      elif a1_act == a2_act:
        a1_pos_new = get_moved_coord(a1_pos, a1_act, True)
        list_next_env.append((1.0, box_states, a1_pos_new, a1_pos_new))
      else:
        a1_pos_new = get_moved_coord(a1_pos, a1_act, True)
        a2_pos_new = get_moved_coord(a2_pos, a2_act, True)
        if a1_pos_new == a2_pos_new:
          list_next_env.append((1.0, box_states, a1_pos_new, a1_pos_new))
        else:
          list_next_env.append((0.5, box_states, a1_pos_new, a1_pos_new))
          list_next_env.append((0.5, box_states, a2_pos_new, a2_pos_new))
  elif hold == "Each":
    box_states_new = list(box_states)
    a1_pos_new = a1_pos
    a2_pos_new = a2_pos
    # if more than one remains on the same grid
    if (a1_act in [EventType.UNHOLD, EventType.STAY]
        or a2_act in [EventType.UNHOLD, EventType.STAY]):
      p_a1_success = 0
      if a1_act == EventType.UNHOLD:
        bidx = get_box_idx(a1_pos)
        assert bidx >= 0
        update_dropped_box_state(bidx, a1_pos, box_states_new)
        p_a1_success = 1
      else:
        a1_pos_new = get_moved_coord(a1_pos, a1_act, True)
        if a1_pos_new != a1_pos:
          p_a1_success = P_MOVE

      p_a2_success = 0
      if a2_act == EventType.UNHOLD:
        bidx = get_box_idx(a2_pos)
        assert bidx >= 0
        update_dropped_box_state(bidx, a2_pos, box_states_new)
        p_a2_success = 1
      else:
        a2_pos_new = get_moved_coord(a2_pos, a2_act, True)
        if a2_pos_new != a2_pos:
          p_a2_success = P_MOVE

      p_ss = p_a1_success * p_a2_success
      p_sf = p_a1_success * (1 - p_a2_success)
      p_fs = (1 - p_a1_success) * p_a2_success
      p_ff = (1 - p_a1_success) * (1 - p_a2_success)
      if p_ss > 0:
        list_next_env.append((p_ss, box_states_new, a1_pos_new, a2_pos_new))
      if p_sf > 0:
        list_next_env.append((p_sf, box_states_new, a1_pos_new, a2_pos))
      if p_fs > 0:
        list_next_env.append((p_fs, box_states_new, a1_pos, a2_pos_new))
      if p_ff > 0:
        list_next_env.append((p_ff, box_states_new, a1_pos, a2_pos))
    # when both try to move
    else:
      agent_dist = (abs(a1_pos[0] - a2_pos[0]) + abs(a1_pos[1] - a2_pos[1]))
      if agent_dist > 2:
        p_a1_success = 0
        a1_pos_new = get_moved_coord(a1_pos, a1_act, True)
        if a1_pos_new != a1_pos:
          p_a1_success = P_MOVE

        p_a2_success = 0
        a2_pos_new = get_moved_coord(a2_pos, a2_act, True)
        if a2_pos_new != a2_pos:
          p_a2_success = P_MOVE

        p_ss = p_a1_success * p_a2_success
        p_sf = p_a1_success * (1 - p_a2_success)
        p_fs = (1 - p_a1_success) * p_a2_success
        p_ff = (1 - p_a1_success) * (1 - p_a2_success)
        if p_ss > 0:
          list_next_env.append((p_ss, box_states, a1_pos_new, a2_pos_new))
        if p_sf > 0:
          list_next_env.append((p_sf, box_states, a1_pos_new, a2_pos))
        if p_fs > 0:
          list_next_env.append((p_fs, box_states, a1_pos, a2_pos_new))
        if p_ff > 0:
          list_next_env.append((p_ff, box_states, a1_pos, a2_pos))
      elif agent_dist == 2:
        p_a1_success = 0
        a1_pos_new = get_moved_coord(a1_pos, a1_act, True)
        if a1_pos_new != a1_pos:
          p_a1_success = P_MOVE

        p_a2_success = 0
        a2_pos_new = get_moved_coord(a2_pos, a2_act, True)
        if a2_pos_new != a2_pos:
          p_a2_success = P_MOVE

        p_ss = p_a1_success * p_a2_success
        p_sf = p_a1_success * (1 - p_a2_success)
        p_fs = (1 - p_a1_success) * p_a2_success
        p_ff = (1 - p_a1_success) * (1 - p_a2_success)

        if a1_pos_new == a2_pos_new:
          if p_ss > 0:
            list_next_env.append((p_ss * 0.5, box_states, a1_pos_new, a2_pos))
            list_next_env.append((p_ss * 0.5, box_states, a1_pos, a2_pos_new))
        else:
          if p_ss > 0:
            list_next_env.append((p_ss, box_states, a1_pos_new, a2_pos_new))

        if p_sf > 0:
          list_next_env.append((p_sf, box_states, a1_pos_new, a2_pos))
        if p_fs > 0:
          list_next_env.append((p_fs, box_states, a1_pos, a2_pos_new))
        if p_ff > 0:
          list_next_env.append((p_ff, box_states, a1_pos, a2_pos))
      else:  # agent_dst == 1
        a1_pos_new = get_moved_coord(a1_pos, a1_act, True)
        a2_pos_new = get_moved_coord(a2_pos, a2_act, True)
        if a1_pos_new != a1_pos and a2_pos_new != a2_pos:
          list_next_env.append(
              (P_MOVE * P_MOVE, box_states, a1_pos_new, a2_pos_new))
          list_next_env.append(
              (P_MOVE * (1 - P_MOVE), box_states, a1_pos_new, a2_pos))
          list_next_env.append(
              (P_MOVE * (1 - P_MOVE), box_states, a1_pos, a2_pos_new))
          list_next_env.append(
              ((1 - P_MOVE) * (1 - P_MOVE), box_states, a1_pos, a2_pos))
        elif a1_pos_new != a1_pos:
          a2_pos_new2 = get_moved_coord(a2_pos, a2_act)
          if a2_pos_new2 == a1_pos:
            list_next_env.append(
                (P_MOVE * P_MOVE, box_states, a1_pos_new, a1_pos))
            list_next_env.append(
                (P_MOVE * (1 - P_MOVE), box_states, a1_pos_new, a2_pos))
            list_next_env.append(((1 - P_MOVE), box_states, a1_pos, a2_pos))
          else:
            list_next_env.append((P_MOVE, box_states, a1_pos_new, a2_pos))
            list_next_env.append((1 - P_MOVE, box_states, a1_pos, a2_pos))
        elif a2_pos_new != a2_pos:
          a1_pos_new2 = get_moved_coord(a1_pos, a1_act)
          if a1_pos_new2 == a2_pos:
            list_next_env.append(
                (P_MOVE * P_MOVE, box_states, a2_pos, a2_pos_new))
            list_next_env.append(
                ((1 - P_MOVE) * P_MOVE, box_states, a1_pos, a2_pos_new))
            list_next_env.append(((1 - P_MOVE), box_states, a1_pos, a2_pos))
          else:
            list_next_env.append((P_MOVE, box_states, a1_pos, a2_pos_new))
            list_next_env.append((1 - P_MOVE, box_states, a1_pos, a2_pos))
        else:
          list_next_env.append((1.0, box_states, a1_pos, a2_pos))
  # only a1 holds a box
  elif hold == "A1":
    if a2_act == EventType.HOLD and a1_pos == a2_pos:
      box_states_new = list(box_states)
      bidx = get_box_idx(a1_pos)
      assert bidx >= 0
      if a1_act == EventType.UNHOLD:
        box_states_new[bidx] = conv_box_state_2_idx((BoxState.WithAgent2, None),
                                                    num_drops)
      else:
        box_states_new[bidx] = conv_box_state_2_idx((BoxState.WithBoth, None),
                                                    num_drops)
      list_next_env.append((1.0, box_states_new, a1_pos, a2_pos))
    else:
      box_states_new = list(box_states)
      p_a1_success = 0
      a1_pos_new = a1_pos
      if a1_act == EventType.UNHOLD:
        bidx = get_box_idx(a1_pos)
        assert bidx >= 0
        update_dropped_box_state(bidx, a1_pos, box_states_new)
        p_a1_success = 1.0
      else:
        a1_pos_new = get_moved_coord(a1_pos, a1_act, True)
        if a1_pos_new != a1_pos:
          p_a1_success = P_MOVE

      a2_pos_new = a2_pos
      if a2_act == EventType.HOLD:
        bidx = get_box_idx(a2_pos)
        if bidx >= 0:
          box_states_new[bidx] = conv_box_state_2_idx(
              (BoxState.WithAgent2, None), num_drops)
      else:
        a2_pos_new = get_moved_coord(a2_pos, a2_act)

      if p_a1_success > 0:
        list_next_env.append(
            (p_a1_success, box_states_new, a1_pos_new, a2_pos_new))
      if 1 - p_a1_success > 0:
        list_next_env.append(
            (1 - p_a1_success, box_states_new, a1_pos, a2_pos_new))
  # only a2 holds a box
  else:  # hold == "A2":
    if a1_act == EventType.HOLD and a1_pos == a2_pos:
      box_states_new = list(box_states)
      bidx = get_box_idx(a2_pos)
      assert bidx >= 0
      if a2_act == EventType.UNHOLD:
        box_states_new[bidx] = conv_box_state_2_idx((BoxState.WithAgent1, None),
                                                    num_drops)
      else:
        box_states_new[bidx] = conv_box_state_2_idx((BoxState.WithBoth, None),
                                                    num_drops)
      list_next_env.append((1.0, box_states_new, a1_pos, a2_pos))
    else:
      box_states_new = list(box_states)
      p_a2_success = 0
      a2_pos_new = a2_pos
      if a2_act == EventType.UNHOLD:
        bidx = get_box_idx(a2_pos)
        assert bidx >= 0
        update_dropped_box_state(bidx, a2_pos, box_states_new)
        p_a2_success = 1.0
      else:
        a2_pos_new = get_moved_coord(a2_pos, a2_act, True)
        if a2_pos_new != a2_pos:
          p_a2_success = P_MOVE

      a1_pos_new = a1_pos
      if a1_act == EventType.HOLD:
        bidx = get_box_idx(a1_pos)
        if bidx >= 0:
          box_states_new[bidx] = conv_box_state_2_idx(
              (BoxState.WithAgent1, None), num_drops)
      else:
        a1_pos_new = get_moved_coord(a1_pos, a1_act)

      if p_a2_success > 0:
        list_next_env.append(
            (p_a2_success, box_states_new, a1_pos_new, a2_pos_new))
      if 1 - p_a2_success > 0:
        list_next_env.append(
            (1 - p_a2_success, box_states_new, a1_pos_new, a2_pos))

  return list_next_env
