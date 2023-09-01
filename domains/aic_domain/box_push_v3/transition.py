from typing import Tuple, Sequence
from aic_domain.box_push_v2 import (BoxState, EventType, conv_box_state_2_idx,
                                    conv_box_idx_2_state)
from aic_domain.box_push.transition import (get_box_idx_impl,
                                            get_moved_coord_impl,
                                            hold_state_impl, is_wall_impl,
                                            update_dropped_box_state_impl,
                                            is_opposite_direction)

Coord = Tuple[int, int]


def possible_positions(coord, x_bound, y_bound, walls, box_states, a1_pos,
                       a2_pos, box_locations, goals, drops, holding_box):
  x, y = coord
  possible_pos = []
  possible_pos.append(coord)

  for i, j in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
    pos = (x + i, y + j)

    if not holding_box:
      if not is_wall_impl(pos, x_bound, y_bound, walls) and pos not in goals:
        possible_pos.append(pos)
    else:
      if (not is_wall_impl(pos, x_bound, y_bound, walls) and get_box_idx_impl(
          pos, box_states, a1_pos, a2_pos, box_locations, goals, drops) < 0):
        possible_pos.append(pos)

  return possible_pos


def transition_mixed_noisy(
    box_states: list, a1_pos: Coord, a2_pos: Coord, a1_act: EventType,
    a2_act: EventType, box_locations: Sequence[Coord], goals: Sequence[Coord],
    walls: Sequence[Coord], drops: Sequence[Coord], x_bound: int, y_bound: int,
    box_types: Sequence[int], a1_init: Coord, a2_init: Coord):
  # num_goals = len(goals)
  num_drops = len(drops)
  num_goals = len(goals)

  # methods
  def get_box_idx(coord):
    return get_box_idx_impl(coord, box_states, a1_pos, a2_pos, box_locations,
                            goals, drops)

  def get_moved_coord(coord, action, box_states=None, holding_box=False):
    coord_new = get_moved_coord_impl(coord, action, x_bound, y_bound, walls,
                                     box_states, a1_pos, a2_pos, box_locations,
                                     goals, drops)
    if coord_new in goals and not holding_box:
      return coord
    else:
      return coord_new

  def get_dist_new_coord(coord, action, box_states=None, holding_box=False):
    list_possible_pos = possible_positions(coord, x_bound, y_bound, walls,
                                           box_states, a1_pos, a2_pos,
                                           box_locations, goals, drops,
                                           holding_box)
    expected_pos = get_moved_coord(coord, action, box_states, holding_box)
    P_EXPECTED = 0.95
    list_dist = []
    for pos in list_possible_pos:
      if pos == expected_pos:
        list_dist.append((P_EXPECTED, pos))
      else:
        p = (1 - P_EXPECTED) / (len(list_possible_pos) - 1)
        list_dist.append((p, pos))
    return list_dist

  def hold_state():
    return hold_state_impl(box_states, drops, goals)

  def update_dropped_box_state(boxidx, coord, box_states_new):
    res = update_dropped_box_state_impl(boxidx, coord, box_states_new,
                                        box_locations, drops, goals)
    return res, conv_box_idx_2_state(box_states_new[boxidx], num_drops,
                                     num_goals)

  list_next_env = []
  hold = hold_state()
  # both do not hold anything
  if hold == "None":
    if (a1_act == EventType.HOLD and a2_act == EventType.HOLD
        and a1_pos == a2_pos):
      bidx = get_box_idx(a1_pos)
      if bidx >= 0:
        if box_types[bidx] == 2:
          state = (BoxState.WithBoth, None)
          box_states_new = list(box_states)
          box_states_new[bidx] = conv_box_state_2_idx(state, num_drops)
          list_next_env.append((1.0, box_states_new, a1_pos, a2_pos))
        elif box_types[bidx] == 1:
          state1 = (BoxState.WithAgent1, None)
          state2 = (BoxState.WithAgent2, None)
          box_states_new = list(box_states)
          box_states_new[bidx] = conv_box_state_2_idx(state1, num_drops)
          list_next_env.append((0.5, box_states_new, a1_pos, a2_pos))
          box_states_new = list(box_states)
          box_states_new[bidx] = conv_box_state_2_idx(state2, num_drops)
          list_next_env.append((0.5, box_states_new, a1_pos, a2_pos))
        else:
          raise ValueError("Box types other than 1 or 2 are not implemented")
      else:
        list_next_env.append((1.0, box_states, a1_pos, a2_pos))

    else:
      a1_pos_dist = [(1.0, a1_pos)]
      box_states_new = list(box_states)
      hold_box = False
      if a1_act == EventType.HOLD:
        bidx = get_box_idx(a1_pos)
        if bidx >= 0:
          if box_types[bidx] == 1:
            state = (BoxState.WithAgent1, None)
            box_states_new[bidx] = conv_box_state_2_idx(state, num_drops)
            hold_box = True

      if not hold_box:
        a1_pos_dist = get_dist_new_coord(a1_pos, a1_act, None, False)

      hold_box = False
      a2_pos_dist = [(1.0, a2_pos)]
      if a2_act == EventType.HOLD:
        bidx = get_box_idx(a2_pos)
        if bidx >= 0:
          if box_types[bidx] == 1:
            state = (BoxState.WithAgent2, None)
            box_states_new[bidx] = conv_box_state_2_idx(state, num_drops)
            hold_box = True

      if not hold_box:
        a2_pos_dist = get_dist_new_coord(a2_pos, a2_act, None, False)

      for p1, pos1 in a1_pos_dist:
        for p2, pos2 in a2_pos_dist:
          list_next_env.append((p1 * p2, box_states_new, pos1, pos2))

      # list_next_env.append((1.0, box_states_new, a1_pos_new, a2_pos_new))
  # both hold the same box
  elif hold == "Both":
    bidx = get_box_idx(a1_pos)
    assert bidx >= 0
    assert box_types[bidx] == 2
    # invalid case
    if a1_pos != a2_pos:
      list_next_env.append((1.0, box_states, a1_pos, a2_pos))
      return list_next_env

    # both try to drop the box
    if a1_act == EventType.UNHOLD and a2_act == EventType.UNHOLD:
      box_states_new = list(box_states)
      _, bstate = update_dropped_box_state(bidx, a1_pos, box_states_new)
      a1_pos_new = a1_pos
      a2_pos_new = a2_pos
      # respawn agents if box is dropped at the goal
      if bstate[0] == BoxState.OnGoalLoc:
        a1_pos_new = a1_init
        a2_pos_new = a2_init
      list_next_env.append((1.0, box_states_new, a1_pos_new, a2_pos_new))
    # only agent1 try to unhold
    elif a1_act == EventType.UNHOLD:
      # a2_pos_new = get_moved_coord(a2_pos, a2_act, box_states, True)
      a2_pos_dist = get_dist_new_coord(a2_pos, a2_act, box_states, True)
      stay_dist = get_dist_new_coord(a2_pos, EventType.STAY, box_states, True)
      dist_union = {elem[1]: elem[0] for elem in a2_pos_dist}
      for p, pos in stay_dist:
        dist_union[pos] += p

      for pos in dist_union:
        list_next_env.append((dist_union[pos] / 2, box_states, pos, pos))
    elif a2_act == EventType.UNHOLD:
      a1_pos_dist = get_dist_new_coord(a1_pos, a1_act, box_states, True)
      stay_dist = get_dist_new_coord(a1_pos, EventType.STAY, box_states, True)
      dist_union = {elem[1]: elem[0] for elem in a1_pos_dist}
      for p, pos in stay_dist:
        dist_union[pos] += p

      for pos in dist_union:
        list_next_env.append((dist_union[pos] / 2, box_states, pos, pos))
    else:
      if (is_opposite_direction(a1_act, a2_act)
          or (a1_act == EventType.STAY and a2_act == EventType.STAY)):
        a1_pos_dist = get_dist_new_coord(a1_pos, EventType.STAY, box_states,
                                         True)
        for p, pos in a1_pos_dist:
          list_next_env.append((p, box_states, pos, pos))
      elif a1_act == a2_act:
        a1_pos_dist = get_dist_new_coord(a1_pos, a1_act, box_states, True)
        for p, pos in a1_pos_dist:
          list_next_env.append((p, box_states, pos, pos))
      else:
        a1_pos_dist = get_dist_new_coord(a1_pos, a1_act, box_states, True)
        a2_pos_dist = get_dist_new_coord(a2_pos, a2_act, box_states, True)
        dist_union = {elem[1]: elem[0] for elem in a1_pos_dist}
        for p, pos in a2_pos_dist:
          dist_union[pos] += p

        for pos in dist_union:
          list_next_env.append((dist_union[pos] / 2, box_states, pos, pos))

  elif hold == "Each":
    agent_dist = (abs(a1_pos[0] - a2_pos[0]) + abs(a1_pos[1] - a2_pos[1]))
    if agent_dist > 2:
      box_states_new = list(box_states)

      a1_pos_dist = [(1.0, a1_pos)]
      a1_dropped = False
      bidx1 = get_box_idx(a1_pos)
      if bidx1 >= 0 and a1_act == EventType.UNHOLD:
        a1_dropped, bstate = update_dropped_box_state(bidx1, a1_pos,
                                                      box_states_new)
        if bstate[0] == BoxState.OnGoalLoc:
          a1_pos_dist = [(1.0, a1_init)]

      if not a1_dropped:
        a1_pos_dist = get_dist_new_coord(a1_pos, a1_act, box_states_new, True)

      a2_pos_dist = [(1.0, a2_pos)]
      a2_dropped = False
      bidx2 = get_box_idx(a2_pos)
      if bidx2 >= 0 and a2_act == EventType.UNHOLD:
        a2_dropped, bstate = update_dropped_box_state(bidx2, a2_pos,
                                                      box_states_new)
        if bstate[0] == BoxState.OnGoalLoc:
          a2_pos_dist = [(1.0, a2_init)]

      if not a2_dropped:
        a2_pos_dist = get_dist_new_coord(a2_pos, a2_act, box_states_new, True)

      for p1, pos1, in a1_pos_dist:
        for p2, pos2, in a2_pos_dist:
          list_next_env.append((p1 * p2, box_states_new, pos1, pos2))

    # when dist is <= 2 and more than one remains on the same grid
    elif (a1_act in [EventType.UNHOLD, EventType.STAY]
          or a2_act in [EventType.UNHOLD, EventType.STAY]):
      box_states_new = list(box_states)

      a1_pos_new = a1_pos
      a1_dropped = False
      bidx1 = get_box_idx(a1_pos)
      if bidx1 >= 0 and a1_act == EventType.UNHOLD:
        a1_dropped, bstate = update_dropped_box_state(bidx1, a1_pos,
                                                      box_states_new)
        if bstate[0] == BoxState.OnGoalLoc:
          a1_pos_new = a1_init

      if not a1_dropped:
        a1_pos_new = get_moved_coord(a1_pos, a1_act, box_states_new, True)

      a2_pos_new = a2_pos
      a2_dropped = False
      bidx2 = get_box_idx(a2_pos)
      if bidx2 >= 0 and a2_act == EventType.UNHOLD:
        a2_dropped, bstate = update_dropped_box_state(bidx2, a2_pos,
                                                      box_states_new)
        if bstate[0] == BoxState.OnGoalLoc:
          a2_pos_new = a2_init

      if not a2_dropped:
        a2_pos_new = get_moved_coord(a2_pos, a2_act, box_states_new, True)

      list_next_env.append((1.0, box_states_new, a1_pos_new, a2_pos_new))

    elif agent_dist == 2:
      a1_pos_new = get_moved_coord(a1_pos, a1_act, box_states, True)
      a2_pos_new = get_moved_coord(a2_pos, a2_act, box_states, True)

      if a1_pos_new == a2_pos_new:
        list_next_env.append((0.5, box_states, a1_pos_new, a2_pos))
        list_next_env.append((0.5, box_states, a1_pos, a2_pos_new))
      else:
        list_next_env.append((1.0, box_states, a1_pos_new, a2_pos_new))
    else:  # agent_dst == 1
      a1_pos_new = get_moved_coord(a1_pos, a1_act, box_states, True)
      a2_pos_new = get_moved_coord(a2_pos, a2_act, box_states, True)
      if a1_pos_new != a1_pos and a2_pos_new != a2_pos:
        list_next_env.append((1.0, box_states, a1_pos_new, a2_pos_new))
      elif a1_pos_new != a1_pos:
        a2_pos_new2 = get_moved_coord(a2_pos, a2_act, None, True)
        if a2_pos_new2 == a1_pos:
          list_next_env.append((1.0, box_states, a1_pos_new, a1_pos))
        else:  # --> a2_posnew2 == a2_pos
          list_next_env.append((1.0, box_states, a1_pos_new, a2_pos))
      elif a2_pos_new != a2_pos:
        a1_pos_new2 = get_moved_coord(a1_pos, a1_act, None, True)
        if a1_pos_new2 == a2_pos:
          list_next_env.append((1.0, box_states, a2_pos, a2_pos_new))
        else:
          list_next_env.append((1.0, box_states, a1_pos, a2_pos_new))
      else:
        list_next_env.append((1.0, box_states, a1_pos, a2_pos))
  # only a1 holds a box
  elif hold == "A1":
    box_states_new = list(box_states)
    a1_dropped = False
    a1_pos_dist = [(1.0, a1_pos)]
    if a1_act == EventType.UNHOLD:
      bidx = get_box_idx(a1_pos)
      assert bidx >= 0
      assert box_types[bidx] == 1
      a1_dropped, bstate = update_dropped_box_state(bidx, a1_pos,
                                                    box_states_new)
      # respawn
      if bstate[0] == BoxState.OnGoalLoc:
        a1_pos_dist = [(1.0, a1_init)]

    if not a1_dropped:
      a1_pos_dist = get_dist_new_coord(a1_pos, a1_act, box_states_new, True)

    a2_pickedup = False
    a2_pos_dist = [(1.0, a2_pos)]
    if a2_act == EventType.HOLD and a2_pos != a1_pos:
      bidx = get_box_idx(a2_pos)
      if bidx >= 0 and box_types[bidx] == 1:
        box_states_new[bidx] = conv_box_state_2_idx((BoxState.WithAgent2, None),
                                                    num_drops)
        a2_pickedup = True

    if not a2_pickedup:
      a2_pos_dist = get_dist_new_coord(a2_pos, a2_act, None, False)

    for p1, pos1 in a1_pos_dist:
      for p2, pos2 in a2_pos_dist:
        list_next_env.append((p1 * p2, box_states_new, pos1, pos2))
  # only a2 holds a box
  else:  # hold == "A2":
    box_states_new = list(box_states)
    a2_dropped = False
    a2_pos_dist = [(1.0, a2_pos)]
    if a2_act == EventType.UNHOLD:
      bidx = get_box_idx(a2_pos)
      assert bidx >= 0
      assert box_types[bidx] == 1
      a2_dropped, bstate = update_dropped_box_state(bidx, a2_pos,
                                                    box_states_new)
      # respawn
      if bstate[0] == BoxState.OnGoalLoc:
        a2_pos_dist = [(1.0, a2_init)]

    if not a2_dropped:
      a2_pos_dist = get_dist_new_coord(a2_pos, a2_act, box_states_new, True)

    a1_pickedup = False
    a1_pos_dist = [(1.0, a1_pos)]
    if a1_act == EventType.HOLD and a1_pos != a2_pos:
      bidx = get_box_idx(a1_pos)
      if bidx >= 0 and box_types[bidx] == 1:
        box_states_new[bidx] = conv_box_state_2_idx((BoxState.WithAgent1, None),
                                                    num_drops)
        a1_pickedup = True

    if not a1_pickedup:
      a1_pos_dist = get_dist_new_coord(a1_pos, a1_act, None, False)

    for p1, pos1 in a1_pos_dist:
      for p2, pos2 in a2_pos_dist:
        list_next_env.append((p1 * p2, box_states_new, pos1, pos2))

  return list_next_env
