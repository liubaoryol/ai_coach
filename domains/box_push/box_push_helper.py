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


def bound(coord, x_bound, y_bound):
  x, y = coord
  if x < 0:
    x = 0
  elif x >= x_bound:
    x = x_bound - 1

  if y < 0:
    y = 0
  elif y >= y_bound:
    y = y_bound - 1

  return (x, y)


def is_box_in(coord, boxes):
  return coord in boxes


def get_moved_coord(coord, direction, x_bound, y_bound, boxes=None):
  x, y = coord
  new_coord = (x, y)
  if direction == EventType.UP:
    new_coord = bound((x, y - 1), x_bound, y_bound)
  elif direction == EventType.DOWN:
    new_coord = bound((x, y + 1), x_bound, y_bound)
  elif direction == EventType.LEFT:
    new_coord = bound((x - 1, y), x_bound, y_bound)
  elif direction == EventType.RIGHT:
    new_coord = bound((x + 1, y), x_bound, y_bound)

  if boxes is not None:
    if is_box_in(new_coord, boxes):
      new_coord = (x, y)

  return new_coord


def is_opposite_direction(dir1, dir2):
  return ((dir1 == EventType.UP and dir2 == EventType.DOWN)
          or (dir1 == EventType.DOWN and dir2 == EventType.UP)
          or (dir1 == EventType.LEFT and dir2 == EventType.RIGHT)
          or (dir1 == EventType.RIGHT and dir2 == EventType.LEFT))


def transition(boxes, a1_pos, a2_pos, a1_hold, a2_hold, a1_act, a2_act, goals,
               x_bound, y_bound):
  a1_pos_new = a1_pos
  a2_pos_new = a2_pos
  a1_hold_new = a1_hold
  a2_hold_new = a2_hold
  list_next_env = []

  # both do not hold anything
  if not a1_hold and not a2_hold:
    if a1_act == EventType.HOLD:
      if is_box_in(a1_pos, boxes):
        a1_hold_new = True
    else:
      a1_pos_new = get_moved_coord(a1_pos, a1_act, x_bound, y_bound)

    if a2_act == EventType.HOLD:
      if is_box_in(a2_pos, boxes):
        a2_hold_new = True
    else:
      a2_pos_new = get_moved_coord(a2_pos, a2_act, x_bound, y_bound)

    list_next_env.append(
        (1.0, boxes, a1_pos_new, a2_pos_new, a1_hold_new, a2_hold_new))
  # both hold something
  elif a1_hold and a2_hold:
    # both hold the same box
    if a1_pos == a2_pos:
      if not is_box_in(a1_pos, boxes):
        raise RuntimeError("agents is at HOLD state without a box")

      if a1_act == EventType.STAY and a2_act == EventType.STAY:
        list_next_env.append((1.0, boxes, a1_pos, a2_pos, a1_hold, a2_hold))
      elif a1_act == EventType.UNHOLD:
        if a2_act == EventType.UNHOLD:
          list_next_env.append((1.0, boxes, a1_pos, a2_pos, False, False))
        else:  # a2_act will be move/stay
          a2_pos_new = get_moved_coord(a2_pos, a2_act, x_bound, y_bound, boxes)
          boxes_new = list(boxes)
          idx = boxes_new.index(a2_pos)
          a2_hold_new = a2_hold
          if a2_pos_new not in goals:
            boxes_new[idx] = a2_pos_new
          else:
            boxes_new[idx] = None
            a2_hold_new = False
          list_next_env.append(
              (0.5, boxes_new, a1_pos, a2_pos_new, False, a2_hold_new))
          list_next_env.append((0.5, boxes, a1_pos, a2_pos, False, a2_hold))
      elif a2_act == EventType.UNHOLD:  # a1_act will be to move / stay
        a1_pos_new = get_moved_coord(a1_pos, a1_act, x_bound, y_bound, boxes)
        boxes_new = list(boxes)
        idx = boxes_new.index(a1_pos)
        a1_hold_new = a1_hold
        if a1_pos_new not in goals:
          boxes_new[idx] = a1_pos_new
        else:
          boxes_new[idx] = None
          a1_hold_new = False
        list_next_env.append(
            (0.5, boxes_new, a1_pos_new, a2_pos, a1_hold_new, False))
        list_next_env.append((0.5, boxes, a1_pos, a2_pos, a1_hold, False))
      # same direction
      elif a1_act == a2_act:
        new_pos = get_moved_coord(a1_pos, a1_act, x_bound, y_bound, boxes)
        boxes_new = list(boxes)
        idx = boxes_new.index(a1_pos)
        hold_new = True
        if new_pos not in goals:
          boxes_new[idx] = new_pos
        else:
          boxes_new[idx] = None
          hold_new = False

        list_next_env.append(
            (1.0, boxes_new, new_pos, new_pos, hold_new, hold_new))
      # opposite directions
      elif is_opposite_direction(a1_act, a2_act):
        list_next_env.append((1.0, boxes, a1_pos, a2_pos, a1_hold, a2_hold))
      # orthogonal directions or one stays while the other moves
      else:
        new_pos1 = get_moved_coord(a1_pos, a1_act, x_bound, y_bound, boxes)
        new_pos2 = get_moved_coord(a2_pos, a2_act, x_bound, y_bound, boxes)
        list_pos = []
        # if new_pos1 != a1_pos:
        list_pos.append(new_pos1)
        # if new_pos2 != a1_pos:
        list_pos.append(new_pos2)

        if len(list_pos) == 0:
          list_next_env.append((1.0, boxes, a1_pos, a2_pos, a1_hold, a2_hold))
        else:
          p_next = 1 / len(list_pos)
          for pos in list_pos:
            boxes_new = list(boxes)
            idx = boxes_new.index(a1_pos)
            hold_new = True
            if pos not in goals:
              boxes_new[idx] = pos
            else:
              boxes_new[idx] = None
              hold_new = False
            list_next_env.append(
                (p_next, boxes_new, pos, pos, hold_new, hold_new))
    # each holds different bags (a1_pos != a2_pos)
    else:
      a1_new = None
      if a1_act == EventType.STAY:
        a1_new = (a1_pos, a1_hold)
      elif a1_act == EventType.UNHOLD:
        a1_new = (a1_pos, False)

      a2_new = None
      if a2_act == EventType.STAY:
        a2_new = (a2_pos, a2_hold)
      elif a2_act == EventType.UNHOLD:
        a2_new = (a2_pos, False)

      if a1_new is not None and a2_new is not None:
        list_next_env.append(
            (1.0, boxes, a1_new[0], a2_new[0], a1_new[1], a2_new[1]))
      elif a1_new is not None:  # a2_act is move
        a2_pos_new = get_moved_coord(a2_pos, a2_act, x_bound, y_bound, boxes)
        boxes_new = list(boxes)
        idx = boxes_new.index(a2_pos)
        a2_hold_new = a2_hold
        if a2_pos_new not in goals:
          boxes_new[idx] = a2_pos_new
        else:
          boxes_new[idx] = None
          a2_hold_new = False
        list_next_env.append(
            (0.5, boxes_new, a1_new[0], a2_pos_new, a1_new[1], a2_hold_new))
        list_next_env.append(
            (0.5, boxes, a1_new[0], a2_pos, a1_new[1], a2_hold))
      elif a2_new is not None:  # a1_act is move
        a1_pos_new = get_moved_coord(a1_pos, a1_act, x_bound, y_bound, boxes)
        boxes_new = list(boxes)
        idx = boxes_new.index(a1_pos)
        a1_hold_new = a1_hold
        if a1_pos_new not in goals:
          boxes_new[idx] = a1_pos_new
        else:
          boxes_new[idx] = None
          a1_hold_new = False
        list_next_env.append(
            (0.5, boxes_new, a1_pos_new, a2_new[0], a1_hold_new, a2_new[1]))
        list_next_env.append(
            (0.5, boxes, a1_pos, a2_new[0], a1_hold, a2_new[1]))
      else:  # a1_new is None and a2_new is None
        agent_dist = (abs(a1_pos[0] - a2_pos[0]) + abs(a1_pos[1] - a2_pos[1]))
        if agent_dist > 2:
          a1_pos_new = get_moved_coord(a1_pos, a1_act, x_bound, y_bound, boxes)
          a2_pos_new = get_moved_coord(a2_pos, a2_act, x_bound, y_bound, boxes)
          boxes_new1 = list(boxes)
          idx1 = boxes_new1.index(a1_pos)
          a1_hold_new = a1_hold
          if a1_pos_new not in goals:
            boxes_new1[idx1] = a1_pos_new
          else:
            boxes_new1[idx1] = None
            a1_hold_new = False

          boxes_new2 = list(boxes)
          boxes_new3 = list(boxes_new1)
          idx2 = boxes_new2.index(a2_pos)
          a2_hold_new = a2_hold
          if a2_pos_new not in goals:
            boxes_new2[idx2] = a2_pos_new
            boxes_new3[idx2] = a2_pos_new
          else:
            boxes_new2[idx2] = None
            boxes_new3[idx2] = None
            a2_hold_new = False
          list_next_env.append((0.25, boxes, a1_pos, a2_pos, a1_hold, a2_hold))
          list_next_env.append(
              (0.25, boxes_new1, a1_pos_new, a2_pos, a1_hold_new, a2_hold))
          list_next_env.append(
              (0.25, boxes_new2, a1_pos, a2_pos_new, a1_hold, a2_hold_new))
          list_next_env.append((0.25, boxes_new3, a1_pos_new, a2_pos_new,
                                a1_hold_new, a2_hold_new))
        elif agent_dist == 2:
          a1_pos_new = get_moved_coord(a1_pos, a1_act, x_bound, y_bound, boxes)
          a2_pos_new = get_moved_coord(a2_pos, a2_act, x_bound, y_bound, boxes)
          if a1_pos_new != a2_pos_new:
            boxes_new1 = list(boxes)
            idx1 = boxes_new1.index(a1_pos)
            a1_hold_new = a1_hold
            if a1_pos_new not in goals:
              boxes_new1[idx1] = a1_pos_new
            else:
              boxes_new1[idx1] = None
              a1_hold_new = False

            boxes_new2 = list(boxes)
            boxes_new3 = list(boxes_new1)
            idx2 = boxes_new2.index(a2_pos)
            a2_hold_new = a2_hold
            if a2_pos_new not in goals:
              boxes_new2[idx2] = a2_pos_new
              boxes_new3[idx2] = a2_pos_new
            else:
              boxes_new2[idx2] = None
              boxes_new3[idx2] = None
              a2_hold_new = False
            list_next_env.append(
                (0.25, boxes, a1_pos, a2_pos, a1_hold, a2_hold))
            list_next_env.append(
                (0.25, boxes_new1, a1_pos_new, a2_pos, a1_hold_new, a2_hold))
            list_next_env.append(
                (0.25, boxes_new2, a1_pos, a2_pos_new, a1_hold, a2_hold_new))
            list_next_env.append((0.25, boxes_new3, a1_pos_new, a2_pos_new,
                                  a1_hold_new, a2_hold_new))
          else:
            boxes_new1 = list(boxes)
            idx1 = boxes_new1.index(a1_pos)
            a1_hold_new = a1_hold
            if a1_pos_new not in goals:
              boxes_new1[idx1] = a1_pos_new
            else:
              boxes_new1[idx1] = None
              a1_hold_new = False

            boxes_new2 = list(boxes)
            idx2 = boxes_new2.index(a2_pos)
            a2_hold_new = a2_hold
            if a2_pos_new not in goals:
              boxes_new2[idx2] = a2_pos_new
            else:
              boxes_new2[idx2] = None
              a2_hold_new = False

            list_next_env.append(
                (0.375, boxes_new1, a1_pos_new, a2_pos, a1_hold_new, a2_hold))
            list_next_env.append(
                (0.375, boxes_new2, a1_pos, a2_pos_new, a1_hold, a2_hold_new))
            list_next_env.append(
                (0.25, boxes, a1_pos, a2_pos, a1_hold, a2_hold))
        else:  # agent_dist == 1
          # case1: a1 moves first
          prop1 = 0.5
          prop2 = prop1 * 0.5
          # case1-1: a1 succeeded to move
          a1_pos_new = get_moved_coord(a1_pos, a1_act, x_bound, y_bound, boxes)
          boxes_new = list(boxes)
          idx = boxes_new.index(a1_pos)
          a1_hold_new = a1_hold
          if a1_pos_new not in goals:
            boxes_new[idx] = a1_pos_new
          else:
            boxes_new[idx] = None
            a1_hold_new = False

          a2_pos_new = get_moved_coord(a2_pos, a2_act, x_bound, y_bound,
                                       boxes_new)
          boxes_new2 = list(boxes_new)
          idx2 = boxes_new.index(a2_pos)
          a2_hold_new = a2_hold
          if a2_pos_new not in goals:
            boxes_new2[idx2] = a2_pos_new
          else:
            boxes_new2[idx2] = None
            a2_hold_new = False

          # case1-1-1: a2 succeeded to move
          list_next_env.append((prop2 * 0.5, boxes_new2, a1_pos_new, a2_pos_new,
                                a1_hold_new, a2_hold_new))
          # case1-1-2: a2 failed to move
          list_next_env.append((prop2 * 0.5, boxes_new, a1_pos_new, a2_pos,
                                a1_hold_new, a2_hold))

          # case1-2: a1 failed to move
          a2_pos_new = get_moved_coord(a2_pos, a2_act, x_bound, y_bound, boxes)
          boxes_new = list(boxes)
          idx = boxes_new.index(a2_pos)
          a2_hold_new = a2_hold
          if a2_pos_new not in goals:
            boxes_new[idx] = a2_pos_new
          else:
            boxes_new[idx] = None
            a2_hold_new = False

          # case1-2-1: a2 succeeded
          list_next_env.append((prop2 * 0.5, boxes_new, a1_pos, a2_pos_new,
                                a1_hold, a2_hold_new))
          # case1-2-2: a2 failed
          list_next_env.append(
              (prop2 * 0.5, boxes, a1_pos, a2_pos, a1_hold, a2_hold))

          # case2: a2 moves first
          # case2-1: a2 succeeded to move
          a2_pos_new = get_moved_coord(a2_pos, a2_act, x_bound, y_bound, boxes)
          boxes_new = list(boxes)
          idx = boxes_new.index(a2_pos)
          a2_hold_new = a2_hold
          if a2_pos_new not in goals:
            boxes_new[idx] = a2_pos_new
          else:
            boxes_new[idx] = None
            a2_hold_new = False

          a1_pos_new = get_moved_coord(a1_pos, a1_act, x_bound, y_bound,
                                       boxes_new)
          boxes_new2 = list(boxes_new)
          idx2 = boxes_new.index(a1_pos)
          a1_hold_new = a1_hold
          if a1_pos_new not in goals:
            boxes_new2[idx2] = a1_pos_new
          else:
            boxes_new2[idx2] = None
            a1_hold_new = False

          # case1-2-1: a1 succeeded to move
          list_next_env.append((prop2 * 0.5, boxes_new2, a1_pos_new, a2_pos_new,
                                a1_hold_new, a2_hold_new))
          # case1-2-2: a1 failed to move
          list_next_env.append((prop2 * 0.5, boxes_new, a1_pos, a2_pos_new,
                                a1_hold, a2_hold_new))

          # case2-2: a2 failed to move
          a1_pos_new = get_moved_coord(a1_pos, a1_act, x_bound, y_bound, boxes)
          boxes_new = list(boxes)
          idx = boxes_new.index(a1_pos)
          a1_hold_new = a1_hold
          if a1_pos_new not in goals:
            boxes_new[idx] = a1_pos_new
          else:
            boxes_new[idx] = None
            a1_hold_new = False

          # case1-2-1: a1 succeeded
          list_next_env.append((prop2 * 0.5, boxes_new, a1_pos_new, a2_pos,
                                a1_hold_new, a2_hold))
          # case1-2-2: a1 failed
          list_next_env.append(
              (prop2 * 0.5, boxes, a1_pos, a2_pos, a1_hold, a2_hold))
  # only a1 holds something
  elif a1_hold:
    if a2_act == EventType.HOLD and a1_pos == a2_pos:
      # a2 action is HOLD, the bag cannot be moved.
      a1_hold_new = False if a1_act == EventType.UNHOLD else True
      list_next_env.append((1.0, boxes, a1_pos, a2_pos, a1_hold_new, True))
    else:
      a1_hold_new = a1_hold
      a1_pos_new = a1_pos
      boxes_new = boxes
      if a1_act == EventType.UNHOLD:
        a1_hold_new = False
      elif a1_act != EventType.STAY:
        a1_pos_new = get_moved_coord(a1_pos, a1_act, x_bound, y_bound, boxes)
        boxes_new = list(boxes)
        idx = boxes_new.index(a1_pos)
        if a1_pos_new not in goals:
          boxes_new[idx] = a1_pos_new
        else:
          boxes_new[idx] = None
          a1_hold_new = False

      a2_hold_new = a2_hold
      a2_pos_new = a2_pos
      if a2_act == EventType.HOLD:
        if is_box_in(a2_pos, boxes):
          a2_hold_new = True
      elif a2_act != EventType.STAY:
        a2_pos_new = get_moved_coord(a2_pos, a2_act, x_bound, y_bound)

      list_next_env.append(
          (0.5, boxes, a1_pos, a2_pos_new, a1_hold, a2_hold_new))
      list_next_env.append(
          (0.5, boxes_new, a1_pos_new, a2_pos_new, a1_hold_new, a2_hold_new))
  else:  # a2_hold
    if a1_act == EventType.HOLD and a1_pos == a2_pos:
      # a1 action is HOLD, the bag cannot be moved.
      a2_hold_new = False if a2_act == EventType.UNHOLD else True
      list_next_env.append((1.0, boxes, a1_pos, a2_pos, True, a2_hold_new))
    else:
      a2_hold_new = a2_hold
      a2_pos_new = a2_pos
      boxes_new = boxes
      if a2_act == EventType.UNHOLD:
        a2_hold_new = False
      elif a2_act != EventType.STAY:
        a2_pos_new = get_moved_coord(a2_pos, a2_act, x_bound, y_bound, boxes)
        boxes_new = list(boxes)
        idx = boxes_new.index(a2_pos)
        if a2_pos_new not in goals:
          boxes_new[idx] = a2_pos_new
        else:
          boxes_new[idx] = None
          a2_hold_new = False

      a1_hold_new = a1_hold
      a1_pos_new = a1_pos
      if a1_act == EventType.HOLD:
        if is_box_in(a1_pos):
          a1_hold_new = True
      elif a1_act != EventType.STAY:
        a1_pos_new = get_moved_coord(a1_pos, a1_act, x_bound, y_bound)

      list_next_env.append(
          (0.5, boxes, a1_pos_new, a2_pos, a1_hold_new, a2_hold))
      list_next_env.append(
          (0.5, boxes_new, a1_pos_new, a2_pos_new, a1_hold_new, a2_hold_new))

  return list_next_env
