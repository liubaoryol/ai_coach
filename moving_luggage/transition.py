import numpy as np
from moving_luggage.constants import NUM_X_GRID, NUM_Y_GRID, AgentActions


def bound(coord):
    x, y = coord
    if x < 0:
        x = 0
    elif x >= NUM_X_GRID:
        x = NUM_X_GRID - 1
    
    if y < 0:
        y = 0
    elif y >= NUM_Y_GRID:
        y = NUM_Y_GRID - 1

    return (x, y)


def get_moved_coord(coord, direction, np_bags=None):
    x, y = coord
    new_coord = (x, y)
    if direction == AgentActions.UP:
        new_coord = bound((x, y - 1))
    elif direction == AgentActions.DOWN:
        new_coord = bound((x, y + 1))
    elif direction == AgentActions.LEFT:
        new_coord = bound((x - 1, y))
    elif direction == AgentActions.RIGHT:
        new_coord = bound((x + 1, y))
    
    if np_bags is not None:
        if np_bags[new_coord] != 0:
            new_coord = (x, y)

    return new_coord


def is_opposite_direction(dir1, dir2):
    return (
        (dir1 == AgentActions.UP and dir2 == AgentActions.DOWN) or
        (dir1 == AgentActions.DOWN and dir2 == AgentActions.UP) or
        (dir1 == AgentActions.LEFT and dir2 == AgentActions.RIGHT) or
        (dir1 == AgentActions.RIGHT and dir2 == AgentActions.LEFT))


def transition(
    np_bags, a1_pos, a2_pos, a1_hold, a2_hold, a1_act, a2_act, goals):
    a1_pos_new = a1_pos
    a2_pos_new = a2_pos
    a1_hold_new = a1_hold
    a2_hold_new = a2_hold
    list_next_env = []
    # both do not hold anything
    if not a1_hold and not a2_hold:
        if a1_act == AgentActions.HOLD:
            if np_bags[a1_pos] != 0:
                a1_hold_new = True
        else:
            a1_pos_new = get_moved_coord(a1_pos, a1_act)
        
        if a2_act == AgentActions.HOLD:
            if np_bags[a2_pos] != 0:
                a2_hold_new = True
        else:
            a2_pos_new = get_moved_coord(a2_pos, a2_act)
            
        list_next_env.append(
            (1.0, np_bags, a1_pos_new, a2_pos_new, a1_hold_new, a2_hold_new))
    # both hold something
    elif a1_hold and a2_hold:
        # both hold the same bag
        if a1_pos == a2_pos:
            if np_bags[a1_pos] == 0:
                raise RuntimeError("agents is at HOLD state without a bag")

            if a1_act == AgentActions.STAY or a2_act == AgentActions.STAY:
                list_next_env.append(
                    (1.0, np_bags, a1_pos, a2_pos, a1_hold, a2_hold))
            elif a1_act == AgentActions.HOLD:
                if a2_act == AgentActions.HOLD:
                    list_next_env.append(
                        (1.0, np_bags, a1_pos, a2_pos, False, False))
                else:
                    a2_pos_new = get_moved_coord(a2_pos, a2_act, np_bags)
                    np_bags_new = np.copy(np_bags)
                    np_bags_new[a2_pos] = 0
                    a2_hold_new = a2_hold
                    if a2_pos_new not in goals:
                        np_bags_new[a2_pos_new] = 1
                    else:
                        a2_hold_new = False
                    list_next_env.append((
                        1.0, np_bags_new,
                        a1_pos, a2_pos_new, False, a2_hold_new))
            elif a2_act == AgentActions.HOLD:
                    a1_pos_new = get_moved_coord(a1_pos, a1_act, np_bags)
                    np_bags_new = np.copy(np_bags)
                    np_bags_new[a1_pos] = 0
                    a1_hold_new = a1_hold
                    if a1_pos_new not in goals:
                        np_bags_new[a1_pos_new] = 1
                    else:
                        a1_hold_new = False
                    list_next_env.append((
                        1.0, np_bags_new,
                        a1_pos_new, a2_pos, a1_hold_new, False))
            # same direction
            elif a1_act == a2_act:
                new_pos = get_moved_coord(a1_pos, a1_act, np_bags)
                np_bags_new = np.copy(np_bags)
                np_bags_new[a1_pos] = 0
                hold_new = True
                if new_pos not in goals:
                    np_bags_new[new_pos] = 1
                else:
                    hold_new = False
                list_next_env.append(
                    (1.0, np_bags_new, new_pos, new_pos, hold_new, hold_new))
            # opposite directions
            elif is_opposite_direction(a1_act, a2_act):
                list_next_env.append(
                    (1.0, np_bags, a1_pos, a2_pos, a1_hold, a2_hold))
            # orthogonal directions
            else:
                new_pos1 = get_moved_coord(a1_pos, a1_act, np_bags)
                new_pos2 = get_moved_coord(a2_pos, a2_act, np_bags)
                list_pos = []
                if new_pos1 != a1_pos:
                    list_pos.append(new_pos1)
                if new_pos2 != a1_pos:
                    list_pos.append(new_pos2)

                if len(list_pos) == 0:
                    list_next_env.append(
                        (1.0, np_bags, a1_pos, a2_pos, a1_hold, a2_hold))
                else:
                    p_next = 1 / len(list_pos)
                    for pos in list_pos:
                        np_new = np.copy(np_bags)
                        np_new[a1_pos] = 0
                        hold_new = True
                        if pos not in goals:
                            np_new[pos] = 1
                        else:
                            hold_new = False
                        list_next_env.append(
                            (p_next, np_new, pos, pos, hold_new, hold_new))
        # each holds different bags (a1_pos != a2_pos)
        else:
            a1_new = None
            if a1_act == AgentActions.STAY:
                a1_new = (a1_pos, a1_hold)
            elif a1_act == AgentActions.HOLD:
                a1_new = (a1_pos, False)

            a2_new = None
            if a2_act == AgentActions.STAY:
                a2_new = (a2_pos, a2_hold)
            elif a2_act == AgentActions.HOLD:
                a2_new = (a2_pos, False)

            if a1_new is not None and a2_new is not None:
                list_next_env.append(
                    (1.0, np_bags, a1_new[0], a2_new[0], a1_new[1], a2_new[1]))
            elif a1_new is not None:
                a2_pos_new = get_moved_coord(a2_pos, a2_act, np_bags)
                np_new = np.copy(np_bags)
                np_new[a2_pos] = 0
                a2_hold_new = a2_hold
                if a2_pos_new not in goals:
                    np_new[a2_pos_new] = 1
                else:
                    a2_hold_new = False
                list_next_env.append((
                    1.0, np_new,
                    a1_new[0], a2_pos_new, a1_new[1], a2_hold_new))
            elif a2_new is not None:
                a1_pos_new = get_moved_coord(a1_pos, a1_act, np_bags)
                np_new = np.copy(np_bags)
                np_new[a1_pos] = 0
                a1_hold_new = a1_hold
                if a1_pos_new not in goals:
                    np_new[a1_pos_new] = 1
                else:
                    a1_hold_new = False
                list_next_env.append((
                    1.0, np_new,
                    a1_pos_new, a2_new[0], a1_hold_new, a2_new[1]))
            else:  # a1_new is None and a2_new is None
                agent_dist = (
                    abs(a1_pos[0] - a2_pos[0]) + abs(a1_pos[1] - a2_pos[1]))
                if agent_dist > 2:
                    a1_pos_new = get_moved_coord(a1_pos, a1_act, np_bags)
                    a2_pos_new = get_moved_coord(a2_pos, a2_act, np_bags)
                    np_new = np.copy(np_bags)
                    np_new[a1_pos] = 0
                    a1_hold_new = a1_hold
                    if a1_pos_new not in goals:
                        np_new[a1_pos_new] = 1
                    else:
                        a1_hold_new = False
                    np_new[a2_pos] = 0
                    a2_hold_new = a2_hold
                    if a2_pos_new not in goals:
                        np_new[a2_pos_new] = 1
                    else:
                        a2_hold_new = False
                    list_next_env.append((
                        1.0, np_new,
                        a1_pos_new, a2_pos_new, a1_hold_new, a2_hold_new))
                elif agent_dist == 2:
                    a1_pos_new = get_moved_coord(a1_pos, a1_act, np_bags)
                    a2_pos_new = get_moved_coord(a2_pos, a2_act, np_bags)
                    if a1_pos_new != a2_pos_new:
                        np_new = np.copy(np_bags)
                        np_new[a1_pos] = 0
                        a1_hold_new = a1_hold
                        if a1_pos_new not in goals:
                            np_new[a1_pos_new] = 1
                        else:
                            a1_hold_new = False
                        np_new[a2_pos] = 0
                        a2_hold_new = a2_hold
                        if a2_pos_new not in goals:
                            np_new[a2_pos_new] = 1
                        else:
                            a2_hold_new = False
                        list_next_env.append((
                                1.0, np_new, a1_pos_new, a2_pos_new,
                                a1_hold_new, a2_hold_new))
                    else:
                        np_new1 = np.copy(np_bags)
                        np_new1[a1_pos] = 0
                        a1_hold_new = a1_hold
                        if a1_pos_new not in goals:
                            np_new1[a1_pos_new] = 1
                        else:
                            a1_hold_new = False
                        list_next_env.append((
                                0.5, np_new1, a1_pos_new,
                                a2_pos, a1_hold_new, a2_hold))
                        np_new2 = np.copy(np_bags)
                        np_new2[a2_pos] = 0
                        a2_hold_new = a2_hold
                        if a2_pos_new not in goals:
                            np_new2[a2_pos_new] = 1
                        else:
                            a2_hold_new = False
                        list_next_env.append((
                                0.5, np_new2, a1_pos,
                                a2_pos_new, a1_hold, a2_hold_new))
                else:  # agent_dist == 1
                    a1_pos_chk = get_moved_coord(a1_pos, a1_act)
                    a2_pos_chk = get_moved_coord(a2_pos, a2_act)
                    if a1_pos_chk == a2_pos:
                        # move a2 first
                        a2_pos_new = get_moved_coord(a2_pos, a2_act, np_bags)
                        np_new = np.copy(np_bags)
                        np_new[a2_pos] = 0
                        a2_hold_new = a2_hold
                        if a2_pos_new not in goals:
                            np_new[a2_pos_new] = 1
                        else:
                            a2_hold_new = False
                        # then, move a1
                        a1_pos_new = get_moved_coord(a1_pos, a1_act, np_new)
                        np_new[a1_pos] = 0
                        a1_hold_new = a1_hold
                        if a1_pos_new not in goals:
                            np_new[a1_pos_new] = 1
                        else:
                            a1_hold_new = False
                        list_next_env.append((
                                1.0, np_new, a1_pos_new,
                                a2_pos_new, a1_hold_new, a2_hold_new))
                    elif a2_pos_chk == a1_pos:
                        # move a1 first
                        a1_pos_new = get_moved_coord(a1_pos, a1_act, np_bags)
                        np_new = np.copy(np_bags)
                        np_new[a1_pos] = 0
                        a1_hold_new = a1_hold
                        if a1_pos_new not in goals:
                            np_new[a1_pos_new] = 1
                        else:
                            a1_hold_new = False
                        # then, move a2
                        a2_pos_new = get_moved_coord(a2_pos, a2_act, np_new)
                        np_new[a2_pos] = 0
                        a2_hold_new = a2_hold
                        if a2_pos_new not in goals:
                            np_new[a2_pos_new] = 1
                        else:
                            a2_hold_new = False
                        list_next_env.append((
                                1.0, np_new, a1_pos_new,
                                a2_pos_new, a1_hold_new, a2_hold_new))
                    else:
                        a1_pos_new = get_moved_coord(a1_pos, a1_act, np_bags)
                        a2_pos_new = get_moved_coord(a2_pos, a2_act, np_bags)
                        np_new = np.copy(np_bags)
                        np_new[a1_pos] = 0
                        a1_hold_new = a1_hold
                        if a1_pos_new not in goals:
                            np_new[a1_pos_new] = 1
                        else:
                            a1_hold_new = False
                        np_new[a2_pos] = 0
                        a2_hold_new = a2_hold
                        if a2_pos_new not in goals:
                            np_new[a2_pos_new] = 1
                        else:
                            a2_hold_new = False
                        list_next_env.append((
                                1.0, np_new, a1_pos_new,
                                a2_pos_new, a1_hold_new, a2_hold_new))
    # only a1 holds something
    elif a1_hold:
        if a2_act == AgentActions.HOLD and a1_pos == a2_pos:
            # a2 action is HOLD, the bag cannot be moved. 
            a1_hold_new = False if a1_act == AgentActions.HOLD else True
            list_next_env.append(
                (1.0, np_bags, a1_pos, a2_pos, a1_hold_new, True))
        else:
            a1_hold_new = a1_hold
            a1_pos_new = a1_pos
            np_bags_new = np_bags
            if a1_act == AgentActions.HOLD:
                a1_hold_new = False
            elif a1_act != AgentActions.STAY:
                a1_pos_new = get_moved_coord(a1_pos, a1_act, np_bags)
                np_bags_new = np.copy(np_bags)
                np_bags_new[a1_pos] = 0
                if a1_pos_new not in goals:
                    np_bags_new[a1_pos_new] = 1
                else:
                    a1_hold_new = False
            
            a2_hold_new = a2_hold
            a2_pos_new = a2_pos
            if a2_act == AgentActions.HOLD:
                if np_bags[a2_pos] != 0:
                    a2_hold_new = True
            elif a2_act != AgentActions.STAY:
                a2_pos_new = get_moved_coord(a2_pos, a2_act)

            list_next_env.append((
                1.0, np_bags_new, a1_pos_new, a2_pos_new,
                a1_hold_new, a2_hold_new))
    else:  # a2_hold
        if a1_act == AgentActions.HOLD and a1_pos == a2_pos:
            # a1 action is HOLD, the bag cannot be moved. 
            a2_hold_new = False if a2_act == AgentActions.HOLD else True
            list_next_env.append(
                (1.0, np_bags, a1_pos, a2_pos, True, a2_hold_new))
        else:
            a2_hold_new = a2_hold
            a2_pos_new = a2_pos
            np_bags_new = np_bags
            if a2_act == AgentActions.HOLD:
                a2_hold_new = False
            elif a2_act != AgentActions.STAY:
                a2_pos_new = get_moved_coord(a2_pos, a2_act, np_bags)
                np_bags_new = np.copy(np_bags)
                np_bags_new[a2_pos] = 0
                if a2_pos_new not in goals:
                    np_bags_new[a2_pos_new] = 1
                else:
                    a2_hold_new = False
            
            a1_hold_new = a1_hold
            a1_pos_new = a1_pos
            if a1_act == AgentActions.HOLD:
                if np_bags[a1_pos] != 0:
                    a1_hold_new = True
            elif a1_act != AgentActions.STAY:
                a1_pos_new = get_moved_coord(a1_pos, a1_act)

            list_next_env.append((
                1.0, np_bags_new, a1_pos_new, a2_pos_new,
                a1_hold_new, a2_hold_new))

    return list_next_env


# def transitions(env, agent1, agent2, action1, action2):
#     # obstacles
#     np_obstacles = np.zeros((self.grid_x, self.grid_y))
#     for obj_key in env:
#         if obj_key[0] == ObjNames.BAG:
#             np_obstacles(env[obj_key].coord) = 1

#     if not agent1.hold and not agent2.hold:
#         if action1 == AgentActions.HOLD:
#             if np_obstacles[agent1.coord] != 0:
#                 agent1.hold = True
#             else:
#                 raise RuntimeError("invalid hold action")
#         else:
#             move_object(agent1, action1)

#         if action2 == AgentActions.HOLD:
#             if np_obstacles[agent2.coord] != 0:
#                 agent2.hold = True
#             else:
#                 raise RuntimeError("invalid hold action")
#         else:
#             move_object(agent2, action2)
#     # move one object together
#     elif agent1.hold and agent2.hold and agent1.coord == agent2.coord:
#         for obj_key in env:
#             if obj_key[0] == ObjNames.BAG:
#                 bag_obj = env[obj_key]
#                 if bag_obj.coord == agent1.coord:
#                     if (
#                         action1 == AgentActions.STAY or
#                         action2 == AgentActions.STAY
#                         ):
#                         pass
#                     elif action1 == AgentActions.UP:
#                         if (
#                             action2 == AgentActions.UP or
#                             action2 == AgentActions.LEFT or
#                             action2 == AgentActions.RIGHT or
#                             action2 == AgentActions.HOLD
#                             ):
#                             move_object(agent1, action1)
#                             move_object(agent2, action1)
#                             move_object(bag_obj, action1)
#                             if (
#                                 action2 == AgentActions.LEFT or
#                                 action2 == AgentActions.RIGHT
#                                 ):
#                                 move_object(agent1, action2)
#                                 move_object(agent2, action2)
#                                 move_object(bag_obj, action2)
#                             elif action2 == AgentActions.HOLD:
#                                 agent2.hold = False
#                         elif action2 == AgentActions.DOWN:
#                             pass
#                     elif action1 == AgentActions.DOWN:
#                         if (
#                             action2 == AgentActions.DOWN or
#                             action2 == AgentActions.LEFT or
#                             action2 == AgentActions.RIGHT or
#                             action2 == AgentActions.HOLD
#                             ):
#                             move_object(agent1, action1)
#                             move_object(agent2, action1)
#                             move_object(bag_obj, action1)
#                             if (
#                                 action2 == AgentActions.LEFT or
#                                 action2 == AgentActions.RIGHT
#                                 ):
#                                 move_object(agent1, action2)
#                                 move_object(agent2, action2)
#                                 move_object(bag_obj, action2)
#                             elif action2 == AgentActions.HOLD:
#                                 agent2.hold = False
#                         elif action2 == AgentActions.UP:
#                             pass
#                     elif action1 == AgentActions.LEFT:
#                         if (
#                             action2 == AgentActions.UP or
#                             action2 == AgentActions.DOWN or
#                             action2 == AgentActions.LEFT or
#                             action2 == AgentActions.HOLD
#                             ):
#                             move_object(agent1, action1)
#                             move_object(agent2, action1)
#                             move_object(bag_obj, action1)

#                             if (
#                                 action2 == AgentActions.UP or
#                                 action2 == AgentActions.DOWN
#                                 ):
#                                 move_object(agent1, action2)
#                                 move_object(agent2, action2)
#                                 move_object(bag_obj, action2)
#                             elif action2 == AgentActions.HOLD:
#                                 agent2.hold = False
#                         elif action2 == AgentActions.RIGHT:
#                             pass
#                     elif action1 == AgentActions.RIGHT:
#                         if (
#                             action2 == AgentActions.UP or
#                             action2 == AgentActions.DOWN or
#                             action2 == AgentActions.RIGHT or
#                             action2 == AgentActions.HOLD
#                             ):
#                             move_object(agent1, action1)
#                             move_object(agent2, action1)
#                             move_object(bag_obj, action1)

#                             if (
#                                 action2 == AgentActions.UP or
#                                 action2 == AgentActions.DOWN
#                                 ):
#                                 move_object(agent1, action2)
#                                 move_object(agent2, action2)
#                                 move_object(bag_obj, action2)
#                             elif action2 == AgentActions.HOLD:
#                                 agent2.hold = False
#                         elif action2 == AgentActions.LEFT:
#                             pass
#                     elif action1 == AgentActions.HOLD:
#                         agent1.hold = False
#                         if action2 == AgentActions.HOLD:
#                             agent2.hold = False
#                         else:
#                             move_object(agent2, action2)
#                             move_object(bag_obj, action2)
#                     break
#     else:
#         if agent1.hold:
#             for obj_key in env:
#                 if obj_key[0] == ObjNames.BAG:
#                     bag_obj = env[obj_key]
#                     if bag_obj.coord == agent1.coord:
#                         if action1 == AgentActions.STAY:
#                             pass
#                         elif action1 == AgentActions.HOLD:
#                             agent1.hold = False
#                         else:
#                             move_object(agent1, action1)
#                             move_object(bag_obj, action1)
#                         break
#         if agent2.hold:
#             for obj_key in env:
#                 if obj_key[0] == ObjNames.BAG:
#                     bag_obj = env[obj_key]
#                     if bag_obj.coord == agent2.coord:
#                         if action2 == AgentActions.STAY:
#                             pass
#                         elif action2 == AgentActions.HOLD:
#                             agent2.hold = False
#                         else:
#                             move_object(agent2, action2)
#                             move_object(bag_obj, action2)
#                         break

