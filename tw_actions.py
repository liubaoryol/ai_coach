from tw_data import *
from tw_utils import *
from tw_timer import CTimingEvent
import copy


def move_left_impl(rendering_obj):
    x_cur, y_cur = rendering_obj.get_pos()
    x_size, y_size = rendering_obj.get_size()
    x_nxt = x_cur - 1

    for y_nxt in range(y_cur, y_cur + y_size):
        if is_boundary((x_nxt, y_nxt)):
            # log: already occupied
            return False

    rendering_obj.set_pos((x_nxt, y_cur))
    return True


def move_right_impl(rendering_obj):
    x_cur, y_cur = rendering_obj.get_pos()
    x_size, y_size = rendering_obj.get_size()
    x_nxt = x_cur + 1

    for y_nxt in range(y_cur, y_cur + y_size):
        if is_boundary((x_cur + x_size, y_nxt)):
            # log: out of the world
            return False

    rendering_obj.set_pos((x_nxt, y_cur))
    return True


def move_up_impl(rendering_obj):
    x_cur, y_cur = rendering_obj.get_pos()
    x_size, y_size = rendering_obj.get_size()
    y_nxt = y_cur - 1

    for x_nxt in range(x_cur, x_cur + x_size):
        if is_boundary((x_nxt, y_nxt)):
            # log: out of the world
            return False

    rendering_obj.set_pos((x_cur, y_nxt))
    return True


def move_down_impl(rendering_obj):
    x_cur, y_cur = rendering_obj.get_pos()
    x_size, y_size = rendering_obj.get_size()
    y_nxt = y_cur + 1

    for x_nxt in range(x_cur, x_cur + x_size):
        if is_boundary((x_nxt, y_nxt)):
            # log: out of the world
            return False

    rendering_obj.set_pos((x_cur, y_nxt))
    return True


def move(obj_name, dir):
    obj = OBJECT_MAP[obj_name]
    if obj is None:
        return False

    if obj.get_mode() != MODE_IDLE:
        return False

    moved = False
    if dir == ACTION_LEFT:
        moved = move_left_impl(obj)
    elif dir == ACTION_RIGHT:
        moved = move_right_impl(obj)
    elif dir == ACTION_UP:
        moved = move_up_impl(obj)
    elif dir == ACTION_DOWN:
        moved = move_down_impl(obj)
    else:
        return False

    return moved


def toggle_mode(obj_name, timers, data_changed=[]):
    obj = OBJECT_MAP[obj_name]
    if obj is None:
        return False

    # toggle mode only when a contamination is underneath
    obj_q = None
    for obj_n in get_position_objs(obj.get_pos()):
        if obj_n.get_type() in [OBJ_TYPE_1, OBJ_TYPE_2, OBJ_TYPE_3]:
            obj_q = obj_n
            break

    if obj_q is None:
        print("no object")
        return False

    if obj.get_mode() == MODE_IN_ACTION:
        obj.set_mode(MODE_IDLE)
        data_changed.append(obj_name)

        # see if the cleaning should be stopped
        if not can_clean(obj_q):
            # stall cleaning
            tm = timers.get_timer(obj_q.get_name(), "clean")
            if tm is not None:
                t_clean_st = tm.set_time - (obj_q.get_time2clean() -
                                            obj_q.get_time_cleaned())
                t_passed = timers.get_current_time() - t_clean_st
                obj_q.set_time_cleaned(obj_q.get_time_cleaned() + t_passed)
                if obj_q.get_time_cleaned() >= obj_q.get_time2clean():
                    done_cleaning(obj_q, data_changed)
            timers.remove_timer(obj_q.get_name(), "clean")

        return True

    elif obj.get_mode() == MODE_IDLE:
        obj.set_mode(MODE_IN_ACTION)
        data_changed.append(obj_name)

        if can_clean(obj_q):
            def clean_process(t_excute, obj_name_c, changed_items):
                list_changed = []
                obj_c = OBJECT_MAP[obj_name_c]
                if done_cleaning(obj_c, list_changed):
                    # find any timers of this obj
                    # maybe we need "nonlocal" keyword here
                    timers.remove_timer(obj_name_c)
                    if changed_items is not None:
                        for name_changed in list_changed:
                            changed_items.add(name_changed)

            t_c = obj_q.get_time2clean()
            t_c_ed = obj_q.get_time_cleaned()
            cleaning_event = CTimingEvent(name=obj_q.get_name(), tag="clean")
            cleaning_event.set_time = timers.get_current_time() + t_c - t_c_ed
            cleaning_event.callback = (lambda t_ex, l_name, changed:
                                       clean_process(t_ex, l_name, changed))
            timers.add_timer(cleaning_event)

        return True

    print("unknown mode")
    return False


def pickup(obj_name, timer, data_changed):
    obj = OBJECT_MAP[obj_name]
    if obj is None:
        return False

    if obj.get_mode() != MODE_IDLE:
        return False

    obj_q = None
    for obj_n in get_position_objs(obj.get_pos()):
        if obj_n.get_type() in [OBJ_TYPE_1, OBJ_TYPE_2, OBJ_TYPE_3]:
            obj_q = obj_n
            break

    if obj_q is None:
        print("no object")
        return False

    # stall timers
    cur_time = timer.get_current_time()
    tm = timer.get_timer(obj_q.get_name(), "grow")
    if tm is not None:
        t_st = tm.set_time - (obj_q.get_time2grow() -
                              obj_q.get_time_progressed())
        t_passed = cur_time - t_st
        obj_q.set_time_progressed(obj_q.get_time_progressed() + t_passed)

    already_cleaned = False
    tm = timer.get_timer(obj_q.get_name(), "clean")
    if tm is not None:
        t_st = tm.set_time - (obj_q.get_time2clean() -
                              obj_q.get_time_cleaned())
        t_passed = cur_time - t_st
        obj_q.set_time_cleaned(obj_q.get_time_cleaned() + t_passed)
        if obj_q.get_time_cleaned() >= obj_q.get_time2clean():
            done_cleaning(obj_q, data_changed)
            already_cleaned = True
    timer.remove_timer(obj_q.get_name())

    if not already_cleaned:
        head = obj_q.get_pos()
        for obj_a in get_position_objs(head):
            if obj_a.get_type() == TYPE_AGENT:
                if obj_a.get_mode() == MODE_IN_ACTION:
                    obj_a.set_mode(MODE_IDLE)
                    data_changed.append(obj_a.get_name())

        obj.add_item(obj_q)
        WORLD_OBJECTS.remove(obj_q.get_name())
        data_changed.append(obj_q.get_name())
        data_changed.append(obj_name)

    return True


def grow_process(timers, t_excute, obj_name_p, changed_items):
    added_obj = []
    obj_p = OBJECT_MAP[obj_name_p]
    if grow_contamination(obj_p, added_obj):
        # add new timers
        if obj_p.get_time2grow() > 0:
            t_next = t_excute + (obj_p.get_time2grow() -
                                 obj_p.get_time_progressed())
            te = CTimingEvent(t_next, obj_name_p,
                              tag="grow",
                              callback=(lambda t_ex, l_name, changed:
                                        grow_process(timers, t_ex,
                                                     l_name, changed)))
            timers.add_timer(te)
            if changed_items is not None:
                changed_items.add(obj_name_p)

        for obj_n_name in added_obj:
            obj_new = OBJECT_MAP[obj_n_name]
            if obj_new.get_time2grow() > 0:
                t_next = t_excute + (obj_new.get_time2grow() -
                                     obj_new.get_time_progressed())
                te = CTimingEvent(t_next, obj_n_name,
                                  tag="grow",
                                  callback=(lambda t_ex, l_name, changed:
                                            grow_process(timers, t_ex,
                                                         l_name, changed)))
                timers.add_timer(te)
                if changed_items is not None:
                    changed_items.add(obj_n_name)


def drop(obj_name, timers, data_changed):
    obj = OBJECT_MAP[obj_name]
    if obj is None:
        return False

    if len(obj.query_items()) == 0:
        return False

    if obj.get_mode() != MODE_IDLE:
        return False

    overlap = False
    for obj_n in get_position_objs(obj.get_pos()):
        if obj_n.get_type() in [OBJ_TYPE_1, OBJ_TYPE_2, OBJ_TYPE_3]:
            overlap = True
            break

    if overlap:
        print("the cell is not empty")
        return False

    itm = obj.pop_item()
    itm.set_pos(obj.get_pos())
    WORLD_OBJECTS.add(itm.get_name())
    data_changed.append(itm.get_name())
    data_changed.append(obj_name)
    if itm.get_time2grow() > 0:
        growing_event = CTimingEvent(name=itm.get_name(), tag="grow")
        growing_event.set_time = (timers.get_current_time() +
                                  itm.get_time2grow() -
                                  itm.get_time_progressed())
        growing_event.callback = (lambda t_ex, l_name, changed:
                                  grow_process(timers, t_ex, l_name, changed))
        timers.add_timer(growing_event)

    return True


def grow_contamination(obj, obj_added):
    if obj is None:
        return False

    mty_coords = []
    for coords in get_nearby_coords(obj):
        objs = get_position_objs(coords)
        empty_cell = True
        for obj_n in objs:
            if obj_n.get_type() in [OBJ_TYPE_1, OBJ_TYPE_2, OBJ_TYPE_3]:
                empty_cell = False
                continue
        if empty_cell:
            mty_coords.append(coords)

    obj.set_time_progressed(0)
    # add new contamination
    obj_type = obj.get_type()

    for idx in range(len(mty_coords)):
        obj_idx = NUM_OBJS[obj_type]
        NUM_OBJS[obj_type] = NUM_OBJS[obj_type] + 1
        new_obj = CContaminationType1(obj_type + str(obj_idx), obj_type)
        obj.copy_data_to(new_obj)
        new_obj.set_pos(mty_coords[idx])
        new_obj.set_text(obj_type[0].upper() + str(obj_idx))

        OBJECT_MAP[new_obj.get_name()] = new_obj
        WORLD_OBJECTS.add(new_obj.get_name())
        obj_added.append(new_obj.get_name())

    return True


def can_clean(obj):
    if obj is None:
        return False

    n_need = obj.get_num_agents()
    # get all agents in this position
    head = obj.get_pos()
    count = 0
    for obj_q in get_position_objs(head):
        if obj_q.get_type() == TYPE_AGENT:
            if obj_q.get_mode() == MODE_IN_ACTION:
                count += 1

    if count < n_need:
        # need more agents
        return False

    return True


def done_cleaning(obj, changed_items):
    if obj is None:
        return False

    obj.set_time_cleaned(obj.get_time2clean())
    obj_nm = obj.get_name()

    head = obj.get_pos()
    for obj_q in get_position_objs(head):
        if obj_q.get_name() in [AGENT_NAME_1, AGENT_NAME_2]:
            if obj_q.get_mode() == MODE_IN_ACTION:
                obj_q.set_mode(MODE_IDLE)
                changed_items.append(obj_q.get_name())

    WORLD_SCORE[0] += obj.get_num_agents()

    WORLD_OBJECTS.remove(obj_nm)
    OBJECT_MAP.pop(obj_nm)
    changed_items.append(obj_nm)

    return True
