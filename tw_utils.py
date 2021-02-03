import numpy as np
import random
from tw_objects import *
from xml.dom import minidom

ITEM_KEY_TYPE = "type"
ITEM_KEY_X = "x"
ITEM_KEY_Y = "y"
ITEM_KEY_NUM_AGENT = "num_of_agents"
ITEM_KEY_TIME_CLEAN = "time_to_clean"
ITEM_KEY_TIME_GROW = "time_to_grow"


def read_xml(xml_file):
    xml_doc = minidom.parse(xml_file)

    obj_property = {}
    for target_type in xml_doc.getElementsByTagName("type"):
        type_name = target_type.getAttribute("name")
        obj_property[type_name] = (
            int(target_type.getAttribute(ITEM_KEY_NUM_AGENT)),
            int(target_type.getAttribute(ITEM_KEY_TIME_CLEAN)),
            int(target_type.getAttribute(ITEM_KEY_TIME_GROW)))

    contamination = []
    for contam in xml_doc.getElementsByTagName("position"):
        contam_dict = {}
        contam_dict[ITEM_KEY_TYPE] = contam.getAttribute(ITEM_KEY_TYPE)
        contam_dict[ITEM_KEY_X] = int(contam.getAttribute(ITEM_KEY_X))
        contam_dict[ITEM_KEY_Y] = int(contam.getAttribute(ITEM_KEY_Y))
        contamination.append(contam_dict)
    xml_doc.unlink()

    return contamination, obj_property


def generate_random_trash(num_trash_dict):
    num_trash = 0
    for key in num_trash_dict:
        num_trash += num_trash_dict[key]

    trash_list = []
    samples = random.sample(range(NUM_GRID_X * NUM_GRID_Y),
                            num_trash)
    for key in num_trash_dict:
        for dummy in range(num_trash_dict[key]):
            trash_dict = {}
            trash_dict[ITEM_KEY_TYPE] = key
            grid_idx = samples.pop()
            trash_dict[ITEM_KEY_X] = grid_idx % NUM_GRID_X
            trash_dict[ITEM_KEY_Y] = grid_idx // NUM_GRID_X

            trash_list.append(trash_dict)

    return trash_list


def init_data_with_file(xml_file):
    contam_data, obj_property = read_xml(xml_file)
    for key in obj_property:
        OBJ_PROP[key] = (max(1, obj_property[key][0]),
                         max(0, obj_property[key][1]),
                         max(0, obj_property[key][2]))
    return init_data(contam_data)


def init_data(targets):
    # {object_type, x, y, num_of_agents, time_to_clean, time_to_grow}
    NUM_OBJS[OBJ_TYPE_1] = 0
    NUM_OBJS[OBJ_TYPE_2] = 0
    NUM_OBJS[OBJ_TYPE_3] = 0

    WORLD_OBJECTS.clear()
    OBJECT_MAP.clear()

    for item in targets:
        obj_type = item[ITEM_KEY_TYPE]
        obj_idx = NUM_OBJS[obj_type]
        NUM_OBJS[obj_type] = NUM_OBJS[obj_type] + 1
        name = obj_type + str(obj_idx)
        contam_obj = CContaminationType1(name, obj_type)

        contam_obj.set_color(COLOR_OBJS[obj_type])
        contam_obj.set_text(obj_type[0].upper() + str(obj_idx))
        contam_obj.set_pos((item[ITEM_KEY_X], item[ITEM_KEY_Y]))

        contam_obj.set_num_agents(OBJ_PROP[obj_type][0])
        contam_obj.set_time2clean(OBJ_PROP[obj_type][1])
        contam_obj.set_time2grow(OBJ_PROP[obj_type][2])

        OBJECT_MAP[name] = contam_obj
        WORLD_OBJECTS.add(name)

    WORLD_SCORE[0] = 0


def get_random_agent_position():
    agent_pos = (random.randrange(0, NUM_GRID_X),
                 random.randrange(0, NUM_GRID_Y))

    return agent_pos


def add_agent(agent1_pos=(0, NUM_GRID_Y), agent2_pos=(NUM_GRID_X, 0)):
    agent_1 = CCleanUpAgent(AGENT_NAME_1)
    agent_2 = CCleanUpAgent(AGENT_NAME_2)
    OBJECT_MAP[AGENT_NAME_1] = agent_1
    OBJECT_MAP[AGENT_NAME_2] = agent_2
    WORLD_OBJECTS.add(AGENT_NAME_1)
    WORLD_OBJECTS.add(AGENT_NAME_2)

    agent_1 = OBJECT_MAP.get(AGENT_NAME_1)
    if agent_1:
        agent_1.set_pos(agent1_pos)
        agent_1.set_text("A1")

    agent_2 = OBJECT_MAP.get(AGENT_NAME_2)
    if agent_2:
        agent_2.set_pos(agent2_pos)
        agent_2.set_text("A2")


def is_boundary(pos):
    x_pos, y_pos = pos
    return x_pos < 0 or x_pos >= NUM_GRID_X or y_pos < 0 or y_pos >= NUM_GRID_Y


def get_position_objs(pos):
    x_pos, y_pos = pos
    objs = []
    for obj_name in WORLD_OBJECTS:
        obj = OBJECT_MAP[obj_name]
        obj_x, obj_y = obj.get_pos()
        obj_w, obj_h = obj.get_size()
        if (x_pos >= obj_x and x_pos < obj_x + obj_w and
                y_pos >= obj_y and y_pos < obj_y + obj_h):
            objs.append(obj)

    return objs


def get_nearby_coords(obj):
    x_pos, y_pos = obj.get_pos()
    x_sz, y_sz = obj.get_size()

    nearby_coords = []
    for y_q in range(y_pos, y_pos + y_sz):
        pos_1 = (x_pos - 1, y_q)
        pos_2 = (x_pos + x_sz, y_q)
        if not is_boundary(pos_1):
            nearby_coords.append(pos_1)
        if not is_boundary(pos_2):
            nearby_coords.append(pos_2)

    for x_q in range(x_pos, x_pos + x_sz):
        pos_1 = (x_q, y_pos - 1)
        pos_2 = (x_q, y_pos + y_sz)
        if not is_boundary(pos_1):
            nearby_coords.append(pos_1)
        if not is_boundary(pos_2):
            nearby_coords.append(pos_2)

    return nearby_coords


def get_nearby(obj):
    x_pos, y_pos = obj.get_pos()
    x_sz, y_sz = obj.get_size()
    nearby_coords = get_nearby_coords(obj)

    near_objs = set()

    for obj_name in WORLD_OBJECTS:
        obj_q = OBJECT_MAP[obj_name]
        obj_x, obj_y = obj_q.get_pos()
        obj_w, obj_h = obj_q.get_size()

        def is_in(x, y):
            return (x >= obj_x and x < obj_x + obj_w and
                    y >= obj_y and y < obj_y + obj_h)

        # nearby
        for coords in nearby_coords:
            if is_in(coords[0], coords[1]):
                near_objs.add(obj_q)
                break

    return near_objs


def compute_gcd(x, y):
    while(y):
        x, y = y, x % y
    return x


def compute_lcm(x, y):
    lcm = (x * y) // compute_gcd(x, y)
    return lcm


def radial_basis_function(x, sigma):
    return np.exp(- abs(x) / sigma)


def value_normal_trash(a_pos, a_mod, trash_x_y, trash_prop, time):
    a_x, a_y = a_pos
    a_m = a_mod
    t_x, t_y = trash_x_y
    num_a, time_c, time_g = trash_prop

    dist = abs(t_x - a_x) + abs(t_y - a_y)
    DIST_SIGMA = (NUM_GRID_X + NUM_GRID_Y - 2) / 3
    distance_weight = radial_basis_function(dist, DIST_SIGMA)
    value = distance_weight

    if time_g > 0:
        time_left = time_g - time % time_g
        step_to_reach = dist
        if time_c < 2:
            step_to_reach += 1
        else:
            step_to_reach += time_c

        tau = 0
        if time_left >= step_to_reach:
            tau = time_left - step_to_reach
        else:
            tau = time_g

        TIME_SIGMA = time_g / 3
        urgency_weight = radial_basis_function(tau, TIME_SIGMA)
        value = value * urgency_weight

    return value


def value_big_trash(a1_pos, a1_mod, a2_pos, a2_mod,
                    trash_x_y, trash_prop, time):
    a1_x, a1_y = a1_pos
    a1_m = a1_mod
    a2_x, a2_y = a2_pos
    a2_m = a2_mod
    t_x, t_y = trash_x_y
    num_a, time_c, time_g = trash_prop

    dist1 = abs(t_x - a1_x) + abs(t_y - a1_y)
    dist2 = abs(t_x - a2_x) + abs(t_y - a2_y)
    DIST_SIGMA = (NUM_GRID_X + NUM_GRID_Y - 2) / 3
    distance_weight1 = radial_basis_function(dist1, DIST_SIGMA)
    distance_weight2 = radial_basis_function(dist2, DIST_SIGMA)

    value = num_a * distance_weight1 * distance_weight2

    if time_g > 0:
        time_left = time_g - time % time_g
        step_to_reach = max(dist1, dist2)
        if time_c < 2:
            step_to_reach += 1
        else:
            step_to_reach += time_c

        tau = 0
        if time_left >= step_to_reach:
            tau = time_left - step_to_reach
        else:
            tau = time_g

        TIME_SIGMA = time_g / 3
        urgency_weight = radial_basis_function(tau, TIME_SIGMA)
        value = value * urgency_weight

    return value


def value_by_agent_distancing(a1_pos, a1_mod, a2_pos, a2_mod):
    a1_x, a1_y = a1_pos
    a1_m = a1_mod
    a2_x, a2_y = a2_pos
    a2_m = a2_mod

    DIST_SIGMA = (NUM_GRID_X + NUM_GRID_Y - 2) / 3

    return 1 - radial_basis_function(abs(a1_x - a2_x) + abs(a1_y - a2_y),
                                     DIST_SIGMA)


def bound_pos(pos):
    x, y = pos
    if x < 0:
        x = 0
    if x >= NUM_GRID_X:
        x = NUM_GRID_X - 1
    if y < 0:
        y = 0
    if y >= NUM_GRID_Y:
        y = NUM_GRID_Y - 1
    return (x, y)
