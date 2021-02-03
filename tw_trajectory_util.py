import os
import glob
import numpy as np
from tw_defines import *
from tw_utils import *


def read_sample(file_name, num_coord=-1):
    traj = []
    trash_props = {}
    with open(file_name, newline='') as txtfile:
        header_line = txtfile.readline().rstrip()
        header_data = [int(elem) for elem in header_line.split(", ")]
        # num_agents, time_to_clean, time_to_grow
        trash_props[OBJ_TYPE_1] = tuple(header_data[0:3])
        trash_props[OBJ_TYPE_2] = tuple(header_data[3:6])
        trash_props[OBJ_TYPE_3] = tuple(header_data[6:9])
        coords = txtfile.readlines()
        count = 0
        for row in coords:
            if num_coord >= 0 and count >= num_coord:
                break
            traj.append([int(elem) if elem.isdigit() else elem
                        for elem in row.rstrip().split(", ")])
            count += 1

    return traj, trash_props


def check_overlap(pos, trash_1, trash_2, trash_3):
    for tr_pos in trash_1:
        if pos == tr_pos:
            return True

    for tr_pos in trash_2:
        if pos == tr_pos:
            return True

    for tr_pos in trash_3:
        if pos == tr_pos:
            return True

    return False


def parse_coord(coord):
    agent1_x, agent1_y = coord[0], coord[1]
    agent2_x, agent2_y = coord[2], coord[3]

    agent1_mode = coord[4]
    agent2_mode = coord[5]
    cur_idx = 6
    trash_1 = []
    trash_2 = []
    trash_3 = []
    for dummy_cnt in range(coord[cur_idx]):
        trash_1.append((coord[cur_idx + 1], coord[cur_idx + 2]))
        cur_idx += 2
    cur_idx += 1
    for dummy_cnt in range(coord[cur_idx]):
        trash_2.append((coord[cur_idx + 1], coord[cur_idx + 2]))
        cur_idx += 2
    cur_idx += 1
    for dummy_cnt in range(coord[cur_idx]):
        trash_3.append((coord[cur_idx + 1], coord[cur_idx + 2]))
        cur_idx += 2
    cur_idx += 1
    timestamp = coord[cur_idx]
    cur_idx += 1
    a1_action = coord[cur_idx]
    cur_idx += 1
    a2_action = coord[cur_idx]

    a1_data = (agent1_x, agent1_y, agent1_mode, a1_action)
    a2_data = (agent2_x, agent2_y, agent2_mode, a2_action)

    return a1_data, a2_data, trash_1, trash_2, trash_3, timestamp
