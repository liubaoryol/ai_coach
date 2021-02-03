from tw_defines import *

WORLD_OBJECTS = set()
OBJECT_MAP = {}
WORLD_SCORE = [0]

NUM_OBJS = {OBJ_TYPE_1: 0, OBJ_TYPE_2: 0, OBJ_TYPE_3: 0}
# prop: tuple of (num_agents, time_2_clean, time_2_grow)
# note: time is in seconds
OBJ_PROP = {OBJ_TYPE_1: None, OBJ_TYPE_2: None, OBJ_TYPE_3: None}
COLOR_OBJS = {OBJ_TYPE_1: "yellow", OBJ_TYPE_2: "grey", OBJ_TYPE_3: "brown"}
