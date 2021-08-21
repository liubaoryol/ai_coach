from enum import Enum


class ToolNames(Enum):
  SCALPEL_P = 0  # prepared scalpel, initially on the table
  SUTURE_P = 1  # prepared suture, initially on the table
  SCALPEL_S = 2  # spare scalpel, initially in the storage
  SUTURE_S = 3  # spare suture, initially in the cabinet


class ToolLoc(Enum):
  '''enum of tool locations'''
  STORAGE = 0
  CABINET = 1
  CN = 2
  SN = 3
  AS = 4
  FLOOR = 5


class StatePatient(Enum):
  '''enum of states associated with patient status'''
  NO_INCISION = 0
  INCISION = 1


def generate_CNPos_states(num_x_grid, num_y_grid):
  '''method to generate state space of circulating nurse position'''
  coords = []
  for i in range(num_x_grid):
    for j in range(num_y_grid):
      coords.append((i, j))
  return coords


class StateAsked(Enum):
  '''enum of states associated with whether SN asked a tool'''
  NOT_ASKED = 0
  ASKED = 1


class LatentState(Enum):
  '''enum of latent states'''
  SCALPEL = 0
  SUTURE = 1


class ActionCN(Enum):
  '''enum of actions associated with circulating nurse'''
  STAY = 0
  MOVE_UP = 1
  MOVE_DOWN = 2
  MOVE_LEFT = 3
  MOVE_RIGHT = 4
  PICKUP = 5
  HANDOVER = 6  # assume CN always hand over tools to SN


class ActionSN(Enum):
  '''enum of actions associated with scrub nurse'''
  STAY = 0
  HO_SCALPEL = 1  # assume SN always hand over tools to AS
  HO_SUTURE = 2  # assume SN always hand over tools to AS
  ASKTOOL = 3
  SCALPEL_RELATED = 4
  SUTURE_RELATED = 5


class ActionAS(Enum):
  '''enum of actions associated with attending surgeon'''
  STAY = 0
  HO_SCALPEL = 1  # assume AS always hand over tools to SN
  USE_SCALPEL = 2
  USE_SUTURE = 3
