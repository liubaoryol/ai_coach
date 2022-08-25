from enum import Enum


class EDomainType(Enum):
  Movers = 0,
  Cleanup = 1,


class EMode(Enum):
  NONE = 0,
  Replay = 1,
  Predicted = 2,
  Collected = 3


GROUP_A = 'A'
GROUP_B = 'B'
GROUP_C = 'C'
GROUP_D = 'D'

GROUP_NAMES = {
    GROUP_A: "Group A",
    GROUP_B: "Group B",
    GROUP_C: "Group C",
    GROUP_D: "Group D",
}
