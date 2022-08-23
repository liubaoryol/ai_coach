from enum import Enum


class EDomainType(Enum):
  Movers = 0,
  Cleanup = 1,


class EMode(Enum):
  NONE = 0,
  Replay = 1,
  Predicted = 2,
  Collected = 3
