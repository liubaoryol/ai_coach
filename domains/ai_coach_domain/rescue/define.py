from typing import Union, Sequence, Tuple
from enum import Enum
from dataclasses import dataclass
from ai_coach_core.utils.mdp_utils import ActionSpace

T_RouteId = int
T_PlaceName = str


class E_EventType(Enum):
  Option0 = 0
  Option1 = 1
  Option2 = 2
  Option3 = 3
  Stay = 4
  Set_Latent = 100


AGENT_ACTIONSPACE = ActionSpace([E_EventType(idx) for idx in range(5)])


class E_Type(Enum):
  Place = 0
  Route = 1


T_Connections = Sequence[Tuple[E_Type, int]]


@dataclass
class Location:
  type: E_Type
  id: Union[T_PlaceName, T_RouteId]
  index: int = 0

  def __repr__(self) -> str:
    return f"{self.type.name}, {self.id}, {self.index}"

  @classmethod
  def from_str(cls, str_loc: str):
    list_loc = str_loc.split(", ")
    e_type = E_Type[list_loc[0]]
    id_place = list_loc[1]
    if e_type == E_Type.Route:
      id_place = int(id_place)
    index = int(list_loc[2])

    return Location(e_type, id_place, index)


@dataclass
class Route:
  start: T_PlaceName
  end: T_PlaceName
  length: int


@dataclass
class Work:
  helps: int
  workload: int


@dataclass
class Place:
  coord: Tuple[float, float]
  connections: T_Connections
