from enum import Enum
from typing import Mapping, Any, Sequence
from web_experiment.define import PageKey, EDomainType
from web_experiment.exp_common.page_base import CanvasPageBase
import web_experiment.exp_common.page_exp1_common as pgc
from web_experiment.exp_common.page_boxpushv2_base import BoxPushV2UserRandom
import web_experiment.exp_common.page_tutorial as pgt
from ai_coach_domain.box_push.maps import TUTORIAL_MAP
from ai_coach_domain.box_push_v2.maps import MAP_CLEANUP, MAP_MOVERS

from ai_coach_domain.rescue.maps import MAP_RESCUE
from web_experiment.exp_common.page_rescue_game_base import RescueGameUserRandom

SESSION_TITLE = {
    PageKey.DataCol_A0: 'A0',
    PageKey.DataCol_A1: 'A1',
    PageKey.DataCol_A2: 'A2',
    PageKey.DataCol_A3: 'A3',
    PageKey.DataCol_B0: 'B0',
    PageKey.DataCol_B1: 'B1',
    PageKey.DataCol_B2: 'B2',
    PageKey.DataCol_B3: 'B3',
    PageKey.DataCol_C0: 'C0',
    PageKey.DataCol_C1: 'C1',
    PageKey.DataCol_C2: 'C2',
    PageKey.DataCol_C3: 'C3',
    PageKey.DataCol_T1: 'Interactive Tutorial',
    PageKey.DataCol_T2: 'Interactive Tutorial',
    PageKey.DataCol_T3: 'Interactive Tutorial',
}


class SocketType(Enum):
  '''
  The socket name should be unique across all experiments.
  (i.e. if DataCollection experiment and Intervention experiment have a socket,
  whose name is the same, socketio cannot distinguish the event handlers to use
  '''
  DataCol_movers_practice = 0
  DataCol_movers_test = 1
  DataCol_cleanup_practice = 2
  DataCol_cleanup_test = 3
  DataCol_movers_tutorial = 4
  DataCol_cleanup_tutorial = 5
  DataCol_rescue_practice = 6
  DataCol_rescue_test = 7
  DataCol_rescue_tutorial = 8


def get_socket_name(page_key):
  socket_type = None
  if page_key == PageKey.DataCol_A0:
    socket_type = SocketType.DataCol_movers_practice
  elif page_key in [PageKey.DataCol_A1, PageKey.DataCol_A2, PageKey.DataCol_A3]:
    socket_type = SocketType.DataCol_movers_test

  elif page_key == PageKey.DataCol_B0:
    socket_type = SocketType.DataCol_cleanup_practice
  elif page_key in [PageKey.DataCol_B1, PageKey.DataCol_B2, PageKey.DataCol_B3]:
    socket_type = SocketType.DataCol_cleanup_test

  elif page_key == PageKey.DataCol_C0:
    socket_type = SocketType.DataCol_rescue_practice
  elif page_key in [PageKey.DataCol_C1, PageKey.DataCol_C2, PageKey.DataCol_C3]:
    socket_type = SocketType.DataCol_rescue_test

  elif page_key == PageKey.DataCol_T1:
    socket_type = SocketType.DataCol_movers_tutorial
  elif page_key == PageKey.DataCol_T2:
    socket_type = SocketType.DataCol_cleanup_tutorial
  elif page_key == PageKey.DataCol_T3:
    socket_type = SocketType.DataCol_rescue_tutorial

  return socket_type.name if socket_type is not None else None


PAGE_LIST_MOVERS_FULL_OBS = [
    pgc.CanvasPageStart(EDomainType.Movers),
    pgc.CanvasPageWarning(EDomainType.Movers),
    BoxPushV2UserRandom(EDomainType.Movers, MAP_MOVERS, False),
    pgc.CanvasPageEnd(EDomainType.Movers)
]
PAGE_LIST_MOVERS = [
    pgc.CanvasPageStart(EDomainType.Movers),
    pgc.CanvasPageWarning(EDomainType.Movers),
    BoxPushV2UserRandom(EDomainType.Movers, MAP_MOVERS, True),
    pgc.CanvasPageEnd(EDomainType.Movers)
]
PAGE_LIST_CLEANUP_FULL_OBS = [
    pgc.CanvasPageStart(EDomainType.Cleanup),
    pgc.CanvasPageWarning(EDomainType.Cleanup),
    BoxPushV2UserRandom(EDomainType.Cleanup, MAP_CLEANUP, False),
    pgc.CanvasPageEnd(EDomainType.Cleanup)
]
PAGE_LIST_CLEANUP = [
    pgc.CanvasPageStart(EDomainType.Cleanup),
    pgc.CanvasPageWarning(EDomainType.Cleanup),
    BoxPushV2UserRandom(EDomainType.Cleanup, MAP_CLEANUP, True),
    pgc.CanvasPageEnd(EDomainType.Cleanup)
]

PAGE_LIST_MOVERS_TUTORIAL = [
    pgt.CanvasPageTutorialStart(EDomainType.Movers),
    pgt.CanvasPageInstruction(EDomainType.Movers),
    pgt.CanvasPageTutorialGameStart(EDomainType.Movers),
    pgt.CanvasPageJoystick(EDomainType.Movers, TUTORIAL_MAP),
    pgt.CanvasPageInvalidAction(EDomainType.Movers, TUTORIAL_MAP),
    pgt.CanvasPageOnlyHuman(EDomainType.Movers, TUTORIAL_MAP),
    pgt.CanvasPageGoToTarget(EDomainType.Movers, TUTORIAL_MAP),
    pgt.CanvasPagePickUpTarget(EDomainType.Movers, TUTORIAL_MAP),
    pgt.CanvasPageGoToGoal(EDomainType.Movers, TUTORIAL_MAP),
    pgt.CanvasPageScore(EDomainType.Movers, TUTORIAL_MAP),
    pgt.CanvasPageTrapped(EDomainType.Movers, TUTORIAL_MAP),
    pgt.CanvasPageTargetHint(EDomainType.Movers, TUTORIAL_MAP),
    pgt.CanvasPageTargetNoHint(EDomainType.Movers, TUTORIAL_MAP),
    pgt.CanvasPageLatent(EDomainType.Movers, TUTORIAL_MAP),
    pgt.CanvasPageSelResult(EDomainType.Movers, TUTORIAL_MAP, False),
    pgt.CanvasPageSelPrompt(EDomainType.Movers, TUTORIAL_MAP),
    pgt.CanvasPageSelResult(EDomainType.Movers, TUTORIAL_MAP, True),
    pgt.CanvasPageMiniGame(EDomainType.Movers, TUTORIAL_MAP)
]
PAGE_LIST_CLEANUP_TUTORIAL = [
    pgt.CanvasPageTutorialStart(EDomainType.Cleanup),
    pgt.CanvasPageInstruction(EDomainType.Cleanup),
    pgt.CanvasPageTutorialGameStart(EDomainType.Cleanup),
    pgt.CanvasPageJoystickShort(EDomainType.Cleanup, TUTORIAL_MAP),
    pgt.CanvasPageOnlyHuman(EDomainType.Cleanup, TUTORIAL_MAP),
    pgt.CanvasPageGoToTarget(EDomainType.Cleanup, TUTORIAL_MAP),
    pgt.CanvasPagePickUpTarget(EDomainType.Cleanup, TUTORIAL_MAP),
    pgt.CanvasPageGoToGoal(EDomainType.Cleanup, TUTORIAL_MAP),
    pgt.CanvasPageScore(EDomainType.Cleanup, TUTORIAL_MAP),
    pgt.CanvasPageTrapped(EDomainType.Cleanup, TUTORIAL_MAP),
    pgt.CanvasPageTargetHint(EDomainType.Cleanup, TUTORIAL_MAP),
    pgt.CanvasPageTargetNoHint(EDomainType.Cleanup, TUTORIAL_MAP),
    pgt.CanvasPageLatent(EDomainType.Cleanup, TUTORIAL_MAP),
    pgt.CanvasPageSelResult(EDomainType.Cleanup, TUTORIAL_MAP, False),
    pgt.CanvasPageSelPrompt(EDomainType.Cleanup, TUTORIAL_MAP),
    pgt.CanvasPageSelResult(EDomainType.Cleanup, TUTORIAL_MAP, True),
    pgt.CanvasPageMiniGame(EDomainType.Cleanup, TUTORIAL_MAP)
]

PAGE_LIST_RESCUE_FULL_OBS = [
    pgc.CanvasPageStart(EDomainType.Rescue),
    pgc.CanvasPageWarning(EDomainType.Rescue),
    RescueGameUserRandom(MAP_RESCUE, False),
    pgc.CanvasPageEnd(EDomainType.Rescue)
]

PAGE_LIST_RESCUE = [
    pgc.CanvasPageStart(EDomainType.Rescue),
    pgc.CanvasPageWarning(EDomainType.Rescue),
    RescueGameUserRandom(MAP_RESCUE, True),
    pgc.CanvasPageEnd(EDomainType.Rescue)
]

GAMEPAGES = {
    SocketType.DataCol_movers_practice: PAGE_LIST_MOVERS,
    SocketType.DataCol_movers_test: PAGE_LIST_MOVERS,
    SocketType.DataCol_cleanup_practice: PAGE_LIST_CLEANUP,
    SocketType.DataCol_cleanup_test: PAGE_LIST_CLEANUP,
    SocketType.DataCol_movers_tutorial: PAGE_LIST_MOVERS_TUTORIAL,
    SocketType.DataCol_cleanup_tutorial: PAGE_LIST_CLEANUP_TUTORIAL,
    SocketType.DataCol_rescue_practice: PAGE_LIST_RESCUE_FULL_OBS,
    SocketType.DataCol_rescue_test: PAGE_LIST_RESCUE,
    SocketType.DataCol_rescue_tutorial: PAGE_LIST_RESCUE,
}  # type: Mapping[Any, Sequence[CanvasPageBase]]
