from enum import Enum
from typing import Mapping, Any, Sequence
from web_experiment.define import PageKey, EDomainType
from web_experiment.exp_common.page_base import CanvasPageBase
import web_experiment.exp_common.page_exp1_common as pgc
from web_experiment.exp_common.page_boxpushv2_base import BoxPushV2UserRandom
import web_experiment.exp_common.page_tutorial as pgt
import web_experiment.exp_common.page_tutorial_rescue as pgr
from ai_coach_domain.box_push_v2.maps import MAP_CLEANUP, MAP_MOVERS

from ai_coach_domain.rescue.maps import MAP_RESCUE_2
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
    pgt.CanvasPageJoystick(EDomainType.Movers, MAP_MOVERS),
    pgt.CanvasPageInvalidAction(EDomainType.Movers, MAP_MOVERS),
    pgt.CanvasPageOnlyHuman(EDomainType.Movers, MAP_MOVERS),
    pgt.CanvasPageGoToTarget(EDomainType.Movers, MAP_MOVERS),
    pgt.CanvasPagePickUpTargetAttempt(EDomainType.Movers, MAP_MOVERS),
    pgt.CanvasPagePickUpTarget(EDomainType.Movers, MAP_MOVERS),
    pgt.CanvasPageGoToGoal(EDomainType.Movers, MAP_MOVERS),
    pgt.CanvasPageRespawn(EDomainType.Movers, MAP_MOVERS),
    pgt.CanvasPageScore(EDomainType.Movers, MAP_MOVERS),
    pgt.CanvasPagePartialObs(EDomainType.Movers, MAP_MOVERS),
    pgt.CanvasPageTarget(EDomainType.Movers, MAP_MOVERS),
    pgt.CanvasPageLatent(EDomainType.Movers, MAP_MOVERS),
    pgt.CanvasPageSelResult(EDomainType.Movers, MAP_MOVERS, False),
    pgt.CanvasPageSelPrompt(EDomainType.Movers, MAP_MOVERS),
    pgt.CanvasPageSelResult(EDomainType.Movers, MAP_MOVERS, True),
    pgt.CanvasPageMiniGame(EDomainType.Movers, MAP_MOVERS)
]
PAGE_LIST_CLEANUP_TUTORIAL = [
    pgt.CanvasPageTutorialStart(EDomainType.Cleanup),
    pgt.CanvasPageInstruction(EDomainType.Cleanup),
    pgt.CanvasPageTutorialGameStart(EDomainType.Cleanup),
    pgt.CanvasPageJoystickShort(EDomainType.Cleanup, MAP_CLEANUP),
    pgt.CanvasPageOnlyHuman(EDomainType.Cleanup, MAP_CLEANUP),
    pgt.CanvasPageGoToTarget(EDomainType.Cleanup, MAP_CLEANUP),
    pgt.CanvasPagePickUpTargetAttempt(EDomainType.Cleanup, MAP_CLEANUP),
    pgt.CanvasPageGoToGoal(EDomainType.Cleanup, MAP_CLEANUP),
    pgt.CanvasPageRespawn(EDomainType.Cleanup, MAP_CLEANUP),
    pgt.CanvasPageScore(EDomainType.Cleanup, MAP_CLEANUP),
    pgt.CanvasPagePartialObs(EDomainType.Cleanup, MAP_CLEANUP),
    pgt.CanvasPageTarget(EDomainType.Cleanup, MAP_CLEANUP),
    pgt.CanvasPageLatent(EDomainType.Cleanup, MAP_CLEANUP),
    pgt.CanvasPageSelResult(EDomainType.Cleanup, MAP_CLEANUP, False),
    pgt.CanvasPageSelPrompt(EDomainType.Cleanup, MAP_CLEANUP),
    pgt.CanvasPageSelResult(EDomainType.Cleanup, MAP_CLEANUP, True),
    pgt.CanvasPageMiniGame(EDomainType.Cleanup, MAP_CLEANUP)
]

PAGE_LIST_RESCUE_FULL_OBS = [
    pgc.CanvasPageStart(EDomainType.Rescue),
    pgc.CanvasPageWarning(EDomainType.Rescue),
    RescueGameUserRandom(MAP_RESCUE_2, False),
    pgc.CanvasPageEnd(EDomainType.Rescue)
]

PAGE_LIST_RESCUE = [
    pgc.CanvasPageStart(EDomainType.Rescue),
    pgc.CanvasPageWarning(EDomainType.Rescue),
    RescueGameUserRandom(MAP_RESCUE_2, True),
    pgc.CanvasPageEnd(EDomainType.Rescue)
]

PAGE_LIST_RESCUE_TUTORIAL = [
    pgt.CanvasPageTutorialStart(EDomainType.Rescue),
    pgt.CanvasPageInstruction(EDomainType.Rescue),
    pgt.CanvasPageTutorialGameStart(EDomainType.Rescue),
    pgr.RescueTutorialActions(MAP_RESCUE_2),
    pgr.RescueTutorialOverallGoal(MAP_RESCUE_2),
    pgr.RescueTutorialOnlyHuman(MAP_RESCUE_2),
    pgr.RescueTutorialSimpleTarget(MAP_RESCUE_2),
    pgr.RescueTutorialResolvedAlone(MAP_RESCUE_2),
    pgr.RescueTutorialComplexTarget(MAP_RESCUE_2),
    pgr.RescueTutorialComplexTargetTogether(MAP_RESCUE_2),
    pgr.RescueTutorialResolvedTogether(MAP_RESCUE_2),
    pgr.RescueTutorialScore(MAP_RESCUE_2),
    pgr.RescueTutorialPartialObs(MAP_RESCUE_2),
    pgr.RescueTutorialDestination(MAP_RESCUE_2),
    pgr.RescueTutorialLatent(MAP_RESCUE_2),
    pgr.RescueTutorialSelResult(MAP_RESCUE_2),
    pgr.RescueTutorialMiniGame(MAP_RESCUE_2)
]

GAMEPAGES = {
    SocketType.DataCol_movers_practice: PAGE_LIST_MOVERS_FULL_OBS,
    SocketType.DataCol_movers_test: PAGE_LIST_MOVERS,
    SocketType.DataCol_cleanup_practice: PAGE_LIST_CLEANUP_FULL_OBS,
    SocketType.DataCol_cleanup_test: PAGE_LIST_CLEANUP,
    SocketType.DataCol_movers_tutorial: PAGE_LIST_MOVERS_TUTORIAL,
    SocketType.DataCol_cleanup_tutorial: PAGE_LIST_CLEANUP_TUTORIAL,
    SocketType.DataCol_rescue_practice: PAGE_LIST_RESCUE_FULL_OBS,
    SocketType.DataCol_rescue_test: PAGE_LIST_RESCUE,
    SocketType.DataCol_rescue_tutorial: PAGE_LIST_RESCUE_TUTORIAL,
}  # type: Mapping[Any, Sequence[CanvasPageBase]]
