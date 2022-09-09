from enum import Enum
from typing import Mapping, Any, Sequence
from web_experiment.exp_common.page_base import CanvasPageBase
from ai_coach_domain.box_push_v2.maps import MAP_CLEANUP, MAP_MOVERS
import web_experiment.exp_common.page_exp1_common as pgc
from web_experiment.exp_common.page_boxpushv2_base import BoxPushV2UserRandom
from web_experiment.exp_intervention.page_intervention import (
    BoxPushV2Intervention)
import web_experiment.exp_common.page_tutorial as pgt
import web_experiment.exp_common.page_tutorial_rescue as pgr
from web_experiment.exp_common.page_rescue_game_base import RescueGameUserRandom
from web_experiment.define import GroupName, PageKey, EDomainType
from ai_coach_domain.rescue.maps import MAP_RESCUE_2

SESSION_TITLE = {
    PageKey.Interv_A0: 'A0',
    PageKey.Interv_A1: 'A1',
    PageKey.Interv_A2: 'A2',
    PageKey.Interv_B0: 'B0',
    PageKey.Interv_B1: 'B1',
    PageKey.Interv_B2: 'B2',
    PageKey.Interv_C0: 'C0',
    PageKey.Interv_C1: 'C1',
    PageKey.Interv_C2: 'C2',
    PageKey.Interv_T1: 'Interactive Tutorial',
    PageKey.Interv_T2: 'Interactive Tutorial',
    PageKey.Interv_T3: 'Interactive Tutorial',
}


class SocketType(Enum):
  '''
  The socket name should be unique across all experiments.
  (i.e. if DataCollection experiment and Intervention experiment have a socket,
  whose name is the same, socketio cannot distinguish which event handler to use
  '''
  Interv_movers_practice = 0
  Interv_movers_normal = 1
  Interv_cleanup_practice = 2
  Interv_cleanup_normal = 3
  Interv_movers_tutorial = 4
  Interv_cleanup_tutorial = 5
  Interv_movers_intervention = 6
  Interv_cleanup_intervention = 7
  Interv_rescue_practice = 8
  Interv_rescue_normal = 9
  Interv_rescue_tutorial = 10
  Interv_rescue_intervention = 11


def get_socket_name(page_key, group_id):
  socket_type = None
  if page_key == PageKey.Interv_A0:
    socket_type = SocketType.Interv_movers_practice
  elif page_key == PageKey.Interv_A1:
    socket_type = SocketType.Interv_movers_normal
  elif page_key == PageKey.Interv_A2:
    if group_id == GroupName.Group_B:
      socket_type = SocketType.Interv_movers_intervention
    else:
      socket_type = SocketType.Interv_movers_normal

  elif page_key == PageKey.Interv_B0:
    socket_type = SocketType.Interv_cleanup_practice
  elif page_key == PageKey.Interv_B1:
    socket_type = SocketType.Interv_cleanup_normal
  elif page_key == PageKey.Interv_B2:
    if group_id == GroupName.Group_B:
      socket_type = SocketType.Interv_cleanup_intervention
    else:
      socket_type = SocketType.Interv_cleanup_normal

  elif page_key == PageKey.Interv_C0:
    socket_type = SocketType.Interv_rescue_practice
  elif page_key == PageKey.Interv_C1:
    socket_type = SocketType.Interv_rescue_normal
  elif page_key == PageKey.Interv_C2:
    if group_id == GroupName.Group_B:
      socket_type = SocketType.Interv_rescue_intervention
    else:
      socket_type = SocketType.Interv_rescue_normal

  elif page_key == PageKey.Interv_T1:
    socket_type = SocketType.Interv_movers_tutorial
  elif page_key == PageKey.Interv_T2:
    socket_type = SocketType.Interv_cleanup_tutorial
  elif page_key == PageKey.Interv_T3:
    socket_type = SocketType.Interv_rescue_tutorial

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
PAGE_LIST_MOVERS_INTERV = [
    pgc.CanvasPageStart(EDomainType.Movers),
    pgc.CanvasPageWarning(EDomainType.Movers),
    BoxPushV2Intervention(EDomainType.Movers, MAP_MOVERS, True),
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
PAGE_LIST_CLEANUP_INTERV = [
    pgc.CanvasPageStart(EDomainType.Cleanup),
    pgc.CanvasPageWarning(EDomainType.Cleanup),
    BoxPushV2Intervention(EDomainType.Cleanup, MAP_CLEANUP, True),
    pgc.CanvasPageEnd(EDomainType.Cleanup)
]

PAGELIST_MOVERS_TUTORIAL = [
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
PAGELIST_CLEANUP_TUTORIAL = [
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
    SocketType.Interv_movers_practice: PAGE_LIST_MOVERS,
    SocketType.Interv_movers_normal: PAGE_LIST_MOVERS,
    SocketType.Interv_movers_intervention: PAGE_LIST_MOVERS_INTERV,
    SocketType.Interv_cleanup_practice: PAGE_LIST_CLEANUP,
    SocketType.Interv_cleanup_normal: PAGE_LIST_CLEANUP,
    SocketType.Interv_cleanup_intervention: PAGE_LIST_CLEANUP_INTERV,
    SocketType.Interv_movers_tutorial: PAGELIST_MOVERS_TUTORIAL,
    SocketType.Interv_cleanup_tutorial: PAGELIST_CLEANUP_TUTORIAL,
    SocketType.Interv_rescue_practice: PAGE_LIST_RESCUE,
    SocketType.Interv_rescue_normal: PAGE_LIST_RESCUE,
    SocketType.Interv_rescue_tutorial: PAGE_LIST_RESCUE_TUTORIAL,
    SocketType.Interv_rescue_intervention: PAGE_LIST_RESCUE,
}  # type: Mapping[Any, Sequence[CanvasPageBase]]
