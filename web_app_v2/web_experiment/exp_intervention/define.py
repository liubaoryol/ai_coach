from enum import Enum
from typing import Mapping, Any, Sequence
from web_experiment.exp_common.page_base import CanvasPageBase
import web_experiment.exp_common.page_exp1_common as pgc
from web_experiment.exp_common.page_boxpushv2_base import BoxPushV2UserRandom
from web_experiment.exp_intervention.page_intervention import (
    BoxPushV2Intervention)
from web_experiment.exp_intervention.page_intervention_rescue import (
    RescueV2Intervention)
import web_experiment.exp_common.page_tutorial as pgt
import web_experiment.exp_common.page_tutorial_rescue as pgr
from web_experiment.exp_common.page_rescue_game import RescueGameUserRandom
from web_experiment.define import GroupName, PageKey, EDomainType

SESSION_TITLE = {
    PageKey.Interv_A0: 'A0',
    PageKey.Interv_A1: 'A1',
    PageKey.Interv_A2: 'A2',
    PageKey.Interv_C0: 'C0',
    PageKey.Interv_C1: 'C1',
    PageKey.Interv_C2: 'C2',
    PageKey.Interv_T1: 'Interactive Tutorial',
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
  elif page_key == PageKey.Interv_T3:
    socket_type = SocketType.Interv_rescue_tutorial

  return socket_type.name if socket_type is not None else None


PAGE_LIST_MOVERS_FULL_OBS = [
    pgc.CanvasPageStart(EDomainType.Movers),
    pgc.CanvasPageWarning(EDomainType.Movers),
    BoxPushV2UserRandom(EDomainType.Movers, False, False),
    pgc.CanvasPageEnd(EDomainType.Movers)
]
PAGE_LIST_MOVERS = [
    pgc.CanvasPageStart(EDomainType.Movers),
    pgc.CanvasPageWarning(EDomainType.Movers),
    BoxPushV2UserRandom(EDomainType.Movers, True, False),
    pgc.CanvasPageEnd(EDomainType.Movers)
]
PAGE_LIST_MOVERS_INTERV = [
    pgc.CanvasPageStart(EDomainType.Movers),
    pgc.CanvasPageWarning(EDomainType.Movers),
    BoxPushV2Intervention(EDomainType.Movers, True),
    pgc.CanvasPageEnd(EDomainType.Movers)
]
PAGE_LIST_CLEANUP_FULL_OBS = [
    pgc.CanvasPageStart(EDomainType.Cleanup),
    pgc.CanvasPageWarning(EDomainType.Cleanup),
    BoxPushV2UserRandom(EDomainType.Cleanup, False, False),
    pgc.CanvasPageEnd(EDomainType.Cleanup)
]
PAGE_LIST_CLEANUP = [
    pgc.CanvasPageStart(EDomainType.Cleanup),
    pgc.CanvasPageWarning(EDomainType.Cleanup),
    BoxPushV2UserRandom(EDomainType.Cleanup, True, False),
    pgc.CanvasPageEnd(EDomainType.Cleanup)
]
PAGE_LIST_CLEANUP_INTERV = [
    pgc.CanvasPageStart(EDomainType.Cleanup),
    pgc.CanvasPageWarning(EDomainType.Cleanup),
    BoxPushV2Intervention(EDomainType.Cleanup, True),
    pgc.CanvasPageEnd(EDomainType.Cleanup)
]

PAGELIST_MOVERS_TUTORIAL = [
    pgt.CanvasPageTutorialStart(EDomainType.Movers),
    pgt.CanvasPageInstruction(EDomainType.Movers),
    pgt.CanvasPageTutorialGameStart(EDomainType.Movers),
    pgt.CanvasPageJoystick(EDomainType.Movers),
    pgt.CanvasPageInvalidAction(EDomainType.Movers),
    pgt.CanvasPageOnlyHuman(EDomainType.Movers),
    pgt.CanvasPageGoToTarget(EDomainType.Movers),
    pgt.CanvasPagePickUpTargetAttempt(EDomainType.Movers),
    pgt.CanvasPagePickUpTarget(EDomainType.Movers),
    pgt.CanvasPageGoToGoal(EDomainType.Movers),
    pgt.CanvasPageRespawn(EDomainType.Movers),
    pgt.CanvasPageScore(EDomainType.Movers),
    pgt.CanvasPagePartialObs(EDomainType.Movers),
    pgt.CanvasPageTarget(EDomainType.Movers),
    pgt.CanvasPageLatent(EDomainType.Movers),
    pgt.CanvasPageSelResult(EDomainType.Movers, False),
    pgt.CanvasPageSelPrompt(EDomainType.Movers),
    pgt.CanvasPageSelResult(EDomainType.Movers, True),
    pgt.CanvasPageMiniGame(EDomainType.Movers)
]
PAGELIST_CLEANUP_TUTORIAL = [
    pgt.CanvasPageTutorialStart(EDomainType.Cleanup),
    pgt.CanvasPageInstruction(EDomainType.Cleanup),
    pgt.CanvasPageTutorialGameStart(EDomainType.Cleanup),
    pgt.CanvasPageJoystickShort(EDomainType.Cleanup),
    pgt.CanvasPageOnlyHuman(EDomainType.Cleanup),
    pgt.CanvasPageGoToTarget(EDomainType.Cleanup),
    pgt.CanvasPagePickUpTargetAttempt(EDomainType.Cleanup),
    pgt.CanvasPageGoToGoal(EDomainType.Cleanup),
    pgt.CanvasPageRespawn(EDomainType.Cleanup),
    pgt.CanvasPageScore(EDomainType.Cleanup),
    pgt.CanvasPagePartialObs(EDomainType.Cleanup),
    pgt.CanvasPageTarget(EDomainType.Cleanup),
    pgt.CanvasPageLatent(EDomainType.Cleanup),
    pgt.CanvasPageSelResult(EDomainType.Cleanup, False),
    pgt.CanvasPageSelPrompt(EDomainType.Cleanup),
    pgt.CanvasPageSelResult(EDomainType.Cleanup, True),
    pgt.CanvasPageMiniGame(EDomainType.Cleanup)
]

PAGE_LIST_RESCUE_FULL_OBS = [
    pgc.CanvasPageStart(EDomainType.Rescue),
    pgc.CanvasPageWarning(EDomainType.Rescue),
    RescueGameUserRandom(False, False),
    pgc.CanvasPageEnd(EDomainType.Rescue)
]

PAGE_LIST_RESCUE = [
    pgc.CanvasPageStart(EDomainType.Rescue),
    pgc.CanvasPageWarning(EDomainType.Rescue),
    RescueGameUserRandom(True, False),
    pgc.CanvasPageEnd(EDomainType.Rescue)
]
PAGE_LIST_RESCUE_INTERV = [
    pgc.CanvasPageStart(EDomainType.Rescue),
    pgc.CanvasPageWarning(EDomainType.Rescue),
    RescueV2Intervention(True),
    pgc.CanvasPageEnd(EDomainType.Rescue)
]
PAGE_LIST_RESCUE_TUTORIAL = [
    pgt.CanvasPageTutorialStart(EDomainType.Rescue),
    pgt.CanvasPageInstruction(EDomainType.Rescue),
    pgt.CanvasPageTutorialGameStart(EDomainType.Rescue),
    pgr.RescueTutorialActions(),
    pgr.RescueTutorialOverallGoal(),
    pgr.RescueTutorialOnlyHuman(),
    pgr.RescueTutorialSimpleTarget(),
    pgr.RescueTutorialResolvedAlone(),
    pgr.RescueTutorialComplexTarget(),
    pgr.RescueTutorialComplexTargetTogether(),
    pgr.RescueTutorialResolvedTogether(),
    pgr.RescueTutorialScore(),
    pgr.RescueTutorialPartialObs(),
    pgr.RescueTutorialDestination(),
    pgr.RescueTutorialLatent(),
    pgr.RescueTutorialSelResult(),
    pgr.RescueTutorialMiniGame()
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
    SocketType.Interv_rescue_intervention: PAGE_LIST_RESCUE_INTERV,
}  # type: Mapping[Any, Sequence[CanvasPageBase]]
