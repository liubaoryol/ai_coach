from enum import Enum
from typing import Mapping, Any, Sequence
from web_experiment.exp_common.page_base import CanvasPageBase
import web_experiment.exp_common.page_exp1_common as pgc
from web_experiment.exp_common.page_boxpushv2_base import BoxPushV2GamePage
from web_experiment.exp_intervention.page_intervention_movers import (
    BoxPushV2InterventionPage)
from web_experiment.exp_intervention.page_intervention_rescue import (
    RescueV2InterventionPage)
import web_experiment.exp_common.page_tutorial as pgt
import web_experiment.exp_common.page_tutorial_rescue as pgr
from web_experiment.exp_common.page_rescue_game import RescueGamePage
from web_experiment.define import GroupName, PageKey, EDomainType
import web_experiment.exp_intervention.page_tutorial_intervention as pgi

SESSION_TITLE = {
    PageKey.Interv_A0: 'A0',
    PageKey.Interv_A1: 'A1',
    PageKey.Interv_A2: 'A2',
    PageKey.Interv_A3: 'A3',
    PageKey.Interv_A4: 'A4',
    PageKey.Interv_C0: 'B0',
    PageKey.Interv_C1: 'B1',
    PageKey.Interv_C2: 'B2',
    PageKey.Interv_C3: 'B3',
    PageKey.Interv_C4: 'B4',
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
  Interv_movers_tutorial_groub_b = 12
  Interv_rescue_tutorial_groub_b = 13


def get_socket_name(page_key, group_id):
  socket_type = None

  if page_key == PageKey.Interv_A1 or page_key == PageKey.Interv_A2 or page_key == PageKey.Interv_A3 or page_key == PageKey.Interv_A4:
    if group_id == GroupName.Group_B:
      socket_type = SocketType.Interv_movers_intervention
    else:
      socket_type = SocketType.Interv_movers_normal

  if page_key == PageKey.Interv_C1 or page_key == PageKey.Interv_C2 or page_key == PageKey.Interv_C3 or page_key == PageKey.Interv_C4:
    if group_id == GroupName.Group_B:
      socket_type = SocketType.Interv_rescue_intervention
    else:
      socket_type = SocketType.Interv_rescue_normal

  elif page_key == PageKey.Interv_T1:
    if group_id == GroupName.Group_B:
      socket_type = SocketType.Interv_movers_tutorial_groub_b
    else:
      socket_type = SocketType.Interv_movers_tutorial
  elif page_key == PageKey.Interv_T3:
    if group_id == GroupName.Group_B:
      socket_type = SocketType.Interv_rescue_tutorial_groub_b
    else:
      socket_type = SocketType.Interv_rescue_tutorial

  return socket_type.name if socket_type is not None else None


PAGE_LIST_MOVERS_FULL_OBS = [
    pgc.CanvasPageStart(EDomainType.Movers),
    pgc.CanvasPageWarning(EDomainType.Movers),
    BoxPushV2GamePage(EDomainType.Movers, False, False),
    pgc.CanvasPageEnd(EDomainType.Movers)
]
PAGE_LIST_MOVERS = [
    pgc.CanvasPageStart(EDomainType.Movers),
    pgc.CanvasPageWarning(EDomainType.Movers),
    BoxPushV2GamePage(EDomainType.Movers, True, False),
    pgc.CanvasPageEnd(EDomainType.Movers)
]
PAGE_LIST_MOVERS_INTERV = [
    pgc.CanvasPageStart(EDomainType.Movers),
    pgc.CanvasPageWarning(EDomainType.Movers),
    BoxPushV2InterventionPage(EDomainType.Movers),
    pgc.CanvasPageEnd(EDomainType.Movers)
]
PAGE_LIST_CLEANUP_FULL_OBS = [
    pgc.CanvasPageStart(EDomainType.Cleanup),
    pgc.CanvasPageWarning(EDomainType.Cleanup),
    BoxPushV2GamePage(EDomainType.Cleanup, False, False),
    pgc.CanvasPageEnd(EDomainType.Cleanup)
]
PAGE_LIST_CLEANUP = [
    pgc.CanvasPageStart(EDomainType.Cleanup),
    pgc.CanvasPageWarning(EDomainType.Cleanup),
    BoxPushV2GamePage(EDomainType.Cleanup, True, False),
    pgc.CanvasPageEnd(EDomainType.Cleanup)
]
PAGE_LIST_CLEANUP_INTERV = [
    pgc.CanvasPageStart(EDomainType.Cleanup),
    pgc.CanvasPageWarning(EDomainType.Cleanup),
    BoxPushV2InterventionPage(EDomainType.Cleanup),
    pgc.CanvasPageEnd(EDomainType.Cleanup)
]

PAGELIST_MOVERS_TUTORIAL = [
    pgc.CanvasPageTutorialStart(EDomainType.Movers),
    pgc.CanvasPageInstruction(EDomainType.Movers),
    pgc.CanvasPageTutorialGameStart(EDomainType.Movers),
    pgt.CanvasPageJoystick(EDomainType.Movers, False),
    pgt.CanvsPageWaitBtn(EDomainType.Movers, False),
    pgt.CanvasPageInvalidAction(EDomainType.Movers, False),
    pgt.CanvasPageOnlyHuman(EDomainType.Movers, False),
    pgt.CanvasPageGoToTarget(EDomainType.Movers, False),
    pgt.CanvasPagePickUpTargetAttempt(EDomainType.Movers, False),
    pgt.CanvasPagePickUpTarget(EDomainType.Movers, False),
    pgt.CanvasPageGoToGoal(EDomainType.Movers, False),
    pgt.CanvasPageRespawn(EDomainType.Movers, False),
    pgt.CanvasPageScore(EDomainType.Movers, False),
    pgt.CanvasPagePartialObs(EDomainType.Movers, False),
    pgt.CanvasPageExpGoal(EDomainType.Movers, False),
    pgt.CanvasPageMiniGame(EDomainType.Movers, False)
]

PAGELIST_CLEANUP_TUTORIAL = [
    pgc.CanvasPageTutorialStart(EDomainType.Cleanup),
    pgc.CanvasPageInstruction(EDomainType.Cleanup),
    pgc.CanvasPageTutorialGameStart(EDomainType.Cleanup),
    pgt.CanvasPageJoystickShort(EDomainType.Cleanup, False),
    pgt.CanvasPageOnlyHuman(EDomainType.Cleanup, False),
    pgt.CanvasPageGoToTarget(EDomainType.Cleanup, False),
    pgt.CanvasPagePickUpTargetAttempt(EDomainType.Cleanup, False),
    pgt.CanvasPageGoToGoal(EDomainType.Cleanup, False),
    pgt.CanvasPageRespawn(EDomainType.Cleanup, False),
    pgt.CanvasPageScore(EDomainType.Cleanup, False),
    pgt.CanvasPagePartialObs(EDomainType.Cleanup, False),
    pgt.CanvasPageMiniGame(EDomainType.Cleanup, False)
]

PAGE_LIST_RESCUE_FULL_OBS = [
    pgc.CanvasPageStart(EDomainType.Rescue),
    pgc.CanvasPageWarning(EDomainType.Rescue),
    RescueGamePage(False, False),
    pgc.CanvasPageEnd(EDomainType.Rescue)
]

PAGE_LIST_RESCUE = [
    pgc.CanvasPageStart(EDomainType.Rescue),
    pgc.CanvasPageWarning(EDomainType.Rescue),
    RescueGamePage(True, False),
    pgc.CanvasPageEnd(EDomainType.Rescue)
]
PAGE_LIST_RESCUE_INTERV = [
    pgc.CanvasPageStart(EDomainType.Rescue),
    pgc.CanvasPageWarning(EDomainType.Rescue),
    RescueV2InterventionPage(),
    pgc.CanvasPageEnd(EDomainType.Rescue)
]
PAGE_LIST_RESCUE_TUTORIAL = [
    pgc.CanvasPageTutorialStart(EDomainType.Rescue),
    pgc.CanvasPageInstruction(EDomainType.Rescue),
    pgc.CanvasPageTutorialGameStart(EDomainType.Rescue),
    pgr.RescueTutorialActions(False),
    pgr.RescueTutorialOverallGoal(False),
    pgr.RescueTutorialOnlyHuman(False),
    pgr.RescueTutorialSimpleTarget(False),
    pgr.RescueTutorialResolvedAlone(False),
    pgr.RescueTutorialScore(False),
    pgr.RescueTutorialComplexTarget(False),
    pgr.RescueTutorialComplexTargetTogether(False),
    pgr.RescueTutorialResolvedTogether(False),
    pgr.RescueTutorialPartialObs(False),
    pgr.RescueTutorialMiniGame(False)
]

PAGELIST_MOVERS_INTERV_TUTORIAL = [
    pgc.CanvasPageTutorialStart(EDomainType.Movers),
    pgc.CanvasPageInstruction(EDomainType.Movers),
    pgc.CanvasPageTutorialGameStart(EDomainType.Movers),
    pgt.CanvasPageJoystick(EDomainType.Movers, False),
    pgt.CanvsPageWaitBtn(EDomainType.Movers, False),
    pgt.CanvasPageInvalidAction(EDomainType.Movers, False),
    pgt.CanvasPageOnlyHuman(EDomainType.Movers, False),
    pgt.CanvasPageGoToTarget(EDomainType.Movers, False),
    pgt.CanvasPagePickUpTargetAttempt(EDomainType.Movers, False),
    pgt.CanvasPagePickUpTarget(EDomainType.Movers, False),
    pgt.CanvasPageGoToGoal(EDomainType.Movers, False),
    pgt.CanvasPageRespawn(EDomainType.Movers, False),
    pgt.CanvasPageScore(EDomainType.Movers, False),
    pgt.CanvasPagePartialObs(EDomainType.Movers, False),
    pgt.CanvasPageExpGoal(EDomainType.Movers, False),
    pgi.BoxPushTutorialInterventionIntro(EDomainType.Movers, False),
    pgi.BoxPushTutorialInterventionUI(EDomainType.Movers),
    pgt.CanvasPageMiniGame(EDomainType.Movers, False)
]

PAGELIST_RESCUE_INTERV_TUTORIAL = [
    pgc.CanvasPageTutorialStart(EDomainType.Rescue),
    pgc.CanvasPageInstruction(EDomainType.Rescue),
    pgc.CanvasPageTutorialGameStart(EDomainType.Rescue),
    pgr.RescueTutorialActions(False),
    pgr.RescueTutorialOverallGoal(False),
    pgr.RescueTutorialOnlyHuman(False),
    pgr.RescueTutorialSimpleTarget(False),
    pgr.RescueTutorialResolvedAlone(False),
    pgr.RescueTutorialScore(False),
    pgr.RescueTutorialComplexTarget(False),
    pgr.RescueTutorialComplexTargetTogether(False),
    pgr.RescueTutorialResolvedTogether(False),
    pgr.RescueTutorialPartialObs(False),
    pgi.RescueTutorialInterventionIntro(False),
    pgr.RescueTutorialMiniGame(False)
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
    SocketType.Interv_movers_tutorial_groub_b: PAGELIST_MOVERS_INTERV_TUTORIAL,
    SocketType.Interv_rescue_tutorial_groub_b: PAGELIST_RESCUE_INTERV_TUTORIAL
}  # type: Mapping[Any, Sequence[CanvasPageBase]]
