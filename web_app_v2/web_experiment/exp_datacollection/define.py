from enum import Enum
from typing import Mapping, Any, Sequence
from web_experiment.define import PageKey, EDomainType
from web_experiment.exp_common.page_base import CanvasPageBase
import web_experiment.exp_common.page_exp1_common as pgc
from web_experiment.exp_common.page_boxpushv2_base import BoxPushV2GamePage
import web_experiment.exp_common.page_tutorial as pgt
import web_experiment.exp_common.page_tutorial_rescue as pgr
from web_experiment.exp_common.page_rescue_game import RescueGamePage
import web_experiment.exp_common.page_tutorial_rescue_v2 as pgrv2

SESSION_TITLE = {
    PageKey.DataCol_A1: 'A1',
    PageKey.DataCol_A2: 'A2',
    PageKey.DataCol_A3: 'A3',
    PageKey.DataCol_A4: 'A4',
    PageKey.DataCol_C1: 'B1',
    PageKey.DataCol_C2: 'B2',
    PageKey.DataCol_C3: 'B3',
    PageKey.DataCol_C4: 'B4',
    PageKey.DataCol_T1: 'Interactive Tutorial',
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
  if page_key in [
      PageKey.DataCol_A1, PageKey.DataCol_A2, PageKey.DataCol_A3,
      PageKey.DataCol_A4
  ]:
    socket_type = SocketType.DataCol_movers_test

  elif page_key in [
      PageKey.DataCol_C1, PageKey.DataCol_C2, PageKey.DataCol_C3,
      PageKey.DataCol_C4
  ]:
    socket_type = SocketType.DataCol_rescue_test

  elif page_key == PageKey.DataCol_T1:
    socket_type = SocketType.DataCol_movers_tutorial
  elif page_key == PageKey.DataCol_T3:
    socket_type = SocketType.DataCol_rescue_tutorial

  return socket_type.name if socket_type is not None else None


PAGE_LIST_MOVERS_FULL_OBS = [
    pgc.CanvasPageStart(EDomainType.Movers),
    pgc.CanvasPageWarning(EDomainType.Movers),
    BoxPushV2GamePage(EDomainType.Movers, False),
    pgc.CanvasPageEnd(EDomainType.Movers)
]
PAGE_LIST_MOVERS = [
    pgc.CanvasPageStart(EDomainType.Movers),
    pgc.CanvasPageWarning(EDomainType.Movers),
    BoxPushV2GamePage(EDomainType.Movers, True),
    pgc.CanvasPageEnd(EDomainType.Movers)
]
PAGE_LIST_CLEANUP_FULL_OBS = [
    pgc.CanvasPageStart(EDomainType.Cleanup),
    pgc.CanvasPageWarning(EDomainType.Cleanup),
    BoxPushV2GamePage(EDomainType.Cleanup, False),
    pgc.CanvasPageEnd(EDomainType.Cleanup)
]
PAGE_LIST_CLEANUP = [
    pgc.CanvasPageStart(EDomainType.Cleanup),
    pgc.CanvasPageWarning(EDomainType.Cleanup),
    BoxPushV2GamePage(EDomainType.Cleanup, True),
    pgc.CanvasPageEnd(EDomainType.Cleanup)
]

PAGE_LIST_MOVERS_TUTORIAL = [
    pgc.CanvasPageTutorialStart(EDomainType.Movers),
    pgc.CanvasPageInstruction(EDomainType.Movers),
    pgc.CanvasPageTutorialGameStart(EDomainType.Movers),
    pgt.CanvasPageJoystick(EDomainType.Movers),
    pgt.CanvsPageWaitBtn(EDomainType.Movers),
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
    pgt.CanvasPageImportance(EDomainType.Movers),
    pgt.CanvasPageSelPrompt(EDomainType.Movers),
    pgt.CanvasPageSelResult(EDomainType.Movers, True),
    pgt.CanvasPageExpGoal(EDomainType.Movers),
    pgt.CanvasPageImportance(EDomainType.Movers),
    pgt.CanvasPageMiniGame(EDomainType.Movers)
]
PAGE_LIST_CLEANUP_TUTORIAL = [
    pgc.CanvasPageTutorialStart(EDomainType.Cleanup),
    pgc.CanvasPageInstruction(EDomainType.Cleanup),
    pgc.CanvasPageTutorialGameStart(EDomainType.Cleanup),
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
    RescueGamePage(False),
    pgc.CanvasPageEnd(EDomainType.Rescue)
]

PAGE_LIST_RESCUE = [
    pgc.CanvasPageStart(EDomainType.Rescue),
    pgc.CanvasPageWarning(EDomainType.Rescue),
    RescueGamePage(True),
    pgc.CanvasPageEnd(EDomainType.Rescue)
]

PAGE_LIST_RESCUE_TUTORIAL = [
    pgc.CanvasPageTutorialStart(EDomainType.Rescue),
    pgc.CanvasPageInstruction(EDomainType.Rescue),
    pgc.CanvasPageTutorialGameStart(EDomainType.Rescue),
    pgr.RescueTutorialActions(),
    pgr.RescueTutorialOverallGoal(),
    pgr.RescueTutorialOnlyHuman(),
    pgr.RescueTutorialSimpleTarget(),
    pgr.RescueTutorialResolvedAlone(),
    pgr.RescueTutorialScore(),
    pgr.RescueTutorialComplexTarget(),
    pgr.RescueTutorialComplexTargetTogether(),
    pgr.RescueTutorialResolvedTogether(),
    pgr.RescueTutorialPartialObs(),
    pgr.RescueTutorialLatent(),
    pgr.RescueTutorialSelResult(),
    pgr.RescueTutorialMiniGame()
]

PAGE_LIST_RESCUE_V2_TUTORIAL = [
    pgc.CanvasPageTutorialStart(EDomainType.Rescue),
    pgc.CanvasPageInstruction(EDomainType.Rescue),
    pgc.CanvasPageTutorialGameStart(EDomainType.Rescue),
    pgrv2.RescueV2TutorialMiniGame()
]

GAMEPAGES = {
    SocketType.DataCol_movers_practice:
    PAGE_LIST_MOVERS_FULL_OBS,
    SocketType.DataCol_movers_test:
    PAGE_LIST_MOVERS,
    SocketType.DataCol_cleanup_practice:
    PAGE_LIST_CLEANUP_FULL_OBS,
    SocketType.DataCol_cleanup_test:
    PAGE_LIST_CLEANUP,
    # SocketType.DataCol_movers_tutorial: PAGE_LIST_CLEANUP_TUTORIAL,
    SocketType.DataCol_movers_tutorial:
    PAGE_LIST_MOVERS_TUTORIAL,
    SocketType.DataCol_cleanup_tutorial:
    PAGE_LIST_CLEANUP_TUTORIAL,
    SocketType.DataCol_rescue_practice:
    PAGE_LIST_RESCUE_FULL_OBS,
    SocketType.DataCol_rescue_test:
    PAGE_LIST_RESCUE,
    # SocketType.DataCol_rescue_tutorial: PAGE_LIST_RESCUE_V2_TUTORIAL,
    SocketType.DataCol_rescue_tutorial:
    PAGE_LIST_RESCUE_TUTORIAL
}  # type: Mapping[Any, Sequence[CanvasPageBase]]
