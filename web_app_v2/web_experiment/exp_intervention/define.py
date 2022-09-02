from enum import Enum
from typing import Mapping, Any, Sequence
from web_experiment.exp_common.page_base import CanvasPageBase
from ai_coach_domain.box_push.maps import TUTORIAL_MAP
from ai_coach_domain.box_push_v2.maps import MAP_CLEANUP, MAP_MOVERS
import web_experiment.exp_common.page_exp1_common as pgc
from web_experiment.exp_common.page_boxpushv2_base import BoxPushV2UserRandom
from web_experiment.exp_intervention.page_intervention import (
    BoxPushV2Intervention)
import web_experiment.exp_common.page_tutorial as pgt
from web_experiment.define import GroupName, PageKey

SESSION_TITLE = {
    PageKey.Interv_A0: 'A0',
    PageKey.Interv_A1: 'A1',
    PageKey.Interv_A2: 'A2',
    PageKey.Interv_B0: 'B0',
    PageKey.Interv_B1: 'B1',
    PageKey.Interv_B2: 'B2',
    PageKey.Interv_T1: 'Interactive Tutorial',
    PageKey.Interv_T2: 'Interactive Tutorial',
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
  elif page_key == PageKey.Interv_T1:
    socket_type = SocketType.Interv_movers_tutorial
  elif page_key == PageKey.Interv_T2:
    socket_type = SocketType.Interv_cleanup_tutorial

  return socket_type.name if socket_type is not None else None


PAGE_LIST_MOVERS_FULL_OBS = [
    pgc.CanvasPageStart(True),
    pgc.CanvasPageWarning(True),
    BoxPushV2UserRandom(True, MAP_MOVERS, False),
    pgc.CanvasPageEnd(True)
]
PAGE_LIST_MOVERS = [
    pgc.CanvasPageStart(True),
    pgc.CanvasPageWarning(True),
    BoxPushV2UserRandom(True, MAP_MOVERS, True),
    pgc.CanvasPageEnd(True)
]
PAGE_LIST_MOVERS_INTERV = [
    pgc.CanvasPageStart(True),
    pgc.CanvasPageWarning(True),
    BoxPushV2Intervention(True, MAP_MOVERS, True),
    pgc.CanvasPageEnd(True)
]
PAGE_LIST_CLEANUP_FULL_OBS = [
    pgc.CanvasPageStart(False),
    pgc.CanvasPageWarning(False),
    BoxPushV2UserRandom(False, MAP_CLEANUP, False),
    pgc.CanvasPageEnd(False)
]
PAGE_LIST_CLEANUP = [
    pgc.CanvasPageStart(False),
    pgc.CanvasPageWarning(False),
    BoxPushV2UserRandom(False, MAP_CLEANUP, True),
    pgc.CanvasPageEnd(False)
]
PAGE_LIST_CLEANUP_INTERV = [
    pgc.CanvasPageStart(False),
    pgc.CanvasPageWarning(False),
    BoxPushV2Intervention(False, MAP_CLEANUP, True),
    pgc.CanvasPageEnd(False)
]

PAGELIST_MOVERS_TUTORIAL = [
    pgt.CanvasPageTutorialStart(True),
    pgt.CanvasPageInstruction(True),
    pgt.CanvasPageTutorialGameStart(True),
    pgt.CanvasPageJoystick(True, TUTORIAL_MAP),
    pgt.CanvasPageInvalidAction(True, TUTORIAL_MAP),
    pgt.CanvasPageOnlyHuman(True, TUTORIAL_MAP),
    pgt.CanvasPageGoToTarget(True, TUTORIAL_MAP),
    pgt.CanvasPagePickUpTarget(True, TUTORIAL_MAP),
    pgt.CanvasPageGoToGoal(True, TUTORIAL_MAP),
    pgt.CanvasPageScore(True, TUTORIAL_MAP),
    pgt.CanvasPageTrapped(True, TUTORIAL_MAP),
    pgt.CanvasPageTargetHint(True, TUTORIAL_MAP),
    pgt.CanvasPageTargetNoHint(True, TUTORIAL_MAP),
    pgt.CanvasPageLatent(True, TUTORIAL_MAP),
    pgt.CanvasPageSelResult(True, TUTORIAL_MAP, False),
    pgt.CanvasPageSelPrompt(True, TUTORIAL_MAP),
    pgt.CanvasPageSelResult(True, TUTORIAL_MAP, True),
    pgt.CanvasPageMiniGame(True, TUTORIAL_MAP)
]
PAGELIST_CLEANUP_TUTORIAL = [
    pgt.CanvasPageTutorialStart(False),
    pgt.CanvasPageInstruction(False),
    pgt.CanvasPageTutorialGameStart(False),
    pgt.CanvasPageJoystickShort(False, TUTORIAL_MAP),
    pgt.CanvasPageOnlyHuman(False, TUTORIAL_MAP),
    pgt.CanvasPageGoToTarget(False, TUTORIAL_MAP),
    pgt.CanvasPagePickUpTarget(False, TUTORIAL_MAP),
    pgt.CanvasPageGoToGoal(False, TUTORIAL_MAP),
    pgt.CanvasPageScore(False, TUTORIAL_MAP),
    pgt.CanvasPageTrapped(False, TUTORIAL_MAP),
    pgt.CanvasPageTargetHint(False, TUTORIAL_MAP),
    pgt.CanvasPageTargetNoHint(False, TUTORIAL_MAP),
    pgt.CanvasPageLatent(False, TUTORIAL_MAP),
    pgt.CanvasPageSelResult(False, TUTORIAL_MAP, False),
    pgt.CanvasPageSelPrompt(False, TUTORIAL_MAP),
    pgt.CanvasPageSelResult(False, TUTORIAL_MAP, True),
    pgt.CanvasPageMiniGame(False, TUTORIAL_MAP)
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
}  # type: Mapping[Any, Sequence[CanvasPageBase]]
