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
from web_experiment.define import GroupName, PageKey, EDomainType

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
PAGELIST_CLEANUP_TUTORIAL = [
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

# NOTE: placeholder
PAGE_LIST_RESCUE = PAGE_LIST_MOVERS

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
    SocketType.Interv_rescue_tutorial: PAGE_LIST_RESCUE,
    SocketType.Interv_rescue_intervention: PAGE_LIST_RESCUE,
}  # type: Mapping[Any, Sequence[CanvasPageBase]]
