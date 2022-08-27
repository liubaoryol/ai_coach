from enum import Enum
from typing import Mapping, Any, Sequence
from web_experiment.define import PageKey
from web_experiment.exp_common.page_base import CanvasPageBase
import web_experiment.exp_common.page_exp1_common as pgc
import web_experiment.exp_common.page_exp1_game as pge
import web_experiment.exp_common.page_tutorial as pgt
from ai_coach_domain.box_push.maps import EXP1_MAP, TUTORIAL_MAP

SESSION_TITLE = {
    PageKey.DataCol_A0: 'A0',
    PageKey.DataCol_A1: 'A1',
    PageKey.DataCol_A2: 'A2',
    PageKey.DataCol_A3: 'A3',
    PageKey.DataCol_B0: 'B0',
    PageKey.DataCol_B1: 'B1',
    PageKey.DataCol_B2: 'B2',
    PageKey.DataCol_B3: 'B3',
    PageKey.DataCol_T1: 'Interactive Tutorial',
    PageKey.DataCol_T2: 'Interactive Tutorial',
}


class SocketType(Enum):
  Movers_tell_aligned = 0
  Movers_user_random = 1
  Cleanup_tell_aligned = 2
  Cleanup_user_random = 3
  Movers_tutorial = 4
  Cleanup_tutorial = 5


def get_socket_type(page_key):
  if page_key == PageKey.DataCol_A0:
    return SocketType.Movers_tell_aligned
  elif page_key in [PageKey.DataCol_A1, PageKey.DataCol_A2, PageKey.DataCol_A3]:
    return SocketType.Movers_user_random
  elif page_key == PageKey.DataCol_B0:
    return SocketType.Cleanup_tell_aligned
  elif page_key in [PageKey.DataCol_B1, PageKey.DataCol_B2, PageKey.DataCol_B3]:
    return SocketType.Cleanup_user_random
  elif page_key == PageKey.DataCol_T1:
    return SocketType.Movers_tutorial
  elif page_key == PageKey.DataCol_T2:
    return SocketType.Cleanup_tutorial
  else:
    return None


def get_socket_namespace(page_key):
  socket_type = get_socket_type(page_key)
  return page_key + "_" + str(socket_type.value)


PAGE_LIST_MOVERS_TELL_ALIGNED = [
    pgc.CanvasPageStart(True),
    pgc.CanvasPageWarning(True),
    pge.CanvasPageMoversTellAligned(EXP1_MAP),
    pgc.CanvasPageEnd(True)
]
PAGE_LIST_MOVERS_USER_RANDOM = [
    pgc.CanvasPageStart(True),
    pgc.CanvasPageWarning(True),
    pge.CanvasPageMoversUserRandom(EXP1_MAP),
    pgc.CanvasPageEnd(True)
]
PAGE_LIST_CLEANUP_TELL_ALIGNED = [
    pgc.CanvasPageStart(False),
    pgc.CanvasPageWarning(False),
    pge.CanvasPageCleanUpTellAligned(EXP1_MAP),
    pgc.CanvasPageEnd(False)
]
PAGE_LIST_CLEANUP_USER_RANDOM = [
    pgc.CanvasPageStart(False),
    pgc.CanvasPageWarning(False),
    pge.CanvasPageCleanUpUserRandom(EXP1_MAP),
    pgc.CanvasPageEnd(False)
]

PAGE_LIST_MOVERS_TUTORIAL = [
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
PAGE_LIST_CLEANUP_TUTORIAL = [
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
    SocketType.Movers_tell_aligned: PAGE_LIST_MOVERS_TELL_ALIGNED,
    SocketType.Movers_user_random: PAGE_LIST_MOVERS_USER_RANDOM,
    SocketType.Cleanup_tell_aligned: PAGE_LIST_CLEANUP_TELL_ALIGNED,
    SocketType.Cleanup_user_random: PAGE_LIST_CLEANUP_USER_RANDOM,
    SocketType.Movers_tutorial: PAGE_LIST_MOVERS_TUTORIAL,
    SocketType.Cleanup_tutorial: PAGE_LIST_CLEANUP_TUTORIAL,
}  # type: Mapping[Any, Sequence[CanvasPageBase]]
