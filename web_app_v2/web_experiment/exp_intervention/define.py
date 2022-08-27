from enum import Enum
from typing import Mapping, Any, Sequence
from web_experiment.exp_common.page_base import CanvasPageBase
from ai_coach_domain.box_push.maps import EXP1_MAP, TUTORIAL_MAP
import web_experiment.exp_common.page_exp1_common as pgc
import web_experiment.exp_common.page_exp1_game as pge
import web_experiment.exp_common.page_tutorial as pgt
import web_experiment.exp_intervention.page_exp_intervention as pgi
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
  Interv_movers_tell_aligned = 0
  Interv_movers_user_random = 1
  Interv_cleanup_tell_aligned = 2
  Interv_cleanup_user_random = 3
  Interv_movers_tutorial = 4
  Interv_cleanup_tutorial = 5
  Interv_movers_intervention = 6
  Interv_cleanup_intervention = 7


def get_socket_type(page_key, group_id):
  if page_key == PageKey.Interv_A0:
    return SocketType.Interv_movers_tell_aligned
  elif page_key == PageKey.Interv_A1:
    return SocketType.Interv_movers_user_random
  elif page_key == PageKey.Interv_A2:
    if group_id == GroupName.Group_B:
      return SocketType.Interv_movers_intervention
    else:
      return SocketType.Interv_movers_user_random
  elif page_key == PageKey.Interv_B0:
    return SocketType.Interv_cleanup_tell_aligned
  elif page_key == PageKey.Interv_B1:
    return SocketType.Interv_cleanup_user_random
  elif page_key == PageKey.Interv_B2:
    if group_id == GroupName.Group_B:
      return SocketType.Interv_cleanup_intervention
    else:
      return SocketType.Interv_cleanup_user_random
  elif page_key == PageKey.Interv_T1:
    return SocketType.Interv_movers_tutorial
  elif page_key == PageKey.Interv_T2:
    return SocketType.Interv_cleanup_tutorial
  else:
    return None


PAGELIST_MOVERS_TELL_ALIGNED = [
    pgc.CanvasPageStart(True),
    pgc.CanvasPageWarning(True),
    pge.CanvasPageMoversTellAligned(EXP1_MAP),
    pgc.CanvasPageEnd(True)
]
PAGELIST_MOVERS_USER_RANDOM = [
    pgc.CanvasPageStart(True),
    pgc.CanvasPageWarning(True),
    pge.CanvasPageMoversUserRandom(EXP1_MAP),
    pgc.CanvasPageEnd(True)
]
PAGELIST_MOVERS_USER_RAND_INTERV = [
    pgc.CanvasPageStart(True),
    pgc.CanvasPageWarning(True),
    pgi.CanvasPageMoversIntervention(EXP1_MAP),
    pgc.CanvasPageEnd(True)
]
PAGELIST_CLEANUP_TELL_ALIGNED = [
    pgc.CanvasPageStart(False),
    pgc.CanvasPageWarning(False),
    pge.CanvasPageCleanUpTellAligned(EXP1_MAP),
    pgc.CanvasPageEnd(False)
]
PAGELIST_CLEANUP_USER_RANDOM = [
    pgc.CanvasPageStart(False),
    pgc.CanvasPageWarning(False),
    pge.CanvasPageCleanUpUserRandom(EXP1_MAP),
    pgc.CanvasPageEnd(False)
]
PAGELIST_CLEANUP_USER_RAND_INTERV = [
    pgc.CanvasPageStart(False),
    pgc.CanvasPageWarning(False),
    pgi.CanvasPageCleanUpIntervention(EXP1_MAP),
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
    SocketType.Interv_movers_tell_aligned: PAGELIST_MOVERS_TELL_ALIGNED,
    SocketType.Interv_movers_user_random: PAGELIST_MOVERS_USER_RANDOM,
    SocketType.Interv_movers_intervention: PAGELIST_MOVERS_USER_RAND_INTERV,
    SocketType.Interv_cleanup_tell_aligned: PAGELIST_CLEANUP_TELL_ALIGNED,
    SocketType.Interv_cleanup_user_random: PAGELIST_CLEANUP_USER_RANDOM,
    SocketType.Interv_cleanup_intervention: PAGELIST_CLEANUP_USER_RAND_INTERV,
    SocketType.Interv_movers_tutorial: PAGELIST_MOVERS_TUTORIAL,
    SocketType.Interv_cleanup_tutorial: PAGELIST_CLEANUP_TUTORIAL,
}  # type: Mapping[Any, Sequence[CanvasPageBase]]
