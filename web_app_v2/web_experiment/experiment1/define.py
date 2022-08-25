from typing import Mapping, Any, Sequence
from web_experiment.experiment1.page_base import CanvasPageBase
from ai_coach_domain.box_push.maps import EXP1_MAP, TUTORIAL_MAP
import web_experiment.experiment1.page_exp1_common as pgc
import web_experiment.experiment1.page_exp1_game as pge
import web_experiment.experiment1.page_tutorial as pgt
from web_experiment.define import GROUP_B

SESSION_A0 = 'session_a0'
SESSION_A1 = 'session_a1'
SESSION_A2 = 'session_a2'
SESSION_B0 = 'session_b0'
SESSION_B1 = 'session_b1'
SESSION_B2 = 'session_b2'
TUTORIAL1 = 'tutorial1'
TUTORIAL2 = 'tutorial2'

LIST_SESSIONS = [
    SESSION_A0, SESSION_A1, SESSION_A2, SESSION_B0, SESSION_B1, SESSION_B2
]

LIST_TUTORIALS = [TUTORIAL1, TUTORIAL2]

LIST_PRACTICE_SESSIONS = [SESSION_A0, SESSION_B0]

LIST_TUTORIAL_AND_PRACTICE = [TUTORIAL1, TUTORIAL2, SESSION_A0, SESSION_B0]

LIST_1ST_SESSIONS = [SESSION_A1, SESSION_B1]

LIST_2ND_SESSIONS = [SESSION_A2, SESSION_B2]


def make_interv_session_name(session_name):
  return session_name + "_intervention"


def get_task_session_name(session_name):
  if len(session_name) > 13:
    if session_name[-13:] == "_intervention":
      return session_name[:-13]

  return session_name


def get_socket_namespace(session_name, group_id):
  if session_name not in LIST_2ND_SESSIONS:
    return EXP1_PAGENAMES[session_name]
  else:
    if group_id == GROUP_B:
      return EXP1_PAGENAMES[make_interv_session_name(session_name)]
    else:
      return EXP1_PAGENAMES[session_name]


EXP1_PAGENAMES = {
    SESSION_A0: 'exp1_both_tell_align',
    SESSION_A1: 'exp1_both_user_random',
    SESSION_A2: 'exp1_both_user_random_2',
    SESSION_B0: 'exp1_indv_tell_align',
    SESSION_B1: 'exp1_indv_user_random',
    SESSION_B2: 'exp1_indv_user_random_2',
    TUTORIAL1: 'tutorial1',
    TUTORIAL2: 'tutorial2',
    make_interv_session_name(SESSION_A2): 'exp1_both_user_random_2_interv',
    make_interv_session_name(SESSION_B2): 'exp1_indv_user_random_2_interv',
}

EXP1_SESSION_TITLE = {
    SESSION_A0: 'A0',
    SESSION_A1: 'A1',
    SESSION_A2: 'A2',
    SESSION_B0: 'B0',
    SESSION_B1: 'B1',
    SESSION_B2: 'B2',
    TUTORIAL1: 'Interactive Tutorial',
    TUTORIAL2: 'Interactive Tutorial',
}

PAGELIST_MOVERS_TELL_ALIGNED = [
    pgc.CanvasPageStart(True),
    pgc.CanvasPageWarning(True),
    pge.CanvasPageMoversTellAligned(EXP1_MAP),
    pgc.CanvasPageEnd(True)
]
PAGELIST_MOVERS_USER_RANDOM = [
    pgc.CanvasPageStart(True),
    pgc.CanvasPageWarning(True),
    pge.CanvasPageMoversUserRandom(EXP1_MAP, False),
    pgc.CanvasPageEnd(True)
]
PAGELIST_MOVERS_USER_RAND_INTERV = [
    pgc.CanvasPageStart(True),
    pgc.CanvasPageWarning(True),
    pge.CanvasPageMoversUserRandom(EXP1_MAP, True),
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
    pge.CanvasPageCleanUpUserRandom(EXP1_MAP, False),
    pgc.CanvasPageEnd(False)
]
PAGELIST_CLEANUP_USER_RAND_INTERV = [
    pgc.CanvasPageStart(False),
    pgc.CanvasPageWarning(False),
    pge.CanvasPageCleanUpUserRandom(EXP1_MAP, True),
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

EXP1_GAMEPAGES = {
    SESSION_A0: PAGELIST_MOVERS_TELL_ALIGNED,
    SESSION_A1: PAGELIST_MOVERS_USER_RANDOM,
    SESSION_A2: PAGELIST_MOVERS_USER_RANDOM,
    make_interv_session_name(SESSION_A2): PAGELIST_MOVERS_USER_RAND_INTERV,
    SESSION_B0: PAGELIST_CLEANUP_TELL_ALIGNED,
    SESSION_B1: PAGELIST_CLEANUP_USER_RANDOM,
    SESSION_B2: PAGELIST_CLEANUP_USER_RANDOM,
    make_interv_session_name(SESSION_B2): PAGELIST_CLEANUP_USER_RAND_INTERV,
    TUTORIAL1: PAGELIST_MOVERS_TUTORIAL,
    TUTORIAL2: PAGELIST_CLEANUP_TUTORIAL,
}  # type: Mapping[Any, Sequence[CanvasPageBase]]
