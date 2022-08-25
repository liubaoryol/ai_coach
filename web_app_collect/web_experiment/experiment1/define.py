from typing import Mapping, Any, Sequence
from web_experiment.experiment1.page_base import CanvasPageBase
import web_experiment.experiment1.page_exp1_common as pgc
import web_experiment.experiment1.page_exp1_game as pge
import web_experiment.experiment1.page_tutorial as pgt
from ai_coach_domain.box_push.maps import EXP1_MAP, TUTORIAL_MAP

SESSION_A0 = 'session_a0'
SESSION_A1 = 'session_a1'
SESSION_A2 = 'session_a2'
SESSION_A3 = 'session_a3'
SESSION_B0 = 'session_b0'
SESSION_B1 = 'session_b1'
SESSION_B2 = 'session_b2'
SESSION_B3 = 'session_b3'
TUTORIAL1 = 'tutorial1'
TUTORIAL2 = 'tutorial2'

LIST_SESSIONS = [
    SESSION_A0, SESSION_A1, SESSION_A2, SESSION_A3, SESSION_B0, SESSION_B1,
    SESSION_B2, SESSION_B3
]
LIST_TUTORIALS = [TUTORIAL1, TUTORIAL2]

EXP1_PAGENAMES = {
    SESSION_A0: 'exp1_both_tell_align',
    SESSION_A1: 'exp1_both_user_random',
    SESSION_A2: 'exp1_both_user_random_2',
    SESSION_A3: 'exp1_both_user_random_3',
    SESSION_B0: 'exp1_indv_tell_align',
    SESSION_B1: 'exp1_indv_user_random',
    SESSION_B2: 'exp1_indv_user_random_2',
    SESSION_B3: 'exp1_indv_user_random_3',
    TUTORIAL1: 'tutorial1',
    TUTORIAL2: 'tutorial2',
}

EXP1_SESSION_TITLE = {
    SESSION_A0: 'A0',
    SESSION_A1: 'A1',
    SESSION_A2: 'A2',
    SESSION_A3: 'A3',
    SESSION_B0: 'B0',
    SESSION_B1: 'B1',
    SESSION_B2: 'B2',
    SESSION_B3: 'B3',
    TUTORIAL1: 'Interactive Tutorial',
    TUTORIAL2: 'Interactive Tutorial',
}

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

EXP1_GAMEPAGES = {
    SESSION_A0: PAGE_LIST_MOVERS_TELL_ALIGNED,
    SESSION_A1: PAGE_LIST_MOVERS_USER_RANDOM,
    SESSION_A2: PAGE_LIST_MOVERS_USER_RANDOM,
    SESSION_A3: PAGE_LIST_MOVERS_USER_RANDOM,
    SESSION_B0: PAGE_LIST_CLEANUP_TELL_ALIGNED,
    SESSION_B1: PAGE_LIST_CLEANUP_USER_RANDOM,
    SESSION_B2: PAGE_LIST_CLEANUP_USER_RANDOM,
    SESSION_B3: PAGE_LIST_CLEANUP_USER_RANDOM,
    TUTORIAL1: PAGE_LIST_MOVERS_TUTORIAL,
    TUTORIAL2: PAGE_LIST_CLEANUP_TUTORIAL,
}  # type: Mapping[Any, Sequence[CanvasPageBase]]
