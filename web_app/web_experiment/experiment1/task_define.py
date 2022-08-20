from typing import Mapping, Any, Sequence
from web_experiment.experiment1.page_base import CanvasPageBase
import web_experiment.experiment1.page_exp1_common as pgc
import web_experiment.experiment1.page_exp1_game as pge
import web_experiment.experiment1.page_tutorial as pgt
from ai_coach_domain.box_push.maps import EXP1_MAP, TUTORIAL_MAP

SESSION_A1 = 'session_a1'
SESSION_A2 = 'session_a2'
SESSION_A3 = 'session_a3'
SESSION_A4 = 'session_a4'
SESSION_B1 = 'session_b1'
SESSION_B2 = 'session_b2'
SESSION_B3 = 'session_b3'
SESSION_B4 = 'session_b4'
SESSION_B5 = 'session_b5'
TUTORIAL1 = 'tutorial1'
TUTORIAL2 = 'tutorial2'

LIST_SESSIONS = [
    SESSION_A1, SESSION_A2, SESSION_A3, SESSION_A4, SESSION_B1, SESSION_B2,
    SESSION_B3, SESSION_B4, SESSION_B5
]
LIST_TUTORIALS = [TUTORIAL1, TUTORIAL2]

EXP1_PAGENAMES = {
    SESSION_A1: 'exp1_both_tell_align',
    SESSION_A2: 'exp1_both_tell_align_2',
    SESSION_A3: 'exp1_both_user_random',
    SESSION_A4: 'exp1_both_user_random_2',
    SESSION_B1: 'exp1_indv_tell_align',
    SESSION_B2: 'exp1_indv_tell_random',
    SESSION_B3: 'exp1_indv_user_random',
    SESSION_B4: 'exp1_indv_user_random_2',
    SESSION_B5: 'exp1_indv_user_random_3',
    TUTORIAL1: 'tutorial1',
    TUTORIAL2: 'tutorial2',
}

EXP1_SESSION_TITLE = {
    SESSION_A1: 'Session A1',
    SESSION_A2: 'Session A2',
    SESSION_A3: 'Session A3',
    SESSION_A4: 'Session A4',
    SESSION_B1: 'Session B1',
    SESSION_B2: 'Session B2',
    SESSION_B3: 'Session B3',
    SESSION_B4: 'Session B4',
    SESSION_B5: 'Session B5',
    TUTORIAL1: 'Interactive Tutorial',
    TUTORIAL2: 'Interactive Tutorial',
}

EXP1_GAMEPAGES = {
    SESSION_A1: [
        pgc.CanvasPageStart(True),
        pgc.CanvasPageWarning(True),
        pge.CanvasPageMoversTellAligned(EXP1_MAP),
        pgc.CanvasPageEnd(True)
    ],
    SESSION_A2: [
        pgc.CanvasPageStart(True),
        pgc.CanvasPageWarning(True),
        pge.CanvasPageMoversTellAligned(EXP1_MAP),
        pgc.CanvasPageEnd(True)
    ],
    SESSION_A3: [
        pgc.CanvasPageStart(True),
        pgc.CanvasPageWarning(True),
        pge.CanvasPageMoversUserRandom(EXP1_MAP),
        pgc.CanvasPageEnd(True)
    ],
    SESSION_A4: [
        pgc.CanvasPageStart(True),
        pgc.CanvasPageWarning(True),
        pge.CanvasPageMoversUserRandom(EXP1_MAP),
        pgc.CanvasPageEnd(True)
    ],
    SESSION_B1: [
        pgc.CanvasPageStart(False),
        pgc.CanvasPageWarning(False),
        pge.CanvasPageCleanUpTellAligned(EXP1_MAP),
        pgc.CanvasPageEnd(False)
    ],
    SESSION_B2: [
        pgc.CanvasPageStart(False),
        pgc.CanvasPageWarning(False),
        pge.CanvasPageCleanUpTellRandom(EXP1_MAP),
        pgc.CanvasPageEnd(False)
    ],
    SESSION_B3: [
        pgc.CanvasPageStart(False),
        pgc.CanvasPageWarning(False),
        pge.CanvasPageCleanUpUserRandom(EXP1_MAP),
        pgc.CanvasPageEnd(False)
    ],
    SESSION_B4: [
        pgc.CanvasPageStart(False),
        pgc.CanvasPageWarning(False),
        pge.CanvasPageCleanUpUserRandom(EXP1_MAP),
        pgc.CanvasPageEnd(False)
    ],
    SESSION_B5: [
        pgc.CanvasPageStart(False),
        pgc.CanvasPageWarning(False),
        pge.CanvasPageCleanUpUserRandom(EXP1_MAP),
        pgc.CanvasPageEnd(False)
    ],
    TUTORIAL1: [
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
    ],
    TUTORIAL2: [
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
    ],
}  # type: Mapping[Any, Sequence[CanvasPageBase]]
