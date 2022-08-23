from web_experiment.auth.page_replay import CanvasPageReplayBoxPush

SESSION_COLLECT_A = "session_collect_movers"
SESSION_COLLECT_B = "session_collect_cleanup"

COLLECT_NAMESPACES = {
    SESSION_COLLECT_A: "collect_movers",
    SESSION_COLLECT_B: "collect_cleanup",
}

COLLECT_CANVAS_PAGELIST = {
    SESSION_COLLECT_A: [CanvasPageReplayBoxPush(True)],
    SESSION_COLLECT_B: [CanvasPageReplayBoxPush(False)],
}

COLLECT_SESSION_GAME_TYPE = {
    SESSION_COLLECT_A: "BoxPushSimulator_AlwaysTogether",
    SESSION_COLLECT_B: "BoxPushSimulator_AlwaysAlone",
}
