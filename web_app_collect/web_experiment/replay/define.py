from web_experiment.auth.page_replay import CanvasPageReplayBoxPush

SESSION_RECORD_A = "session_record_movers"
SESSION_RECORD_B = "session_record_cleanup"

RECORD_NAMESPACES = {
    SESSION_RECORD_A: "record_movers",
    SESSION_RECORD_B: "record_cleanup",
}

RECORD_CANVAS_PAGELIST = {
    SESSION_RECORD_A: [CanvasPageReplayBoxPush(True)],
    SESSION_RECORD_B: [CanvasPageReplayBoxPush(False)],
}
