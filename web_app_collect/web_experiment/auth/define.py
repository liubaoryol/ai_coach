from web_experiment.define import EDomainType
from web_experiment.auth.page_replay import CanvasPageReplayBoxPush

SESSION_REPLAY_A = "session_replay_movers"
SESSION_REPLAY_B = "session_replay_cleanup"

REPLAY_NAMESPACES = {
    SESSION_REPLAY_A: "replay_movers",
    SESSION_REPLAY_B: "replay_cleanup",
}

REPLAY_CANVAS_PAGELIST = {
    SESSION_REPLAY_A: [CanvasPageReplayBoxPush(True)],
    SESSION_REPLAY_B: [CanvasPageReplayBoxPush(False)],
}

REPALY_SESSION_TYPE = {
    SESSION_REPLAY_A: EDomainType.Movers,
    SESSION_REPLAY_B: EDomainType.Cleanup
}
