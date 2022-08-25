from web_experiment.define import EDomainType
from web_experiment.auth.page_replay import CanvasPageReplayBoxPush

REPLAY_NAMESPACES = {
    EDomainType.Movers: "replay_movers",
    EDomainType.Cleanup: "replay_cleanup",
}

REPLAY_CANVAS_PAGELIST = {
    EDomainType.Movers: [CanvasPageReplayBoxPush(True)],
    EDomainType.Cleanup: [CanvasPageReplayBoxPush(False)],
}
