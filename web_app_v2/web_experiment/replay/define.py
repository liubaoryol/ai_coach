from web_experiment.auth.page_replay import CanvasPageReplayBoxPush
from web_experiment.define import EDomainType

RECORD_NAMESPACES = {
    EDomainType.Movers: "record_movers",
    EDomainType.Cleanup: "record_cleanup",
}

RECORD_CANVAS_PAGELIST = {
    EDomainType.Movers: [CanvasPageReplayBoxPush(True)],
    EDomainType.Cleanup: [CanvasPageReplayBoxPush(False)],
}
