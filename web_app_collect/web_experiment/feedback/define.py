from web_experiment.define import EDomainType
from web_experiment.auth.page_replay import CanvasPageReplayBoxPush

COLLECT_NAMESPACES = {
    EDomainType.Movers: "collect_movers",
    EDomainType.Cleanup: "collect_cleanup",
}

COLLECT_CANVAS_PAGELIST = {
    EDomainType.Movers: [CanvasPageReplayBoxPush(True)],
    EDomainType.Cleanup: [CanvasPageReplayBoxPush(False)],
}

COLLECT_SESSION_GAME_TYPE = {
    EDomainType.Movers: "BoxPushSimulator_AlwaysTogether",
    EDomainType.Cleanup: "BoxPushSimulator_AlwaysAlone",
}
