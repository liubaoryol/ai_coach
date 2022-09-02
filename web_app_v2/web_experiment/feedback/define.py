from web_experiment.define import EDomainType, PageKey
from web_experiment.exp_common.page_replay import CanvasPageReplayBoxPush


class SocketType:
  Collect_Movers = "collect_movers"
  Collect_Cleanup = "collect_cleanup"
  Collect_Movers_practice = "collect_movers_practice"
  Collect_Cleanup_practice = "collect_cleanup_practice"


PRACTICE_SESSION = [PageKey.DataCol_A0, PageKey.DataCol_B0]

COLLECT_CANVAS_PAGELIST = {
    SocketType.Collect_Movers: [CanvasPageReplayBoxPush(True, True)],
    SocketType.Collect_Cleanup: [CanvasPageReplayBoxPush(False, True)],
    SocketType.Collect_Movers_practice: [CanvasPageReplayBoxPush(True, False)],
    SocketType.Collect_Cleanup_practice:
    [CanvasPageReplayBoxPush(False, False)],
}

FEEDBACK_NAMESPACES = {
    EDomainType.Movers: "feedback_movers",
    EDomainType.Cleanup: "feedback_cleanup",
}

FEEDBACK_CANVAS_PAGELIST = {
    EDomainType.Movers: [CanvasPageReplayBoxPush(True, True)],
    EDomainType.Cleanup: [CanvasPageReplayBoxPush(False, True)],
}
