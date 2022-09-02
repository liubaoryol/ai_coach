from web_experiment.define import EDomainType
from web_experiment.exp_common.page_replay import (CanvasPageReplayBoxPush,
                                                   CanvasPageReviewBoxPush)


def get_socket_name(page_key, domain_type: EDomainType):
  return page_key + domain_type.name


REPLAY_CANVAS_PAGELIST = {
    EDomainType.Movers: [CanvasPageReplayBoxPush(True, True)],
    EDomainType.Cleanup: [CanvasPageReplayBoxPush(False, True)],
}

REVIEW_CANVAS_PAGELIST = {
    EDomainType.Movers: [CanvasPageReviewBoxPush(True, True)],
    EDomainType.Cleanup: [CanvasPageReviewBoxPush(False, True)],
}
