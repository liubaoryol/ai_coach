from web_experiment.define import EDomainType
from web_experiment.exp_common.page_replay import (BoxPushReplayPage,
                                                   BoxPushReviewPage,
                                                   RescueReplayPage,
                                                   RescueReviewPage)
from aic_domain.box_push_v2.maps import MAP_MOVERS
from aic_domain.box_push_v2.maps import MAP_CLEANUP_V2 as MAP_CLEANUP

from aic_domain.rescue.maps import MAP_RESCUE


def get_socket_name(page_key, domain_type: EDomainType):
  return page_key + domain_type.name


REPLAY_CANVAS_PAGELIST = {
    EDomainType.Movers:
    [BoxPushReplayPage(EDomainType.Movers, True, MAP_MOVERS)],
    EDomainType.Cleanup:
    [BoxPushReplayPage(EDomainType.Cleanup, True, MAP_CLEANUP)],
    EDomainType.Rescue: [RescueReplayPage(True, MAP_RESCUE)]
}

REVIEW_CANVAS_PAGELIST = {
    EDomainType.Movers:
    [BoxPushReviewPage(EDomainType.Movers, True, MAP_MOVERS)],
    EDomainType.Cleanup:
    [BoxPushReviewPage(EDomainType.Cleanup, True, MAP_CLEANUP)],
    EDomainType.Rescue: [RescueReviewPage(True, MAP_RESCUE)]
}
