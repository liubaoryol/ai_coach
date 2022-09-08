from web_experiment.define import EDomainType, PageKey
from web_experiment.exp_common.page_replay import (BoxPushReplayPage,
                                                   RescueReplayPage)
from ai_coach_domain.box_push_v2.maps import MAP_CLEANUP, MAP_MOVERS
from ai_coach_domain.rescue.maps import MAP_RESCUE_2


class SocketType:
  Collect_Movers = "collect_movers"
  Collect_Cleanup = "collect_cleanup"
  Collect_Rescue = "collect_rescue"
  Collect_Movers_practice = "collect_movers_practice"
  Collect_Cleanup_practice = "collect_cleanup_practice"
  Collect_Rescue_practice = "collect_rescue_practice"


PRACTICE_SESSION = [PageKey.DataCol_A0, PageKey.DataCol_B0, PageKey.DataCol_C0]


def get_socket_name(pagekey, domain_type: EDomainType):
  if domain_type == EDomainType.Movers:
    if pagekey in PRACTICE_SESSION:
      socket_name = SocketType.Collect_Movers_practice
    else:
      socket_name = SocketType.Collect_Movers
  elif domain_type == EDomainType.Cleanup:
    if pagekey in PRACTICE_SESSION:
      socket_name = SocketType.Collect_Cleanup_practice
    else:
      socket_name = SocketType.Collect_Cleanup
  elif domain_type == EDomainType.Rescue:
    if pagekey in PRACTICE_SESSION:
      socket_name = SocketType.Collect_Rescue_practice
    else:
      socket_name = SocketType.Collect_Rescue
  else:
    raise NotImplementedError

  return socket_name


COLLECT_SOCKET_DOMAIN = {
    SocketType.Collect_Movers: EDomainType.Movers,
    SocketType.Collect_Cleanup: EDomainType.Cleanup,
    SocketType.Collect_Movers_practice: EDomainType.Movers,
    SocketType.Collect_Cleanup_practice: EDomainType.Movers,
    SocketType.Collect_Rescue: EDomainType.Rescue,
    SocketType.Collect_Rescue_practice: EDomainType.Rescue,
}

COLLECT_CANVAS_PAGELIST = {
    SocketType.Collect_Movers:
    [BoxPushReplayPage(EDomainType.Movers, True, MAP_MOVERS)],
    SocketType.Collect_Cleanup:
    [BoxPushReplayPage(EDomainType.Cleanup, True, MAP_CLEANUP)],
    SocketType.Collect_Movers_practice:
    [BoxPushReplayPage(EDomainType.Movers, False, MAP_MOVERS)],
    SocketType.Collect_Cleanup_practice:
    [BoxPushReplayPage(EDomainType.Cleanup, False, MAP_CLEANUP)],
    SocketType.Collect_Rescue: [RescueReplayPage(True, MAP_RESCUE_2)],
    SocketType.Collect_Rescue_practice: [RescueReplayPage(False, MAP_RESCUE_2)],
}

FEEDBACK_NAMESPACES = {
    EDomainType.Movers: "feedback_movers",
    EDomainType.Cleanup: "feedback_cleanup",
    EDomainType.Rescue: "feedback_rescue",
}

FEEDBACK_CANVAS_PAGELIST = {
    EDomainType.Movers:
    [BoxPushReplayPage(EDomainType.Movers, True, MAP_MOVERS)],
    EDomainType.Cleanup:
    [BoxPushReplayPage(EDomainType.Cleanup, True, MAP_CLEANUP)],
    EDomainType.Rescue: [RescueReplayPage(True, MAP_RESCUE_2)],
}
