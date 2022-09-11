from enum import Enum
from typing import Mapping, Sequence
from ai_coach_domain.box_push_v2.maps import MAP_CLEANUP, MAP_MOVERS
from web_experiment.define import EDomainType
from web_experiment.exp_common.page_base import ExperimentPageBase
import web_experiment.exp_common.page_exp1_common as pgc
from web_experiment.demo.pages import BoxPushV2Demo, RescueDemo
from ai_coach_domain.rescue.maps import MAP_RESCUE_2


class E_SessionName(Enum):
  Movers_full_dcol = 0
  Movers_partial_dcol = 1
  Cleanup_full_dcol = 2
  Cleanup_partial_dcol = 3
  Rescue_full_dcol = 4
  Rescue_partial_dcol = 5


SESSION_TITLE = {
    E_SessionName.Movers_full_dcol: 'Movers - Fully Observable',
    E_SessionName.Movers_partial_dcol: 'Movers - Partially Observable',
    E_SessionName.Cleanup_full_dcol: 'Cleanup - Fully Observable',
    E_SessionName.Cleanup_partial_dcol: 'Cleanup - Partially Observable',
    E_SessionName.Rescue_full_dcol: 'Rescue - Fully Observable',
    E_SessionName.Rescue_partial_dcol: 'Rescue - Partially Observable',
}

PAGE_LIST_MOVERS_FULL_OBS = [
    pgc.CanvasPageStart(EDomainType.Movers),
    BoxPushV2Demo(EDomainType.Movers, MAP_MOVERS, False),
]
PAGE_LIST_MOVERS = [
    pgc.CanvasPageStart(EDomainType.Movers),
    BoxPushV2Demo(EDomainType.Movers, MAP_MOVERS, True),
]
PAGE_LIST_CLEANUP_FULL_OBS = [
    pgc.CanvasPageStart(EDomainType.Cleanup),
    BoxPushV2Demo(EDomainType.Cleanup, MAP_CLEANUP, False),
]
PAGE_LIST_CLEANUP = [
    pgc.CanvasPageStart(EDomainType.Cleanup),
    BoxPushV2Demo(EDomainType.Cleanup, MAP_CLEANUP, True),
]

PAGE_LIST_RESCUE_FULL_OBS = [
    pgc.CanvasPageStart(EDomainType.Rescue),
    RescueDemo(MAP_RESCUE_2, False),
]

PAGE_LIST_RESCUE = [
    pgc.CanvasPageStart(EDomainType.Rescue),
    RescueDemo(MAP_RESCUE_2, True),
]

GAMEPAGES = {
    E_SessionName.Movers_full_dcol: PAGE_LIST_MOVERS_FULL_OBS,
    E_SessionName.Movers_partial_dcol: PAGE_LIST_MOVERS,
    E_SessionName.Cleanup_full_dcol: PAGE_LIST_CLEANUP_FULL_OBS,
    E_SessionName.Cleanup_partial_dcol: PAGE_LIST_CLEANUP,
    E_SessionName.Rescue_full_dcol: PAGE_LIST_RESCUE_FULL_OBS,
    E_SessionName.Rescue_partial_dcol: PAGE_LIST_RESCUE
}  # type: Mapping[E_SessionName, Sequence[ExperimentPageBase]]
