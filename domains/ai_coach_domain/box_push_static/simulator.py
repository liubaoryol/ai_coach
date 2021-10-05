from ai_coach_domain.box_push.simulator import BoxPushSimulator
from ai_coach_domain.box_push.helper import transition_alone_and_together


class StaticBoxPushSimulator_1(BoxPushSimulator):
  def __init__(self, id) -> None:
    super().__init__(id, transition_alone_and_together)
