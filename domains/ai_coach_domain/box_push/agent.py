import numpy as np
from ai_coach_core.utils.feature_utils import (get_gridworld_astar_distance,
                                               manhattan_distance)
from ai_coach_core.models.policy import CachedPolicyInterface
from ai_coach_domain.agent import SimulatorAgent, AIAgent_Abstract
from ai_coach_domain.box_push.agent_model import (BoxPushAM, BoxPushAM_Alone,
                                                  BoxPushAM_Together,
                                                  BoxPushAM_EmptyMind,
                                                  BoxPushAM_WebExp_Both)
from ai_coach_domain.box_push import conv_box_idx_2_state, BoxState, EventType


class BoxPushAIAgent_Host(AIAgent_Abstract):
  def __init__(self, policy_model: CachedPolicyInterface) -> None:
    super().__init__(policy_model, has_mind=False)

  def _create_agent_model(self,
                          policy_model: CachedPolicyInterface) -> BoxPushAM:
    return BoxPushAM_EmptyMind(policy_model)

  def init_latent(self, tup_states):
    'do nothing - a user should set the latent state manually'
    pass

  def update_mental_state(self, tup_cur_state, tup_actions, tup_nxt_state):
    'do nothing - a user should set the latent state manually'
    pass


class BoxPushAIAgent_Team1(AIAgent_Abstract):
  def __init__(self,
               policy_model: CachedPolicyInterface,
               has_mind: bool = True) -> None:
    super().__init__(policy_model, has_mind, agent_idx=0)

  def _create_agent_model(self,
                          policy_model: CachedPolicyInterface) -> BoxPushAM:
    return BoxPushAM_Together(agent_idx=self.agent_idx,
                              policy_model=policy_model)


class BoxPushAIAgent_Team2(AIAgent_Abstract):
  def __init__(self,
               policy_model: CachedPolicyInterface,
               has_mind: bool = True) -> None:
    super().__init__(policy_model, has_mind, agent_idx=1)

  def _create_agent_model(self,
                          policy_model: CachedPolicyInterface) -> BoxPushAM:
    return BoxPushAM_Together(agent_idx=self.agent_idx,
                              policy_model=policy_model)


class BoxPushAIAgent_WebExp_Both_A2(AIAgent_Abstract):
  def _create_agent_model(self,
                          policy_model: CachedPolicyInterface) -> BoxPushAM:
    return BoxPushAM_WebExp_Both(policy_model=policy_model)


class BoxPushAIAgent_Indv1(AIAgent_Abstract):
  def __init__(self,
               policy_model: CachedPolicyInterface,
               has_mind: bool = True) -> None:
    super().__init__(policy_model, has_mind, agent_idx=0)

  def _create_agent_model(self,
                          policy_model: CachedPolicyInterface) -> BoxPushAM:
    return BoxPushAM_Alone(self.agent_idx, policy_model)


class BoxPushAIAgent_Indv2(AIAgent_Abstract):
  def __init__(self,
               policy_model: CachedPolicyInterface,
               has_mind: bool = True) -> None:
    super().__init__(policy_model, has_mind, agent_idx=1)

  def _create_agent_model(self,
                          policy_model: CachedPolicyInterface) -> BoxPushAM:
    return BoxPushAM_Alone(self.agent_idx, policy_model)


class BoxPushSimpleAgent(SimulatorAgent):
  def __init__(self,
               agent_id,
               x_grid,
               y_grid,
               boxes,
               goals,
               walls,
               drops,
               v2=False) -> None:
    super().__init__(has_mind=False, has_policy=True)
    self.agent_id = agent_id
    self.x_grid = x_grid
    self.y_grid = y_grid
    self.boxes = boxes
    self.goals = goals
    self.walls = walls
    self.drops = drops
    self.v2 = v2

  def get_current_latent(self):
    return None

  def init_latent(self, tup_states):
    pass

  def get_action(self, tup_states):
    box_states, a1_pos, a2_pos = tup_states
    np_gridworld = np.zeros((self.x_grid, self.y_grid))
    for coord in self.walls:
      np_gridworld[coord] = 1

    a1_hold = False
    a2_hold = False
    for idx, bidx in enumerate(box_states):
      bstate = conv_box_idx_2_state(bidx, len(self.drops), len(self.goals))
      if bstate[0] == BoxState.WithAgent1:
        a1_hold = True
      elif bstate[0] == BoxState.WithAgent2:
        a2_hold = True
      elif bstate[0] == BoxState.WithBoth:
        a1_hold = True
        a2_hold = True

    my_pos = a1_pos
    my_hold = a1_hold
    if self.agent_id == 1:
      my_pos = a2_pos
      my_hold = a2_hold

    if my_hold:
      if my_pos in self.goals:
        return EventType.HOLD

      for idx, bidx in enumerate(box_states):
        bstate = conv_box_idx_2_state(bidx, len(self.drops), len(self.goals))
        if bstate[0] == BoxState.Original:
          np_gridworld[self.boxes[idx]] = 1
        elif bstate[0] in [BoxState.WithAgent1, BoxState.WithBoth]:
          np_gridworld[a1_pos] = 1
        elif bstate[0] == BoxState.WithAgent2:
          np_gridworld[a2_pos] = 1
        elif bstate[0] == BoxState.OnDropLoc:
          np_gridworld[self.drops[bstate[1]]] = 1

      path = get_gridworld_astar_distance(np_gridworld, my_pos, self.goals,
                                          manhattan_distance)
      if len(path) == 0:
        return EventType.STAY
      else:
        x = path[0][0] - my_pos[0]
        y = path[0][1] - my_pos[1]
        if x > 0:
          return EventType.RIGHT
        elif x < 0:
          return EventType.LEFT
        elif y > 0:
          return EventType.DOWN
        elif y < 0:
          return EventType.UP
        else:
          return EventType.STAY
    else:
      if self.v2:
        for coord in self.goals:
          np_gridworld[coord] = 1

      valid_boxes = []
      for idx, bidx in enumerate(box_states):
        bstate = conv_box_idx_2_state(bidx, len(self.drops), len(self.goals))
        if bstate[0] == BoxState.Original:
          valid_boxes.append(self.boxes[idx])
        elif bstate[0] == BoxState.OnDropLoc:
          valid_boxes.append(self.drops[bstate[1]])

      if len(valid_boxes) == 0:
        return EventType.STAY

      if my_pos in valid_boxes:
        return EventType.HOLD
      else:
        path = get_gridworld_astar_distance(np_gridworld, my_pos, valid_boxes,
                                            manhattan_distance)
        if len(path) == 0:
          return EventType.STAY
        else:
          x = path[0][0] - my_pos[0]
          y = path[0][1] - my_pos[1]
          if x > 0:
            return EventType.RIGHT
          elif x < 0:
            return EventType.LEFT
          elif y > 0:
            return EventType.DOWN
          elif y < 0:
            return EventType.UP
          else:
            return EventType.STAY

  def set_latent(self, latent):
    pass

  def set_action(self, action):
    pass

  def update_mental_state(self, tup_cur_state, tup_actions, tup_nxt_state):
    pass
