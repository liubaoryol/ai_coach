import abc
import numpy as np
from ai_coach_core.utils.feature_utils import (get_gridworld_astar_distance,
                                               manhattan_distance)
from ai_coach_domain.box_push.mdpagent import (BoxPushMDPAgent,
                                               BoxPushMDPAgent_Alone,
                                               BoxPushMDPAgent_Together,
                                               BoxPushMDPAgent_EmptyMind,
                                               BoxPushMDPAgent_WebExp_Both)
from ai_coach_domain.box_push.mdppolicy import BoxPushPolicyInterface
from ai_coach_domain.box_push.mdp import (BoxPushMDP,
                                          get_agent_switched_boxstates)
from ai_coach_domain.box_push.helper import (conv_box_idx_2_state, BoxState,
                                             EventType)


class BoxPushSimulatorAgent:
  __metaclass__ = abc.ABCMeta

  def __init__(self, has_mind: bool, has_policy: bool) -> None:
    self.bool_mind = has_mind
    self.bool_policy = has_policy

  def has_mind(self):
    return self.bool_mind

  def has_policy(self):
    return self.bool_policy

  @abc.abstractmethod
  def init_latent(self, box_states, a1_pos, a2_pos):
    raise NotImplementedError

  @abc.abstractmethod
  def get_current_latent(self):
    raise NotImplementedError

  @abc.abstractmethod
  def get_action(self, box_states, a1_pos, a2_pos):
    raise NotImplementedError

  @abc.abstractmethod
  def set_latent(self, latent):
    'to set latent manually'
    raise NotImplementedError

  @abc.abstractmethod
  def set_action(self, action):
    'to set what to do as next actions manually'
    raise NotImplementedError

  @abc.abstractmethod
  def update_mental_state(self, tup_cur_state, tup_actions, tup_nxt_state):
    raise NotImplementedError


class BoxPushInteractiveAgent(BoxPushSimulatorAgent):
  def __init__(self, start_latent=None) -> None:
    super().__init__(has_mind=False, has_policy=False)
    self.current_latent = None
    self.start_latent = start_latent
    self.action_queue = []

  def init_latent(self, box_states, a1_pos, a2_pos):
    self.current_latent = self.start_latent

  def get_current_latent(self):
    return self.current_latent

  def get_action(self, box_states, a1_pos, a2_pos):
    if len(self.action_queue) == 0:
      return None

    return self.action_queue.pop()

  def set_latent(self, latent):
    self.current_latent = latent

  def set_action(self, action):
    self.action_queue = [action]

  def update_mental_state(self, tup_cur_state, tup_actions, tup_nxt_state):
    'do nothing'
    pass


class BoxPushAIAgent_Abstract(BoxPushSimulatorAgent):
  def __init__(self,
               policy_model: BoxPushPolicyInterface,
               use_flipped_state_space: bool = False,
               has_mind: bool = True) -> None:
    super().__init__(has_mind=has_mind, has_policy=True)
    self.agent_model = self._create_agent_model(policy_model)
    self.use_flipped_state_space = use_flipped_state_space
    self.manual_action = None

  @abc.abstractmethod
  def _create_agent_model(
      self, policy_model: BoxPushPolicyInterface) -> BoxPushMDPAgent:
    'Should be implemented at inherited method'
    raise NotImplementedError

  def init_latent(self, box_states, a1_pos, a2_pos):
    sidx = self._conv_sim_states_to_mdp_sidx(box_states, a1_pos, a2_pos)

    self.agent_model.set_init_mental_state_idx(sidx)

  def get_current_latent(self):
    if self.agent_model.is_current_latent_valid():
      return self._conv_idx_to_latent(self.agent_model.current_latent)
    else:
      return None

  def get_action(self, box_states, a1_pos, a2_pos):
    if self.manual_action is not None:
      next_action = self.manual_action
      self.manual_action = None
      return next_action

    sidx = self._conv_sim_states_to_mdp_sidx(box_states, a1_pos, a2_pos)
    tup_aidx = self.agent_model.get_action_idx(sidx)
    return self.agent_model.policy_model.conv_idx_to_action(tup_aidx)[0]

  def update_mental_state(self, tup_cur_state, tup_actions, tup_nxt_state):
    'tup_actions: tuple of actions'

    bstate_cur, pos1_cur, pos2_cur = tup_cur_state
    bstate_nxt, pos1_nxt, pos2_nxt = tup_nxt_state

    sidx_cur = self._conv_sim_states_to_mdp_sidx(bstate_cur, pos1_cur, pos2_cur)
    sidx_nxt = self._conv_sim_states_to_mdp_sidx(bstate_nxt, pos1_nxt, pos2_nxt)

    tup_aidx = self.agent_model.policy_model.conv_action_to_idx(tup_actions)

    self.agent_model.update_mental_state_idx(sidx_cur, tup_aidx, sidx_nxt)

  def set_latent(self, latent):
    xidx = self._conv_latent_to_idx(latent)
    self.agent_model.set_init_mental_state_idx(None, xidx)

  def set_action(self, action):
    self.manual_action = action

  def policy_from_task_mdp_POV(self, state_idx, latent_idx):
    mdp = self.agent_model.get_reference_mdp()  # type: BoxPushMDP
    pos1, pos2, bstate = mdp.conv_mdp_sidx_to_sim_states(state_idx)
    sidx = self._conv_sim_states_to_mdp_sidx(bstate, pos1, pos2)
    return self.agent_model.policy_model.policy(sidx, latent_idx)

  def transition_model_from_task_mdp_POV(self, latent_idx, state_idx,
                                         tuple_action_idx, next_state_idx):
    mdp = self.agent_model.get_reference_mdp()  # type: BoxPushMDP
    pos1, pos2, bstate = mdp.conv_mdp_sidx_to_sim_states(state_idx)
    sidx = self._conv_sim_states_to_mdp_sidx(bstate, pos1, pos2)
    return self.agent_model.transition_mental_state(latent_idx, sidx,
                                                    tuple_action_idx,
                                                    next_state_idx)

  def init_latent_dist_from_task_mdp_POV(self, state_idx):
    mdp = self.agent_model.get_reference_mdp()  # type: BoxPushMDP
    pos1, pos2, bstate = mdp.conv_mdp_sidx_to_sim_states(state_idx)
    sidx = self._conv_sim_states_to_mdp_sidx(bstate, pos1, pos2)
    return self.agent_model.initial_mental_distribution(sidx)

  def _conv_sim_states_to_mdp_sidx(self, box_states, a1_pos, a2_pos):
    mdp = self.agent_model.get_reference_mdp()  # type: BoxPushMDP

    pos_1 = a1_pos
    pos_2 = a2_pos
    bstate = box_states
    if self.use_flipped_state_space:
      pos_1 = a2_pos
      pos_2 = a1_pos
      bstate = get_agent_switched_boxstates(box_states, len(mdp.drops),
                                            len(mdp.goals))
    sidx = mdp.conv_sim_states_to_mdp_sidx(pos_1, pos_2, bstate)
    return sidx

  def _conv_idx_to_latent(self, latent_idx):
    mdp = self.agent_model.get_reference_mdp()
    return mdp.latent_space.idx_to_state[latent_idx]

  def _conv_latent_to_idx(self, latent):
    mdp = self.agent_model.get_reference_mdp()
    return mdp.latent_space.state_to_idx[latent]


class BoxPushAIAgent_Host(BoxPushAIAgent_Abstract):
  def __init__(self,
               policy_model: BoxPushPolicyInterface,
               use_flipped_state_space=False) -> None:
    super().__init__(policy_model,
                     use_flipped_state_space=use_flipped_state_space,
                     has_mind=False)

  def _create_agent_model(
      self, policy_model: BoxPushPolicyInterface) -> BoxPushMDPAgent:
    return BoxPushMDPAgent_EmptyMind(policy_model)

  def init_latent(self, box_states, a1_pos, a2_pos):
    'do nothing - a user should set the latent state manually'
    pass

  def update_mental_state(self, tup_cur_state, tup_actions, tup_nxt_state):
    'do nothing - a user should set the latent state manually'
    pass


class BoxPushAIAgent_Team1(BoxPushAIAgent_Abstract):
  def __init__(self, policy_model: BoxPushPolicyInterface) -> None:
    super().__init__(policy_model, use_flipped_state_space=False)

  def _create_agent_model(
      self, policy_model: BoxPushPolicyInterface) -> BoxPushMDPAgent:
    AGENT1_IDX = 0
    return BoxPushMDPAgent_Together(agent_idx=AGENT1_IDX,
                                    policy_model=policy_model)


class BoxPushAIAgent_Team2(BoxPushAIAgent_Abstract):
  def __init__(self, policy_model: BoxPushPolicyInterface) -> None:
    super().__init__(policy_model, use_flipped_state_space=False)

  def _create_agent_model(
      self, policy_model: BoxPushPolicyInterface) -> BoxPushMDPAgent:
    AGENT2_IDX = 1
    return BoxPushMDPAgent_Together(agent_idx=AGENT2_IDX,
                                    policy_model=policy_model)


class BoxPushAIAgent_WebExp_Both_A2(BoxPushAIAgent_Abstract):
  def __init__(self, policy_model: BoxPushPolicyInterface) -> None:
    super().__init__(policy_model, use_flipped_state_space=False)

  def _create_agent_model(
      self, policy_model: BoxPushPolicyInterface) -> BoxPushMDPAgent:
    return BoxPushMDPAgent_WebExp_Both(policy_model=policy_model)


class BoxPushAIAgent_Indv1(BoxPushAIAgent_Abstract):
  def __init__(self, policy_model: BoxPushPolicyInterface) -> None:
    super().__init__(policy_model, use_flipped_state_space=False)

  def _create_agent_model(
      self, policy_model: BoxPushPolicyInterface) -> BoxPushMDPAgent:
    return BoxPushMDPAgent_Alone(policy_model)


class BoxPushAIAgent_Indv2(BoxPushAIAgent_Abstract):
  def __init__(self, policy_model: BoxPushPolicyInterface) -> None:
    super().__init__(policy_model, use_flipped_state_space=True)

  def _create_agent_model(
      self, policy_model: BoxPushPolicyInterface) -> BoxPushMDPAgent:
    return BoxPushMDPAgent_Alone(policy_model)


class BoxPushSimpleAgent(BoxPushSimulatorAgent):
  def __init__(self, agent_id, x_grid, y_grid, boxes, goals, walls,
               drops) -> None:
    super().__init__(has_mind=False, has_policy=True)
    self.agent_id = agent_id
    self.x_grid = x_grid
    self.y_grid = y_grid
    self.boxes = boxes
    self.goals = goals
    self.walls = walls
    self.drops = drops

  def get_current_latent(self):
    return None

  def init_latent(self, box_states, a1_pos, a2_pos):
    pass

  def get_action(self, box_states, a1_pos, a2_pos):
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
