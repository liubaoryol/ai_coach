from typing import Mapping, Sequence, Any
import random
from ai_coach_domain.box_push.mdp import (BoxPushTeamMDP_AlwaysTogether,
                                          BoxPushTeamMDP_AlwaysAlone,
                                          BoxPushAgentMDP_AlwaysAlone)
from ai_coach_domain.box_push.mdppolicy import (BoxPushPolicyTeamExp1,
                                                BoxPushPolicyIndvExp1)
from ai_coach_domain.box_push.agent import (BoxPushInteractiveAgent,
                                            BoxPushAIAgent_Indv2,
                                            BoxPushAIAgent_Team2,
                                            BoxPushAIAgent_Host)
from ai_coach_domain.box_push import EventType
from web_experiment.experiment1.page_exp1_game_base import (
    Exp1UserData, Exp1PageGame, get_valid_box_to_pickup,
    are_agent_states_changed)
from web_experiment.experiment1.helper import task_intervention
from web_experiment.define import EDomainType


class CanvasPageMoversTellAligned(Exp1PageGame):
  def __init__(self, game_map) -> None:
    super().__init__(True, False, game_map, False, False, 5)

    TEMPERATURE = 0.3
    mdp = BoxPushTeamMDP_AlwaysTogether(**game_map)
    self._TEAMMATE_POLICY = BoxPushPolicyTeamExp1(mdp, TEMPERATURE,
                                                  self._AGENT2)

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    agent1 = BoxPushInteractiveAgent()
    agent2 = BoxPushAIAgent_Host(self._TEAMMATE_POLICY)
    game = user_game_data.get_game_ref()
    game.set_autonomous_agent(agent1, agent2)

    user_game_data.data[Exp1UserData.SELECT] = False
    # ################ manually set latent state #############
    valid_boxes = get_valid_box_to_pickup(game)
    if len(valid_boxes) > 0:
      box_idx = random.choice(valid_boxes)
      game.event_input(self._AGENT1, EventType.SET_LATENT, ("pickup", box_idx))
      game.event_input(self._AGENT2, EventType.SET_LATENT, ("pickup", box_idx))

  def _get_instruction(self, user_game_data: Exp1UserData):
    if user_game_data.data[Exp1UserData.SELECT]:
      return (
          "Please select your current destination among the circled options. " +
          "It can be the same destination as you had previously selected.")
    else:
      return "Please choose your next action."

  def _on_action_taken(self, user_game_data: Exp1UserData,
                       dict_prev_game: Mapping[str, Any],
                       tuple_actions: Sequence[Any]):
    super()._on_action_taken(user_game_data, dict_prev_game, tuple_actions)

    # ################ manually set latent state #############
    game = user_game_data.get_game_ref()
    (a1_pos_changed, a2_pos_changed, a1_hold_changed, a2_hold_changed, a1_box,
     a2_box) = are_agent_states_changed(dict_prev_game, game.get_env_info())

    if a1_hold_changed:
      if a1_box >= 0:
        game.event_input(self._AGENT1, EventType.SET_LATENT, ("goal", 0))
        game.event_input(self._AGENT2, EventType.SET_LATENT, ("goal", 0))
      else:
        valid_boxes = get_valid_box_to_pickup(game)
        if len(valid_boxes) > 0:
          box_idx = random.choice(valid_boxes)
          game.event_input(self._AGENT1, EventType.SET_LATENT,
                           ("pickup", box_idx))
          game.event_input(self._AGENT2, EventType.SET_LATENT,
                           ("pickup", box_idx))


class CanvasPageMoversUserRandom(Exp1PageGame):
  def __init__(self, game_map, intervention: bool) -> None:
    super().__init__(True, False, game_map, False, False, 5)

    TEMPERATURE = 0.3
    mdp = BoxPushTeamMDP_AlwaysTogether(**game_map)
    self._TEAMMATE_POLICY = BoxPushPolicyTeamExp1(mdp, TEMPERATURE,
                                                  self._AGENT2)
    self._INTERVENTION = intervention

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    game = user_game_data.get_game_ref()
    agent1 = BoxPushInteractiveAgent()
    agent2 = BoxPushAIAgent_Team2(self._TEAMMATE_POLICY)
    game.set_autonomous_agent(agent1, agent2)

    user_game_data.data[Exp1UserData.SELECT] = False

  def _on_action_taken(self, user_game_data: Exp1UserData,
                       dict_prev_game: Mapping[str, Any],
                       tuple_actions: Sequence[Any]):
    super()._on_action_taken(user_game_data, dict_prev_game, tuple_actions)

    if self._INTERVENTION:
      game = user_game_data.get_game_ref()
      domain = EDomainType.Movers if self._IS_MOVERS else EDomainType.Cleanup
      task_intervention(game.history, game, domain)


class CanvasPageCleanUpTellAligned(Exp1PageGame):
  def __init__(self, game_map) -> None:
    super().__init__(False, False, game_map, False, False, 5)

    TEMPERATURE = 0.3
    mdp = BoxPushAgentMDP_AlwaysAlone(**game_map)
    task_mdp = BoxPushTeamMDP_AlwaysAlone(**game_map)
    self._TEAMMATE_POLICY = BoxPushPolicyIndvExp1(task_mdp, mdp, TEMPERATURE,
                                                  self._AGENT2)

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    game = user_game_data.get_game_ref()
    agent1 = BoxPushInteractiveAgent()
    agent2 = BoxPushAIAgent_Host(self._TEAMMATE_POLICY)
    game.set_autonomous_agent(agent1, agent2)

    # ################ manually set latent state #############
    valid_boxes = get_valid_box_to_pickup(game)
    if len(valid_boxes) > 0:
      box_idx = random.choice(valid_boxes)
      game.event_input(self._AGENT1, EventType.SET_LATENT, ("pickup", box_idx))
      game.event_input(self._AGENT2, EventType.SET_LATENT, ("pickup", box_idx))

  def _get_instruction(self, user_game_data: Exp1UserData):
    if user_game_data.data[Exp1UserData.SELECT]:
      return (
          "Please select your current destination among the circled options. " +
          "It can be the same destination as you had previously selected.")
    else:
      return "Please choose your next action."

  def _on_action_taken(self, user_game_data: Exp1UserData,
                       dict_prev_game: Mapping[str, Any],
                       tuple_actions: Sequence[Any]):
    super()._on_action_taken(user_game_data, dict_prev_game, tuple_actions)

    game = user_game_data.get_game_ref()
    # ################ manually set latent state #############
    (a1_pos_changed, a2_pos_changed, a1_hold_changed, a2_hold_changed, a1_box,
     a2_box) = are_agent_states_changed(dict_prev_game, game.get_env_info())

    a1_latent_prev = game.agent_1.get_current_latent()
    a2_latent_prev = game.agent_2.get_current_latent()

    a1_pickup = a1_hold_changed and (a1_box >= 0)
    a1_drop = a1_hold_changed and not (a1_box >= 0)
    a2_pickup = a2_hold_changed and (a2_box >= 0)
    a2_drop = a2_hold_changed and not (a2_box >= 0)

    if a1_pickup and a2_pickup:
      game.event_input(self._AGENT1, EventType.SET_LATENT, ("goal", 0))
      game.event_input(self._AGENT2, EventType.SET_LATENT, ("goal", 0))

    elif a1_pickup and a2_drop:
      game.event_input(self._AGENT1, EventType.SET_LATENT, ("goal", 0))
      valid_boxes = get_valid_box_to_pickup(game)
      if len(valid_boxes) > 0:
        box_idx = random.choice(valid_boxes)
        game.event_input(self._AGENT2, EventType.SET_LATENT,
                         ("pickup", box_idx))
      else:
        game.event_input(self._AGENT2, EventType.SET_LATENT, ("pickup", a1_box))

    elif a1_drop and a2_pickup:
      valid_boxes = get_valid_box_to_pickup(game)
      if len(valid_boxes) > 0:
        box_idx = random.choice(valid_boxes)
        game.event_input(self._AGENT1, EventType.SET_LATENT,
                         ("pickup", box_idx))
      else:
        game.event_input(self._AGENT1, EventType.SET_LATENT, ("pickup", a2_box))
      game.event_input(self._AGENT2, EventType.SET_LATENT, ("goal", 0))

    elif a1_drop and a2_drop:  # TODO: agent1 follows agent2 latent
      valid_boxes = get_valid_box_to_pickup(game)
      if len(valid_boxes) > 0:
        box_idx = random.choice(valid_boxes)
        game.event_input(self._AGENT1, EventType.SET_LATENT,
                         ("pickup", box_idx))
        game.event_input(self._AGENT2, EventType.SET_LATENT,
                         ("pickup", box_idx))

    elif a1_pickup:
      game.event_input(self._AGENT1, EventType.SET_LATENT, ("goal", 0))
      # a2 has no box and was targetting the same box that a1 picked up
      # --> set to another box
      if a2_box < 0 and a1_box == a2_latent_prev[1]:
        valid_boxes = get_valid_box_to_pickup(game)
        if len(valid_boxes) > 0:
          box_idx = random.choice(valid_boxes)
          game.event_input(self._AGENT2, EventType.SET_LATENT,
                           ("pickup", box_idx))

    elif a1_drop:
      # a2 has a box --> set to another box
      if a2_box >= 0:
        valid_boxes = get_valid_box_to_pickup(game)
        if len(valid_boxes) > 0:
          box_idx = random.choice(valid_boxes)
          game.event_input(self._AGENT1, EventType.SET_LATENT,
                           ("pickup", box_idx))
        else:
          game.event_input(self._AGENT1, EventType.SET_LATENT,
                           ("pickup", a2_box))
      # a2 has no box --> set to the same box a2 is aiming for
      else:
        game.event_input(self._AGENT1, EventType.SET_LATENT,
                         ("pickup", a2_latent_prev[1]))
    elif a2_pickup:
      game.event_input(self._AGENT2, EventType.SET_LATENT, ("goal", 0))
      # a1 has no box and was targetting the same box that a2 just picked up
      # --> set to another box
      if a1_box < 0 and a2_box == a1_latent_prev[1]:
        valid_boxes = get_valid_box_to_pickup(game)
        if len(valid_boxes) > 0:
          box_idx = random.choice(valid_boxes)
          game.event_input(self._AGENT1, EventType.SET_LATENT,
                           ("pickup", box_idx))

    elif a2_drop:
      # a1 has a box --> set to another box
      if a1_box >= 0:
        valid_boxes = get_valid_box_to_pickup(game)
        if len(valid_boxes) > 0:
          box_idx = random.choice(valid_boxes)
          game.event_input(self._AGENT2, EventType.SET_LATENT,
                           ("pickup", box_idx))
        else:
          game.event_input(self._AGENT2, EventType.SET_LATENT,
                           ("pickup", a1_box))
      # a1 has no box --> set to the same box a1 is aiming for
      else:
        game.event_input(self._AGENT2, EventType.SET_LATENT,
                         ("pickup", a1_latent_prev[1]))


class CanvasPageCleanUpTellRandom(Exp1PageGame):
  def __init__(self, game_map) -> None:
    super().__init__(False, False, game_map, False, False, 5)

    TEMPERATURE = 0.3
    mdp = BoxPushAgentMDP_AlwaysAlone(**game_map)
    task_mdp = BoxPushTeamMDP_AlwaysAlone(**game_map)
    self._TEAMMATE_POLICY = BoxPushPolicyIndvExp1(task_mdp, mdp, TEMPERATURE,
                                                  self._AGENT2)

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    game = user_game_data.get_game_ref()
    agent1 = BoxPushInteractiveAgent()
    agent2 = BoxPushAIAgent_Host(self._TEAMMATE_POLICY)
    game.set_autonomous_agent(agent1, agent2)

    # ################ manually set latent state #############
    valid_boxes = get_valid_box_to_pickup(game)
    if len(valid_boxes) > 0:
      box_idx = random.choice(valid_boxes)
      game.event_input(self._AGENT1, EventType.SET_LATENT, ("pickup", box_idx))
      valid_boxes.remove(box_idx)

    if len(valid_boxes) > 0:
      box_idx2 = random.choice(valid_boxes)
      game.event_input(self._AGENT2, EventType.SET_LATENT, ("pickup", box_idx2))

  def _get_instruction(self, user_game_data: Exp1UserData):
    if user_game_data.data[Exp1UserData.SELECT]:
      return (
          "Please select your current destination among the circled options. " +
          "It can be the same destination as you had previously selected.")
    else:
      return "Please choose your next action."

  def _on_action_taken(self, user_game_data: Exp1UserData,
                       dict_prev_game: Mapping[str, Any],
                       tuple_actions: Sequence[Any]):
    super()._on_action_taken(user_game_data, dict_prev_game, tuple_actions)

    game = user_game_data.get_game_ref()
    # ################ manually set latent state #############
    (a1_pos_changed, a2_pos_changed, a1_hold_changed, a2_hold_changed, a1_box,
     a2_box) = are_agent_states_changed(dict_prev_game, game.get_env_info())

    # ################ manually set latent state #############
    a1_latent_prev = game.agent_1.get_current_latent()
    a2_latent_prev = game.agent_2.get_current_latent()

    a1_pickup = a1_hold_changed and (a1_box >= 0)
    a1_drop = a1_hold_changed and not (a1_box >= 0)
    a2_pickup = a2_hold_changed and (a2_box >= 0)
    a2_drop = a2_hold_changed and not (a2_box >= 0)

    if a1_pickup and a2_pickup:
      game.event_input(self._AGENT1, EventType.SET_LATENT, ("goal", 0))
      game.event_input(self._AGENT2, EventType.SET_LATENT, ("goal", 0))

    elif a1_pickup and a2_drop:
      game.event_input(self._AGENT1, EventType.SET_LATENT, ("goal", 0))
      valid_boxes = get_valid_box_to_pickup(game)
      if len(valid_boxes) > 0:
        box_idx = random.choice(valid_boxes)
        game.event_input(self._AGENT2, EventType.SET_LATENT,
                         ("pickup", box_idx))
      else:
        game.event_input(self._AGENT2, EventType.SET_LATENT, ("pickup", a1_box))

    elif a1_drop and a2_pickup:
      valid_boxes = get_valid_box_to_pickup(game)
      if len(valid_boxes) > 0:
        box_idx = random.choice(valid_boxes)
        game.event_input(self._AGENT1, EventType.SET_LATENT,
                         ("pickup", box_idx))
      else:
        game.event_input(self._AGENT1, EventType.SET_LATENT, ("pickup", a2_box))
      game.event_input(self._AGENT2, EventType.SET_LATENT, ("goal", 0))

    elif a1_drop and a2_drop:
      valid_boxes = get_valid_box_to_pickup(game)
      if len(valid_boxes) == 1:
        box_idx = random.choice(valid_boxes)
        game.event_input(self._AGENT1, EventType.SET_LATENT,
                         ("pickup", box_idx))
        game.event_input(self._AGENT2, EventType.SET_LATENT,
                         ("pickup", box_idx))
      elif len(valid_boxes) > 1:
        box_idx = random.choice(valid_boxes)
        game.event_input(self._AGENT1, EventType.SET_LATENT,
                         ("pickup", box_idx))
        valid_boxes.remove(box_idx)
        box_idx2 = random.choice(valid_boxes)
        game.event_input(self._AGENT2, EventType.SET_LATENT,
                         ("pickup", box_idx2))

    elif a1_pickup:
      game.event_input(self._AGENT1, EventType.SET_LATENT, ("goal", 0))
      # a2 has no box and was targetting the same box that a1 picked up
      # --> set to another box
      if a2_box < 0 and a1_box == a2_latent_prev[1]:
        valid_boxes = get_valid_box_to_pickup(game)
        if len(valid_boxes) > 0:
          box_idx = random.choice(valid_boxes)
          game.event_input(self._AGENT2, EventType.SET_LATENT,
                           ("pickup", box_idx))

    elif a1_drop:
      valid_boxes = get_valid_box_to_pickup(game)
      # a2 has a box --> set to another box
      if a2_box >= 0:
        if len(valid_boxes) > 0:
          box_idx = random.choice(valid_boxes)
          game.event_input(self._AGENT1, EventType.SET_LATENT,
                           ("pickup", box_idx))
        else:
          game.event_input(self._AGENT1, EventType.SET_LATENT,
                           ("pickup", a2_box))
      # a2 has no box --> set to the box different from one a2 is aiming for
      else:
        if a2_latent_prev[1] in valid_boxes:
          valid_boxes.remove(a2_latent_prev[1])
        if len(valid_boxes) > 0:
          box_idx = random.choice(valid_boxes)
          game.event_input(self._AGENT1, EventType.SET_LATENT,
                           ("pickup", box_idx))
        else:
          game.event_input(self._AGENT1, EventType.SET_LATENT,
                           ("pickup", a2_latent_prev[1]))

    elif a2_pickup:
      game.event_input(self._AGENT2, EventType.SET_LATENT, ("goal", 0))
      # a1 has no box and was targetting the same box that a2 just picked up
      # --> set to another box
      if a1_box < 0 and a2_box == a1_latent_prev[1]:
        valid_boxes = get_valid_box_to_pickup(game)
        if len(valid_boxes) > 0:
          box_idx = random.choice(valid_boxes)
          game.event_input(self._AGENT1, EventType.SET_LATENT,
                           ("pickup", box_idx))

    elif a2_drop:
      valid_boxes = get_valid_box_to_pickup(game)
      # a1 has a box --> set to another box
      if a1_box >= 0:
        if len(valid_boxes) > 0:
          box_idx = random.choice(valid_boxes)
          game.event_input(self._AGENT2, EventType.SET_LATENT,
                           ("pickup", box_idx))
        else:
          game.event_input(self._AGENT2, EventType.SET_LATENT,
                           ("pickup", a1_box))
      # a1 has no box --> set to the box different from one a1 is aiming for
      else:
        if a1_latent_prev[1] in valid_boxes:
          valid_boxes.remove(a1_latent_prev[1])
        if len(valid_boxes) > 0:
          box_idx = random.choice(valid_boxes)
          game.event_input(self._AGENT2, EventType.SET_LATENT,
                           ("pickup", box_idx))
        else:
          game.event_input(self._AGENT2, EventType.SET_LATENT,
                           ("pickup", a1_latent_prev[1]))


class CanvasPageCleanUpUserRandom(Exp1PageGame):
  def __init__(self, game_map, intervention: bool) -> None:
    super().__init__(False, False, game_map, False, False, 5)

    TEMPERATURE = 0.3
    mdp = BoxPushAgentMDP_AlwaysAlone(**game_map)
    task_mdp = BoxPushTeamMDP_AlwaysAlone(**game_map)
    self._TEAMMATE_POLICY = BoxPushPolicyIndvExp1(task_mdp, mdp, TEMPERATURE,
                                                  self._AGENT2)
    self._INTERVENTION = intervention

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    game = user_game_data.get_game_ref()
    agent1 = BoxPushInteractiveAgent()
    agent2 = BoxPushAIAgent_Indv2(self._TEAMMATE_POLICY)
    game.set_autonomous_agent(agent1, agent2)

    user_game_data.data[Exp1UserData.SELECT] = False

  def _on_action_taken(self, user_game_data: Exp1UserData,
                       dict_prev_game: Mapping[str, Any],
                       tuple_actions: Sequence[Any]):
    super()._on_action_taken(user_game_data, dict_prev_game, tuple_actions)

    if self._INTERVENTION:
      game = user_game_data.get_game_ref()
      domain = EDomainType.Movers if self._IS_MOVERS else EDomainType.Cleanup
      task_intervention(game.history, game, domain)
