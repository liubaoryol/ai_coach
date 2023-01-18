from ai_coach_domain.box_push_v2.mdp import (MDP_Movers_Agent, MDP_Movers_Task,
                                             MDP_Cleanup_Agent,
                                             MDP_Cleanup_Task)
from ai_coach_domain.box_push_v3.transition import transition_mixed_noisy


class MDP_MoversV3_Agent(MDP_Movers_Agent):

  def _transition_impl(self, box_states, a1_pos, a2_pos, a1_action, a2_action):
    return transition_mixed_noisy(box_states, a1_pos, a2_pos, a1_action,
                                  a2_action, self.boxes, self.goals, self.walls,
                                  self.drops, self.x_grid, self.y_grid,
                                  self.box_types, self.a1_init, self.a2_init)


class MDP_CleanupV3_Agent(MDP_Cleanup_Agent):

  def _transition_impl(self, box_states, a1_pos, a2_pos, a1_action, a2_action):
    return transition_mixed_noisy(box_states, a1_pos, a2_pos, a1_action,
                                  a2_action, self.boxes, self.goals, self.walls,
                                  self.drops, self.x_grid, self.y_grid,
                                  self.box_types, self.a1_init, self.a2_init)


class MDP_MoversV3_Task(MDP_Movers_Task):

  def _transition_impl(self, box_states, a1_pos, a2_pos, a1_action, a2_action):
    return transition_mixed_noisy(box_states, a1_pos, a2_pos, a1_action,
                                  a2_action, self.boxes, self.goals, self.walls,
                                  self.drops, self.x_grid, self.y_grid,
                                  self.box_types, self.a1_init, self.a2_init)


class MDP_CleanupV3_Task(MDP_Cleanup_Task):

  def _transition_impl(self, box_states, a1_pos, a2_pos, a1_action, a2_action):
    return transition_mixed_noisy(box_states, a1_pos, a2_pos, a1_action,
                                  a2_action, self.boxes, self.goals, self.walls,
                                  self.drops, self.x_grid, self.y_grid,
                                  self.box_types, self.a1_init, self.a2_init)
