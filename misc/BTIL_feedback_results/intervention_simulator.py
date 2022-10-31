from typing import List, Union, Callable
import numpy as np
from ai_coach_core.intervention.feedback_strategy import InterventionAbstract
from ai_coach_core.latent_inference.decoding import forward_inference
from ai_coach_domain.box_push_v2 import BoxPushSimulatorV2
from ai_coach_domain.rescue.simulator import RescueSimulator
from ai_coach_domain.agent.partial_obs_agent import AIAgent_PartialObs


class InterventionSimulator:

  def __init__(self,
               game: Union[BoxPushSimulatorV2, RescueSimulator],
               list_np_policy: List[np.ndarray],
               list_np_tx: List[np.ndarray],
               intervention: InterventionAbstract,
               cb_get_prev_state_action: Callable,
               fix_illegal: bool = False,
               increase_step: bool = False) -> None:
    self.game = game
    self.list_np_policy = list_np_policy
    self.list_np_tx = list_np_tx
    self.intervention = intervention
    self.cb_get_prev_state_action = cb_get_prev_state_action
    self.fix_illegal = fix_illegal
    self.increase_step = increase_step

  def run_game(self, num_runs):
    list_score = []
    list_num_feedback = []
    for _ in range(num_runs):
      self.reset_game()
      while not self.game.is_finished():
        map_agent_2_action = self.game.get_joint_action()
        self.game.take_a_step(map_agent_2_action)
        self.intervene()
      list_score.append(self.game.get_score())
      list_num_feedback.append(self.num_feedback)

    return list_score, list_num_feedback

  def reset_game(self):
    self.game.reset_game()
    self.num_feedback = 0
    self.list_prev_np_x_dist = None

  def policy_nxsa(self, nidx, xidx, sidx, tuple_aidx):
    return self.list_np_policy[nidx][xidx, sidx, tuple_aidx[nidx]]

  def Tx_nxsasx(self, nidx, xidx, sidx, tuple_aidx, sidx_n, xidx_n):
    np_idx = tuple([xidx, *tuple_aidx, sidx_n])
    np_dist = self.list_np_tx[nidx][np_idx]

    # for illegal states or states that haven't appeared during the training,
    # we assume mental model was maintained.
    if self.fix_illegal:
      if np.all(np_dist == np_dist[0]):
        np_dist = np.zeros_like(np_dist)
        np_dist[xidx] = 1

    return np_dist[xidx_n]

  def init_latent_nxs(self, nidx, xidx, sidx):
    agent: AIAgent_PartialObs
    agent = self.game.agents[nidx]

    num_latents = agent.agent_model.policy_model.get_num_latent_states()
    return 1 / num_latents  # uniform

  def intervene(self) -> None:
    if self.intervention is None:
      return

    # inference
    # for agent_idx in range(self.game.get_num_agents()):
    #   if not isinstance(self.game.agents[agent_idx], AIAgent_PartialObs):
    #     raise RuntimeError("Invalid agent class")

    task_mdp = self.game.agent_1.agent_model.get_reference_mdp()

    tup_state_prev, tup_action_prev = self.cb_get_prev_state_action(
        self.game.history[-1])

    sidx = task_mdp.conv_sim_states_to_mdp_sidx(tup_state_prev)
    joint_action = []
    for agent_idx in range(self.game.get_num_agents()):
      aidx_i, = self.game.agents[
          agent_idx].agent_model.policy_model.conv_action_to_idx(
              (tup_action_prev[agent_idx], ))
      joint_action.append(aidx_i)

    sidx_n = task_mdp.conv_sim_states_to_mdp_sidx(
        tuple(self.game.get_state_for_each_agent(0)))
    list_state = [sidx, sidx_n]
    list_action = [tuple(joint_action)]

    num_lat = self.game.agent_1.agent_model.policy_model.get_num_latent_states()

    _, list_np_x_dist = forward_inference(list_state, list_action,
                                          self.game.get_num_agents(), num_lat,
                                          self.policy_nxsa, self.Tx_nxsasx,
                                          self.init_latent_nxs,
                                          self.list_prev_np_x_dist)
    self.list_prev_np_x_dist = list_np_x_dist

    # intervention
    feedback = self.intervention.get_intervention(self.list_prev_np_x_dist,
                                                  sidx_n)
    if feedback is None:
      return

    self.num_feedback += 1
    if self.increase_step:
      self.game.current_step += 1

    for agent_idx in range(self.game.get_num_agents()):
      if agent_idx in feedback:
        lat1 = feedback[agent_idx]
        self.game.agents[agent_idx].set_latent(
            self.game.agents[agent_idx].agent_model.policy_model.
            conv_idx_to_latent(lat1))
        np_int_x_dist = np.zeros(len(self.list_prev_np_x_dist[agent_idx]))
        np_int_x_dist[lat1] = 1.0
        self.list_prev_np_x_dist[agent_idx] = np_int_x_dist
