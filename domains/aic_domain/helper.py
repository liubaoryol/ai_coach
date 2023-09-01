import numpy as np
from typing import Sequence
from aic_domain.agent import AIAgent_Abstract


class TrueModelConverter:

  def __init__(self, agents: Sequence[AIAgent_Abstract], num_latents) -> None:
    self.agents = agents
    self.num_latents = num_latents

  def get_true_policy(self, agent_idx, latent_idx, state_idx):
    return self.agents[agent_idx].get_action_distribution(state_idx, latent_idx)

  def get_true_Tx_nxsas(self, agent_idx, latent_idx, state_idx,
                        tuple_action_idx, next_state_idx):
    return self.agents[agent_idx].get_next_latent_distribution(
        latent_idx, state_idx, tuple_action_idx, next_state_idx)

  def get_init_latent_dist(self, agent_idx, state_idx):
    return self.agents[agent_idx].get_initial_latent_distribution(state_idx)

  def true_Tx_for_var_infer(self, agent_idx, state_idx, joint_action_idx,
                            next_state_idx):
    np_Txx = np.zeros((self.num_latents, self.num_latents))
    for xidx in range(self.num_latents):
      np_Txx[xidx, :] = self.get_true_Tx_nxsas(agent_idx, xidx, state_idx,
                                               joint_action_idx, next_state_idx)

    return np_Txx
