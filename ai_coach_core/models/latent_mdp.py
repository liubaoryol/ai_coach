import abc
import numpy as np

from models.mdp import MDP
from utils.mdp_utils import StateSpace


class LatentMDP(MDP):
  """MDP with one latent state"""
  def __init__(self, fast_cache_mode: bool = False):
    super().__init__(fast_cache_mode)

    # Define latent state space.
    self.init_latentspace()
    self.init_latentspace_helper_vars()

  @abc.abstractmethod
  def init_latentspace(self):
    """Defines MDP latent state space. """
    self.latent_space = StateSpace()

  def init_latentspace_helper_vars(self):
    """Creates helper variables for the latent state space."""
    self.num_latents = self.latent_space.num_states

  @abc.abstractmethod
  def reward(self, state_idx: int, action_idx: int, latent_idx: int, *args,
             **kwargs) -> float:
    """Defines MDP reward function.

      Args:
        state_idx: Index of an MDP state.
        action_idx: Index of an MDP action.
        latent_idx: Index of an MDP latent.

      Returns:
        A scalar reward.
    """
    raise NotImplementedError

  @property
  def np_reward_model(self):
    """Returns reward model as a np ndarray."""
    # This code is largely duplicated with the parent method but more readable

    # If already computed, return the computed value.
    # This model does not change after the MDP is defined.
    if self._np_reward_model is not None:
      return self._np_reward_model

    # Else: Compute using the reward method.
    self._np_reward_model = np.zeros(
        (self.num_states, self.num_actions, self.num_latents))
    for state in range(self.num_states):
      for action in range(self.num_actions):
        for latent in range(self.num_latents):
          self._np_reward_model[state, action,
                                latent] = self.reward(state, action, latent)
    return self._np_reward_model

  # def init_state_distribution(self):
  #   'Return: numpy array with probability and states'
  #   raise NotImplementedError
