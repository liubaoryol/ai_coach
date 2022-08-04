import torch
import torch.nn.functional as F

from torch import nn
from .utils import build_mlp


class GAILDiscrim(nn.Module):
  """
    Discriminator used by GAIL, which takes s-a pair as input and output
    the probability that the s-a pair is sampled from demonstrations

    Parameters
    ----------
    state_size: int
        size of the state space
    action_size: int
        size of the action space
    hidden_units: tuple
        hidden units of the discriminator
    hidden_activation: nn.Module
        hidden activation of the discriminator
    """
  def __init__(self,
               state_size: int,
               action_size: int,
               hidden_units: tuple = (100, 100),
               hidden_activation: nn.Module = nn.Tanh()):
    super().__init__()
    self.net = build_mlp(input_dim=state_size + action_size,
                         output_dim=1,
                         hidden_units=hidden_units,
                         hidden_activation=hidden_activation)

  def forward(self, states: torch.Tensor,
              actions: torch.Tensor) -> torch.Tensor:
    """
        Run discriminator

        Parameters
        ----------
        states: torch.Tensor
            input states
        actions: torch.Tensor
            actions corresponding to the states

        Returns
        -------
        result: torch.Tensor
            probability that this s-a pair belongs to demonstrations
        """
    return self.net(torch.cat([states, actions], dim=-1))

  def calculate_reward(self, states: torch.Tensor,
                       actions: torch.Tensor) -> torch.Tensor:
    """
        Calculate reward using GAIL's learned reward signal log(D)

        Parameters
        ----------
        states: torch.Tensor
            input states
        actions: torch.Tensor
            actions corresponding to the states

        Returns
        -------
        rewards: torch.Tensor
            reward signal
        """
    # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D)].
    with torch.no_grad():
      return -F.logsigmoid(-self.forward(states, actions))
