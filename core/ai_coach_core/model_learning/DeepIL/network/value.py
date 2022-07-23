import torch

from torch import nn
from .utils import build_mlp


class StateFunction(nn.Module):
  """
    Value function that takes s-x as input

    Parameters
    ----------
    state_size: int
        size of the state space
    latent_size: int
        size of the latent state space
    hidden_units: tuple
        hidden units of the value function
    hidden_activation: nn.Module
        hidden activation of the value function
    """
  def __init__(self,
               state_size: int,
               latent_size: int,
               hidden_units: tuple = (64, 64),
               hidden_activation: nn.Module = nn.Tanh()):
    super().__init__()
    self.net = build_mlp(input_dim=state_size + latent_size,
                         output_dim=1,
                         hidden_units=hidden_units,
                         hidden_activation=hidden_activation,
                         init=True)

  def forward(self, states: torch.Tensor,
              latents: torch.Tensor) -> torch.Tensor:
    """
        Return values of the states

        Parameters
        ----------
        states: torch.Tensor
            input states
        latents: torch.Tensor
            input latent states

        Returns
        -------
        values: torch.Tensor
            values of the s-x pairs
        """
    return self.net(torch.cat((states, latents), dim=-1))


class StateActionFunction(nn.Module):
  """
    Value function that takes s-x-a pairs as input

    Parameters
    ----------
    state_size: int
        size of the state space
    latent_size: int
        size of the latent state space
    action_size: int
        size of the action space
    hidden_units: tuple
        hidden units of the value function
    hidden_activation: nn.Module
        hidden activation of the value function
    """
  def __init__(self,
               state_size: int,
               latent_size: int,
               action_size: int,
               hidden_units: tuple = (100, 100),
               hidden_activation=nn.Tanh()):
    super().__init__()

    self.net = build_mlp(input_dim=state_size + latent_size + action_size,
                         output_dim=1,
                         hidden_units=hidden_units,
                         hidden_activation=hidden_activation)

  def forward(self, states: torch.Tensor, latents: torch.Tensor,
              actions: torch.Tensor) -> torch.Tensor:
    """
        Return values of the s-a pairs

        Parameters
        ----------
        states: torch.Tensor
            input states
        latents: torch.Tensor
            input latent states
        actions: torch.Tensor
            actions corresponding to the states

        Returns
        -------
        values: torch.Tensor
            values of the s-x-a pairs
        """
    return self.net(torch.cat((states, latents, actions), dim=-1))
