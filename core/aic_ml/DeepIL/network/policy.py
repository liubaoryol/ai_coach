import torch
import abc

from torch import nn
from torch.distributions import Categorical
from typing import Tuple
from .utils import build_mlp, reparameterize, evaluate_log_p


class AbstractPolicy(nn.Module):
  """
    Stochastic policy \pi(a|s, x)

    Parameters
    ----------
    state_size: np.array
        size of the state space
    latent_size: np.array
        size of latent states
    action_size: np.array
        size of the action space
    hidden_units: tuple
        hidden units of the policy
    hidden_activation: nn.Module
        hidden activation of the policy
    """
  def __init__(self,
               state_size: int,
               latent_size: int,
               action_size: int,
               hidden_units: tuple = (64, 64),
               hidden_activation: nn.Module = nn.Tanh()):
    super().__init__()
    self.net = build_mlp(input_dim=state_size + latent_size,
                         output_dim=action_size,
                         hidden_units=hidden_units,
                         hidden_activation=hidden_activation,
                         init=True)

  @abc.abstractmethod
  def forward(self, states: torch.Tensor,
              latents: torch.Tensor) -> torch.Tensor:
    pass

  @abc.abstractmethod
  def exploit(self, states: torch.Tensor,
              latents: torch.Tensor) -> torch.Tensor:
    pass

  @abc.abstractmethod
  def sample(self, states: torch.Tensor,
             latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pass

  @abc.abstractmethod
  def evaluate_log_pi(self, states: torch.Tensor, latents: torch.Tensor,
                      actions: torch.Tensor) -> torch.Tensor:
    pass


class ContinousPolicy(AbstractPolicy):
  def __init__(self,
               state_size: int,
               latent_size: int,
               action_size: int,
               hidden_units: tuple = (64, 64),
               hidden_activation: nn.Module = nn.Tanh()):
    super().__init__(state_size, latent_size, action_size, hidden_units,
                     hidden_activation)
    self.log_stds = nn.Parameter(torch.zeros(1, action_size))

  def forward(self, states: torch.Tensor,
              latents: torch.Tensor) -> torch.Tensor:
    """
        Get the mean of the stochastic policy

        Parameters
        ----------
        states: torch.Tensor
            input states
        latents: torch.Tensor
            input (discrete) latent states

        Returns
        -------
        actions: torch.Tensor
            mean of the action (note that values are squashed within [-1, 1])
        """
    return torch.tanh(self.net(torch.cat((states, latents), dim=-1)))

  def exploit(self, states: torch.Tensor,
              latents: torch.Tensor) -> torch.Tensor:
    """
        ----------
        states: torch.Tensor
            input states
        latents: torch.Tensor
            input latent states

        Returns
        -------
        actions: torch.Tensor
            actions to take
        """

    return self.forward(states, latents)

  def sample(self, states: torch.Tensor,
             latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Sample actions given states

        Parameters
        ----------
        states: torch.Tensor
            input states
        latents: torch.Tensor
            input latent states

        Returns
        -------
        actions: torch.Tensor
            actions to take
        log_pi: torch.Tensor
            log_pi of the actions
        """
    return reparameterize(self.net(torch.cat((states, latents), dim=-1)),
                          self.log_stds)

  def evaluate_log_pi(self, states: torch.Tensor, latents: torch.Tensor,
                      actions: torch.Tensor) -> torch.Tensor:
    """
        Evaluate the log(\pi(a|s, x)) of the given action

        Parameters
        ----------
        states: torch.Tensor
            states that the actions act in
        latents: torch.Tensor
            latent states that the actions act in
        actions: torch.Tensor
            actions taken

        Returns
        -------
        log_pi: : torch.Tensor
            log(\pi(a|s, x))
        """
    return evaluate_log_p(self.net(torch.cat((states, latents), dim=-1)),
                          self.log_stds, actions)


class DiscretePolicy(AbstractPolicy):
  def __init__(self,
               state_size: int,
               latent_size: int,
               action_size: int,
               hidden_units: tuple = (64, 64),
               hidden_activation: nn.Module = nn.Tanh()):
    super().__init__(state_size, latent_size, action_size, hidden_units,
                     hidden_activation)

  def forward(self, states: torch.Tensor,
              latents: torch.Tensor) -> torch.Tensor:
    """
        Get the mean of the stochastic policy
        Parameters
        ----------
        states: torch.Tensor
            input states
        latents: torch.Tensor
            input latent states

        Returns
        -------
        unnormalized logits: torch.Tensor
            unnormalized logits of the categorical action distribution
            (cf. normalized logit is log_p)
        """
    return self.net(torch.cat((states, latents), dim=-1))

  def exploit(self, states: torch.Tensor,
              latents: torch.Tensor) -> torch.Tensor:
    """
        ----------
        states: torch.Tensor
            input states
        latents: torch.Tensor
            input latent states

        Returns
        -------
        actions: torch.Tensor
            actions to take
        """

    logits = self.forward(states, latents)
    return logits.argmax(dim=-1, keepdim=True)

  def sample(self, states: torch.Tensor,
             latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Sample actions given states
        Parameters
        ----------
        states: torch.Tensor
            input states
        latents: torch.Tensor
            input latent states

        Returns
        -------
        actions: torch.Tensor
            actions to take
        log_pi: torch.Tensor
            log_pi of the actions
        """
    logits = self.forward(states, latents)
    dist = Categorical(logits=logits)

    samples = dist.sample()
    action_log_probs = dist.log_prob(samples)

    return samples.view(-1, 1), action_log_probs.view(-1, 1)

  def evaluate_log_pi(self, states: torch.Tensor, latents: torch.Tensor,
                      actions: torch.Tensor) -> torch.Tensor:
    """
        Evaluate the log(\pi(a|s, x)) of the given action
        Parameters
        ----------
        states: torch.Tensor
            states that the actions act in
        actions: torch.Tensor
            actions taken
        Returns
        -------
        log_pi: : torch.Tensor
            log(\pi(a|s, x))
        """
    logits = self.forward(states, latents)
    dist = Categorical(logits=logits)
    return dist.log_prob(actions.squeeze(-1)).view(-1, 1)
