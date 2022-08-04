import torch
import abc

from torch import nn
from torch.distributions import Categorical
from typing import Tuple
from .utils import build_mlp, reparameterize, evaluate_log_p


class AbstractTransition(nn.Module):
  """
    transition of latent state Tx(x|s, x, a)

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
    self.net = build_mlp(input_dim=state_size + latent_size + action_size,
                         output_dim=latent_size,
                         hidden_units=hidden_units,
                         hidden_activation=hidden_activation,
                         init=True)

  @abc.abstractmethod
  def forward(self, next_states: torch.Tensor, latents: torch.Tensor,
              actions: torch.Tensor) -> torch.Tensor:
    pass

  @abc.abstractmethod
  def sample(self, next_states: torch.Tensor, latents: torch.Tensor,
             actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pass

  @abc.abstractmethod
  def exploit(self, next_states: torch.Tensor, latents: torch.Tensor,
              actions: torch.Tensor) -> torch.Tensor:
    pass

  @abc.abstractmethod
  def evaluate_log_Tx(self, next_states: torch.Tensor, latents: torch.Tensor,
                      actions: torch.Tensor,
                      next_latents: torch.Tensor) -> torch.Tensor:
    pass


class ContinousTransition(AbstractTransition):
  def __init__(self,
               state_size: int,
               latent_size: int,
               action_size: int,
               hidden_units: tuple = (64, 64),
               hidden_activation: nn.Module = nn.Tanh()):
    super().__init__(state_size, latent_size, action_size, hidden_units,
                     hidden_activation)

    self.log_stds = nn.Parameter(torch.zeros(1, latent_size))

  def forward(self, next_states: torch.Tensor, latents: torch.Tensor,
              actions: torch.Tensor) -> torch.Tensor:
    """
        Get the mean of the next_latents

        Parameters
        ----------
        next_states: torch.Tensor
            input states
        latents: torch.Tensor
            input latent states
        actions: torch.Tensor
            input actions

        Returns
        -------
        next_latents: torch.Tensor
            mean of the next_latents (tanh is to squash values in [-1, 1])
        """
    return torch.tanh(
        self.net(torch.cat((next_states, latents, actions), dim=-1)))

  def sample(self, next_states: torch.Tensor, latents: torch.Tensor,
             actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Sample next_latents given states

        Parameters
        ----------
        next_states: torch.Tensor
            input states
        latents: torch.Tensor
            input latent states
        actions: torch.Tensor
            input actions

        Returns
        -------
        next_latents: torch.Tensor
            updated latents
        log_Tx: torch.Tensor
            log_Tx of the next_latents
        """
    return reparameterize(
        self.net(torch.cat((next_states, latents, actions), dim=-1)),
        self.log_stds)

  def exploit(self, next_states: torch.Tensor, latents: torch.Tensor,
              actions: torch.Tensor) -> torch.Tensor:
    return self.forward(next_states, latents, actions)

  def evaluate_log_Tx(self, next_states: torch.Tensor, latents: torch.Tensor,
                      actions: torch.Tensor,
                      next_latents: torch.Tensor) -> torch.Tensor:
    """
        Evaluate the log(Tx(x|s, x, a)) of the given next_latents

        Parameters
        ----------
        next_states: torch.Tensor
            states that the actions act in
        latents: torch.Tensor
            latent states that the actions act in
        actions: torch.Tensor
            actions taken
        next_latents: torch.Tensor
            next latent states 

        Returns
        -------
        log_Tx: : torch.Tensor
            log(Tx(x|s, x, a))
        """
    return evaluate_log_p(
        self.net(torch.cat((next_states, latents, actions), dim=-1)),
        self.log_stds, next_latents)


class DiscreteTransition(AbstractTransition):
  def __init__(self,
               state_size: int,
               latent_size: int,
               action_size: int,
               hidden_units: tuple = (64, 64),
               hidden_activation: nn.Module = nn.Tanh()):
    super().__init__(state_size, latent_size, action_size, hidden_units,
                     hidden_activation)

  def forward(self, next_states: torch.Tensor, latents: torch.Tensor,
              actions: torch.Tensor) -> torch.Tensor:
    """
        Get the mean of the next_latent

        Parameters
        ----------
        next_states: torch.Tensor
            input states
        latents: torch.Tensor
            input latent states
        actions: torch.Tensor
            input actions

        Returns
        -------
        unnormalized logits: torch.Tensor
            unnormalized logits of the categorical latent distribution
            (cf. normalized logit is log_Tx)
        """
    return self.net(torch.cat((next_states, latents, actions), dim=-1))

  def sample(self, next_states: torch.Tensor, latents: torch.Tensor,
             actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Sample next_latents given states

        Parameters
        ----------
        next_states: torch.Tensor
            input states
        latents: torch.Tensor
            input latent states
        actions: torch.Tensor
            input actions

        Returns
        -------
        next_latents: torch.Tensor
            updated latents
        log_Tx: torch.Tensor
            log_Tx of the next_latents
        """
    logits = self.forward(next_states, latents, actions)
    dist = Categorical(logits=logits)

    samples = dist.sample()
    next_latent_log_probs = dist.log_prob(samples)

    return samples.view(-1, 1), next_latent_log_probs.view(-1, 1)

  def exploit(self, next_states: torch.Tensor, latents: torch.Tensor,
              actions: torch.Tensor) -> torch.Tensor:
    logits = self.forward(next_states, latents, actions)
    return logits.argmax(dim=-1, keepdim=True)

  def evaluate_log_Tx(self, next_states: torch.Tensor, latents: torch.Tensor,
                      actions: torch.Tensor,
                      next_latents: torch.Tensor) -> torch.Tensor:
    """
        Evaluate the log(Tx(x|s, x, a)) of the given next_latents

        Parameters
        ----------
        next_states: torch.Tensor
            states that the actions act in
        latents: torch.Tensor
            latent states that the actions act in
        actions: torch.Tensor
            actions taken
        next_latents: torch.Tensor
            next latent states 

        Returns
        -------
        log_Tx: : torch.Tensor
            log(Tx(x|s, x, a))
        """
    logits = self.forward(next_states, latents, actions)
    dist = Categorical(logits=logits)
    return dist.log_prob(next_latents.squeeze(-1)).view(-1, 1)
