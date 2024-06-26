import math
import torch

from torch import nn
from typing import Tuple


def init_param(module: nn.Module, gain: float = 1.) -> nn.Module:
  """
    Init the input neural network's linear layers Y = AX + B according to
        1.) A is orthogonal with gain
        2.) B = 0

    Parameters
    ----------
    module: nn.Module
        input neural network
    gain: float

    Returns
    -------
    module: nn.Module
        initialized neural network
    """
  nn.init.orthogonal_(module.weight.data, gain=gain)
  nn.init.constant_(module.bias.data, 0)
  return module


def build_mlp(input_dim: int,
              output_dim: int,
              hidden_units: tuple = (64, 64),
              hidden_activation: nn.Module = nn.Tanh(),
              init: bool = False,
              output_activation: nn.Module = None,
              gain: float = 1.) -> nn.Module:
  """
    Build a MLP network

    Parameters
    ----------
    input_dim: int
        dimension of the input of the neural network
    output_dim: int
        dimension of the output of the neural network
    hidden_units: tuple
        hidden units of the neural network
    hidden_activation: nn.Module
        activation function of the hidden layers
    init: bool
        whether to init the neural network to be orthogonal weighted
    output_activation: nn.Module
        activation function of the output layer
    gain: float
        gain for the init function

    Returns
    -------
    nn: nn.Module
        MLP net
    """
  layers = []
  units = input_dim
  for next_units in hidden_units:
    if init:
      layers.append(init_param(nn.Linear(units, next_units), gain=gain))
    else:
      layers.append(nn.Linear(units, next_units))
    layers.append(hidden_activation)
    units = next_units
  if init:
    layers.append(init_param(nn.Linear(units, output_dim), gain=gain))
  else:
    layers.append(nn.Linear(units, output_dim))
  if output_activation is not None:
    layers.append(output_activation)
  return nn.Sequential(*layers)


def calculate_log_p(log_stds: torch.Tensor, noises: torch.Tensor,
                    samples: torch.Tensor) -> torch.Tensor:
  """
    Calculate log(p) given log(std) of the distribution, noises,
      and samples

    Parameters
    ----------
    log_stds: torch.Tensor
        log(std) of the distribution
    noises: torch.Tensor
        noises added to the action
    samples: torch.Tensor
        (continuous) samples

    Returns
    -------
    log_p: torch.Tensor
        log(p)
    """
  gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(
      dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)

  return gaussian_log_probs - torch.log(1 - samples.pow(2) + 1e-6).sum(
      dim=-1, keepdim=True)


def reparameterize(means: torch.Tensor,
                   log_stds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  """
    Get action and its log_pi according to mean and log_std

    Parameters
    ----------
    means: torch.Tensor
        mean value of the action
    log_stds: torch.Tensor
        log(std) of the action

    Returns
    -------
    actions: torch.Tensor
        actions to take
    log_pi: torch.Tensor
        log_pi of the actions
    """
  noises = torch.randn_like(means)
  us = means + noises * log_stds.exp()
  actions = torch.tanh(us)
  return actions, calculate_log_p(log_stds, noises, actions)


def atanh(x: torch.Tensor) -> torch.Tensor:
  """
    Return atanh of the input. Modified torch.atanh in case the output is nan.

    Parameters
    ----------
    x: torch.Tensor
        input

    Returns
    -------
    y: torch.Tensor
        atanh(x)
    """
  return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))


def evaluate_log_p(means: torch.Tensor, log_stds: torch.Tensor,
                   samples: torch.Tensor) -> torch.Tensor:
  """
    Evaluate the log(p) of the given samples

    Parameters
    ----------
    means: torch.Tensor
        mean value of the action distribution
    log_stds: torch.Tensor
        log(std) of the action distribution
    samples: torch.Tensor
        (continuous) samples taken

    Returns
    -------
    log_p: : torch.Tensor
        log(p)
    """
  noises = (atanh(samples) - means) / (log_stds.exp() + 1e-8)
  return calculate_log_p(log_stds, noises, samples)
