import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple


def disable_gradient(network: nn.Module):
  """Disable the gradients of parameters in the network"""
  for param in network.parameters():
    param.requires_grad = False


def calculate_gae(values: torch.Tensor, rewards: torch.Tensor,
                  dones: torch.Tensor, next_values: torch.Tensor, gamma: float,
                  lambd: float) -> Tuple[torch.Tensor, torch.Tensor]:
  """
    Calculate generalized advantage estimator

    Parameters
    ----------
    values: torch.Tensor
        values of the states
    rewards: torch.Tensor
        rewards given by the reward function
    dones: torch.Tensor
        if this state is the end of the episode
    next_values: torch.Tensor
        values of the next states
    gamma: float
        discount factor
    lambd: float
        lambd factor

    Returns
    -------
    advantages: torch.Tensor
        advantages
    gaes: torch.Tensor
        normalized gae
    """
  # calculate TD errors
  deltas = rewards + gamma * next_values * (1 - dones) - values
  # initialize gae
  gaes = torch.empty_like(rewards)

  # calculate gae recursively from behind
  gaes[-1] = deltas[-1]
  for t in reversed(range(rewards.shape[0] - 1)):
    gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

  return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


def one_hot(indices: torch.Tensor, num_classes, device):
  return F.one_hot(indices.squeeze(1).long(),
                   num_classes=num_classes).to(device=device, dtype=torch.float)


def gumbel_softmax_sample(logits: torch.Tensor,
                          temperature: float,
                          eps=1e-20) -> torch.Tensor:
  """
    Adds Gumbel noise to `logits` and applies softmax along the last dimension.
    """
  # gumbel samples
  U = torch.rand_like(logits)
  g_samples = -torch.log(-torch.log(U + eps) + eps)

  # add gumbel noise to logits
  y = logits + g_samples
  return F.softmax(y / temperature, dim=-1)
