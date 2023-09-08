import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .envs import VecNormalize

import collections.abc
import torch.utils.data as torch_data


# Get a render function
def get_render_func(venv):
  if hasattr(venv, 'envs'):
    return venv.envs[0].render
  elif hasattr(venv, 'venv'):
    return get_render_func(venv.venv)
  elif hasattr(venv, 'env'):
    return get_render_func(venv.env)

  return None


def get_vec_normalize(venv):
  if isinstance(venv, VecNormalize):
    return venv
  elif hasattr(venv, 'venv'):
    return get_vec_normalize(venv.venv)

  return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):

  def __init__(self, bias):
    super(AddBias, self).__init__()
    self._bias = nn.Parameter(bias.unsqueeze(1))

  def forward(self, x):
    if x.dim() == 2:
      bias = self._bias.t().view(1, -1)
    else:
      bias = self._bias.t().view(1, -1, 1, 1)

    return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
  """Decreases the learning rate linearly"""
  lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
  weight_init(module.weight.data, gain=gain)
  bias_init(module.bias.data)
  return module


def cleanup_log_dir(log_dir):
  try:
    os.makedirs(log_dir)
  except OSError:
    files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
    for f in files:
      os.remove(f)


def conv_discrete_2_onehot(indices: torch.Tensor, num_classes):
  return F.one_hot(indices.squeeze(1), num_classes=num_classes).float()


class TorchDatasetConverter(torch_data.Dataset):

  def __init__(self, sa_trajectories, use_confidence=False) -> None:
    super().__init__()
    self.use_confidence = use_confidence

    self.confidences = []
    self.states = []
    self.actions = []
    for traj in sa_trajectories:
      for elem in traj:
        if use_confidence:
          state, action, conf = elem
        else:
          state, action = elem

        if action is not None:
          self.states.append(state)
          self.actions.append(action)
          if use_confidence:
            self.confidences.append(conf)

    self.length = len(self.states)

  def __getitem__(self, index):

    state = self.states[index]
    if not isinstance(state, collections.abc.Sequence):
      state = [state]

    action = self.actions[index]
    if not isinstance(action, collections.abc.Sequence):
      action = [action]

    if self.use_confidence:
      confidence = self.confidences[index]
      if not isinstance(confidence, collections.abc.Sequence):
        confidence = [confidence]
      return (torch.Tensor(state).long(), torch.Tensor(action).long(),
              torch.Tensor(confidence).float())
    else:
      return (torch.Tensor(state).long(), torch.Tensor(action).long())
    # return (torch.Tensor([self.states[index]]).long(),
    #         torch.Tensor([self.actions[index]]).long())

  def __len__(self):
    return self.length


class TorchLatentDatasetConverter(torch_data.Dataset):

  def __init__(self, sax_trajectories_no_terminal) -> None:
    super().__init__()
    self.states = []
    self.actions = []
    for traj in sax_trajectories_no_terminal:
      for state, action, latent in traj:
        self.states.append((state, *latent))
        self.actions.append(action)

    self.length = len(self.states)

  def __getitem__(self, index):

    state = self.states[index]
    if not isinstance(state, collections.abc.Sequence):
      state = [state]

    action = self.actions[index]
    if not isinstance(action, collections.abc.Sequence):
      action = [action]

    return (torch.Tensor(state).long(), torch.Tensor(action).long())
    # return (torch.Tensor([self.states[index]]).long(),
    #         torch.Tensor([self.actions[index]]).long())

  def __len__(self):
    return self.length