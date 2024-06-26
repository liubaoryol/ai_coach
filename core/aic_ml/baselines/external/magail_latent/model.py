# flake8: noqa

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

from ..gail_common_utils.distributions import (Bernoulli, Categorical,
                                               DiagGaussian)
from ..gail_common_utils.utils import init, conv_discrete_2_onehot


class Flatten(nn.Module):
  def forward(self, x):
    return x.view(x.size(0), -1)


class Policy(nn.Module):
  def __init__(self,
               observation_space,
               action_space,
               num_history=1,
               base=None,
               base_kwargs=None):
    super(Policy, self).__init__()
    if base_kwargs is None:
      base_kwargs = {}

    if not isinstance(observation_space, spaces.MultiDiscrete):
      raise NotImplementedError

    assert len(observation_space.nvec) == 3
    self.num_obs = observation_space.nvec[0]
    self.tuple_num_ls = (observation_space.nvec[1], observation_space.nvec[2])

    if base is None:
      base = MLPBase

    self.base = base(self.num_obs, self.tuple_num_ls, **base_kwargs)

    if not isinstance(action_space, spaces.MultiDiscrete):
      raise NotImplementedError

    self.num_agents = action_space.shape[0]
    assert self.num_agents == 2

    self.tuple_num_outputs = tuple(action_space.nvec)
    self.dist1 = Categorical(self.base.output_size, self.tuple_num_outputs[0])
    self.dist2 = Categorical(self.base.output_size, self.tuple_num_outputs[1])

  @property
  def is_recurrent(self):
    return self.base.is_recurrent

  @property
  def recurrent_hidden_state_size(self):
    """Size of rnn_hx."""
    return self.base.recurrent_hidden_state_size

  def forward(self, inputs, rnn_hxs, masks):
    raise NotImplementedError

  def act(self, inputs, rnn_hxs, masks, deterministic=False):
    # print(inputs)
    obs = conv_discrete_2_onehot(inputs[:, 0].unsqueeze(1), self.num_obs)
    ls1 = conv_discrete_2_onehot(inputs[:, 1].unsqueeze(1),
                                 self.tuple_num_ls[0])
    ls2 = conv_discrete_2_onehot(inputs[:, 2].unsqueeze(1),
                                 self.tuple_num_ls[1])

    # value, actor_features, rnn_hxs = self.base(obs, ls1, ls2, rnn_hxs, masks)
    value, feat1, feat2, rnn_hxs = self.base(obs, ls1, ls2, rnn_hxs, masks)
    dist1 = self.dist1(feat1)
    dist2 = self.dist1(feat2)

    if deterministic:
      action1 = dist1.mode()
      action2 = dist2.mode()
    else:
      action1 = dist1.sample()
      action2 = dist2.sample()
    action = torch.cat((action1, action2), dim=1)

    action_log_probs1 = dist1.log_probs(action1)
    action_log_probs2 = dist2.log_probs(action2)
    action_log_probs = action_log_probs1 + action_log_probs2

    dist_entropy1 = dist1.entropy().mean()
    dist_entropy2 = dist2.entropy().mean()
    dist_entropy = dist_entropy1 + dist_entropy2

    return value, action, action_log_probs, rnn_hxs

  def get_distribution(self, inputs, rnn_hxs, masks):
    # inputs = conv_discrete_2_onehot(inputs, self.num_obs)
    # _, actor_features, _ = self.base(inputs, rnn_hxs, masks)
    obs = conv_discrete_2_onehot(inputs[:, 0].unsqueeze(1), self.num_obs)
    ls1 = conv_discrete_2_onehot(inputs[:, 1].unsqueeze(1),
                                 self.tuple_num_ls[0])
    ls2 = conv_discrete_2_onehot(inputs[:, 2].unsqueeze(1),
                                 self.tuple_num_ls[1])
    _, feat1, feat2, _ = self.base(obs, ls1, ls2, rnn_hxs, masks)
    return self.dist1(feat1), self.dist2(feat2)

  def get_value(self, inputs, rnn_hxs, masks):
    # inputs = conv_discrete_2_onehot(inputs, self.num_obs)
    # value, _, _ = self.base(inputs, rnn_hxs, masks)
    obs = conv_discrete_2_onehot(inputs[:, 0].unsqueeze(1), self.num_obs)
    ls1 = conv_discrete_2_onehot(inputs[:, 1].unsqueeze(1),
                                 self.tuple_num_ls[0])
    ls2 = conv_discrete_2_onehot(inputs[:, 2].unsqueeze(1),
                                 self.tuple_num_ls[1])
    value, _, _, _ = self.base(obs, ls1, ls2, rnn_hxs, masks)
    return value

  def evaluate_actions(self, inputs, rnn_hxs, masks, action):
    # inputs = conv_discrete_2_onehot(inputs, self.num_obs)
    # value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
    obs = conv_discrete_2_onehot(inputs[:, 0].unsqueeze(1), self.num_obs)
    ls1 = conv_discrete_2_onehot(inputs[:, 1].unsqueeze(1),
                                 self.tuple_num_ls[0])
    ls2 = conv_discrete_2_onehot(inputs[:, 2].unsqueeze(1),
                                 self.tuple_num_ls[1])
    value, feat1, feat2, rnn_hxs = self.base(obs, ls1, ls2, rnn_hxs, masks)

    dist1 = self.dist1(feat1)
    dist2 = self.dist2(feat2)

    action_log_probs1 = dist1.log_probs(action[:, 0].unsqueeze(1))
    action_log_probs2 = dist2.log_probs(action[:, 1].unsqueeze(1))
    action_log_probs = action_log_probs1 + action_log_probs2

    dist_entropy1 = dist1.entropy().mean()
    dist_entropy2 = dist2.entropy().mean()
    dist_entropy = dist_entropy1 + dist_entropy2

    return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
  def __init__(self, recurrent, recurrent_input_size, hidden_size):
    super(NNBase, self).__init__()

    self._hidden_size = hidden_size
    self._recurrent = recurrent

    if recurrent:
      self.gru = nn.GRU(recurrent_input_size, hidden_size)
      for name, param in self.gru.named_parameters():
        if 'bias' in name:
          nn.init.constant_(param, 0)
        elif 'weight' in name:
          nn.init.orthogonal_(param)

  @property
  def is_recurrent(self):
    return self._recurrent

  @property
  def recurrent_hidden_state_size(self):
    if self._recurrent:
      return self._hidden_size
    return 1

  @property
  def output_size(self):
    return self._hidden_size

  def _forward_gru(self, x, hxs, masks):
    if x.size(0) == hxs.size(0):
      x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
      x = x.squeeze(0)
      hxs = hxs.squeeze(0)
    else:
      # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
      N = hxs.size(0)
      T = int(x.size(0) / N)

      # unflatten
      x = x.view(T, N, x.size(1))

      # Same deal with masks
      masks = masks.view(T, N)

      # Let's figure out which steps in the sequence have a zero for any agent
      # We will always assume t=0 has a zero in it as that makes the logic cleaner
      has_zeros = ((masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu())

      # +1 to correct the masks[1:]
      if has_zeros.dim() == 0:
        # Deal with scalar
        has_zeros = [has_zeros.item() + 1]
      else:
        has_zeros = (has_zeros + 1).numpy().tolist()

      # add t=0 and t=T to the list
      has_zeros = [0] + has_zeros + [T]

      hxs = hxs.unsqueeze(0)
      outputs = []
      for i in range(len(has_zeros) - 1):
        # We can now process steps that don't have any zeros in masks together!
        # This is much faster
        start_idx = has_zeros[i]
        end_idx = has_zeros[i + 1]

        rnn_scores, hxs = self.gru(x[start_idx:end_idx],
                                   hxs * masks[start_idx].view(1, -1, 1))

        outputs.append(rnn_scores)

      # assert len(outputs) == T
      # x is a (T, N, -1) tensor
      x = torch.cat(outputs, dim=0)
      # flatten
      x = x.view(T * N, -1)
      hxs = hxs.squeeze(0)

    return x, hxs


class CNNBase(NNBase):
  def __init__(self, num_inputs, recurrent=False, hidden_size=512):
    super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(
        x, 0), nn.init.calculate_gain('relu'))

    self.main = nn.Sequential(init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
                              nn.ReLU(), init_(nn.Conv2d(32, 64, 4, stride=2)),
                              nn.ReLU(), init_(nn.Conv2d(64, 32, 3, stride=1)),
                              nn.ReLU(), Flatten(),
                              init_(nn.Linear(32 * 7 * 7, hidden_size)),
                              nn.ReLU())

    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(
        x, 0))

    self.critic_linear = init_(nn.Linear(hidden_size, 1))

    self.train()

  def forward(self, inputs, rnn_hxs, masks):
    x = self.main(inputs / 255.0)

    if self.is_recurrent:
      x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

    return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
  def __init__(self, num_obs, tuple_num_ls, recurrent=False, hidden_size=64):
    super(MLPBase, self).__init__(recurrent, num_obs, hidden_size)

    self.hidden_size = hidden_size

    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(
        x, 0), np.sqrt(2))

    self.actor1 = nn.Sequential(
        init_(nn.Linear(num_obs + tuple_num_ls[0], hidden_size)), nn.ReLU(),
        init_(nn.Linear(hidden_size, hidden_size * 2)), nn.ReLU(),
        init_(nn.Linear(hidden_size * 2, hidden_size * 2)), nn.Tanh(),
        init_(nn.Linear(hidden_size * 2, hidden_size)), nn.Tanh())

    self.actor2 = nn.Sequential(
        init_(nn.Linear(num_obs + tuple_num_ls[1], hidden_size)), nn.ReLU(),
        init_(nn.Linear(hidden_size, hidden_size * 2)), nn.ReLU(),
        init_(nn.Linear(hidden_size * 2, hidden_size * 2)), nn.Tanh(),
        init_(nn.Linear(hidden_size * 2, hidden_size)), nn.Tanh())

    self.critic = nn.Sequential(
        init_(nn.Linear(num_obs + sum(tuple_num_ls), hidden_size)), nn.ReLU(),
        init_(nn.Linear(hidden_size, hidden_size * 2)), nn.ReLU(),
        init_(nn.Linear(hidden_size * 2, hidden_size * 2)), nn.Tanh(),
        init_(nn.Linear(hidden_size * 2, hidden_size)), nn.Tanh())

    self.critic_linear = init_(nn.Linear(hidden_size, 1))

    self.train()

  def forward(self, obs, ls1, ls2, rnn_hxs, masks):
    x1 = torch.cat([obs, ls1], dim=1)
    x2 = torch.cat([obs, ls2], dim=1)

    x_all = torch.cat([obs, ls1, ls2], dim=1)

    hidden_actor1 = self.actor1(x1)
    hidden_actor2 = self.actor2(x2)

    val = self.critic_linear(self.critic(x_all))

    return val, hidden_actor1, hidden_actor2, rnn_hxs
