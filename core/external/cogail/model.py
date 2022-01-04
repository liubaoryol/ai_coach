import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

from external.cogail.distributions import (Bernoulli, Categorical, DiagGaussian,
                                           FixedCategorical)
from external.cogail.utils import init, conv_discrete_2_onehot


class Flatten(nn.Module):
  def forward(self, x):
    return x.view(x.size(0), -1)


class Policy(nn.Module):
  def __init__(self,
               observation_space,
               action_space,
               num_code,
               num_history=1,
               base=None,
               base_kwargs=None):
    super(Policy, self).__init__()
    if base_kwargs is None:
      base_kwargs = {}

    if not isinstance(observation_space, spaces.Discrete):
      raise NotImplementedError
    self.num_obs = observation_space.n
    if base is None:
      base = MLPBase

    self.num_code = num_code
    self.base = base(self.num_obs, code_size=num_code, **base_kwargs)

    if not isinstance(action_space, spaces.MultiDiscrete):
      raise NotImplementedError

    self.num_agents = action_space.shape[0]
    assert self.num_agents == 2

    self.tuple_num_outputs = tuple(action_space.nvec)
    self.dist1 = Categorical(self.base.output_size, self.tuple_num_outputs[0])
    self.dist2 = Categorical(self.base.output_size, self.tuple_num_outputs[1])

    self.recode = CodePosterior(
        num_history * self.num_obs + self.tuple_num_outputs[0], num_code)

  @property
  def is_recurrent(self):
    return self.base.is_recurrent

  @property
  def recurrent_hidden_state_size(self):
    """Size of rnn_hx."""
    return self.base.recurrent_hidden_state_size

  def forward(self, inputs, rnn_hxs, masks):
    raise NotImplementedError

  def act(self, inputs, random_seeds, rnn_hxs, masks, deterministic=False):
    # print(inputs)
    inputs = conv_discrete_2_onehot(inputs, self.num_obs)
    value, actor_features, rnn_hxs = self.base(inputs, random_seeds, rnn_hxs,
                                               masks)
    dist1 = self.dist1(actor_features)
    dist2 = self.dist1(actor_features)

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

  def get_distribution(self, inputs, random_seeds, rnn_hxs, masks):
    inputs = conv_discrete_2_onehot(inputs, self.num_obs)
    _, actor_features, _ = self.base(inputs, random_seeds, rnn_hxs, masks)
    return self.dist1(actor_features), self.dist2(actor_features)

  def get_value(self, inputs, random_seeds, rnn_hxs, masks):
    inputs = conv_discrete_2_onehot(inputs, self.num_obs)
    value, _, _ = self.base(inputs, random_seeds, rnn_hxs, masks)
    return value

  def evaluate_actions(self, inputs, random_seeds, rnn_hxs, masks, action):
    inputs = conv_discrete_2_onehot(inputs, self.num_obs)
    value, actor_features, rnn_hxs = self.base(inputs, random_seeds, rnn_hxs,
                                               masks)
    dist1 = self.dist1(actor_features)
    dist2 = self.dist2(actor_features)

    action_log_probs1 = dist1.log_probs(action[:, 0])
    action_log_probs2 = dist2.log_probs(action[:, 1])
    action_log_probs = action_log_probs1 + action_log_probs2

    dist_entropy1 = dist1.entropy().mean()
    dist_entropy2 = dist2.entropy().mean()
    dist_entropy = dist_entropy1 + dist_entropy2

    pred_action1 = dist1.mode()
    pred_action1 = conv_discrete_2_onehot(pred_action1,
                                          self.tuple_num_outputs[0])

    input_code = torch.cat((inputs, pred_action1), dim=1)
    pred_code = self.recode(input_code)

    return value, action_log_probs, dist_entropy, rnn_hxs, pred_code

  def evaluate_code(self, inputs, action):
    inputs = conv_discrete_2_onehot(inputs, self.num_obs)
    action1 = conv_discrete_2_onehot(action[:, 0], self.tuple_num_outputs[0])
    input_code = torch.cat((inputs, action1), dim=1)
    pred_code = self.recode(input_code)

    return pred_code

  def process_exp_dataset_no_balance(self, inputs, action):
    inputs = conv_discrete_2_onehot(inputs, self.num_obs)
    action1 = conv_discrete_2_onehot(action[:, 0], self.tuple_num_outputs[0])

    input_code = torch.cat((inputs, action1), dim=1)
    pred_code = self.recode(input_code).detach()

    return inputs, action, pred_code


class CodePosterior(nn.Module):
  def __init__(self, num_inputs, num_outputs):
    super(Categorical, self).__init__()

    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(
        x, 0), np.sqrt(2))
    self.recode = nn.Sequential(init_(nn.Linear(num_inputs, 64)), nn.ReLU(),
                                init_(nn.Linear(64, num_outputs)), nn.Softmax())

    self.train()

  def forward(self, x):
    return self.recode(x)
    # return FixedCategorical(logits=x)


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
      has_zeros = ((masks[1:] == 0.0) \
                      .any(dim=-1)
                      .nonzero()
                      .squeeze()
                      .cpu())

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
  def __init__(self,
               num_inputs,
               recurrent=False,
               hidden_size=64,
               code_size=2,
               base_net_small=False):
    super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

    self.hidden_size = hidden_size
    self.code_size = code_size
    self.base_net_small = base_net_small

    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(
        x, 0), np.sqrt(2))

    if self.base_net_small:
      self.actor = nn.Sequential(
          init_(nn.Linear(num_inputs + self.code_size, hidden_size)), nn.Tanh(),
          init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

      self.critic = nn.Sequential(
          init_(nn.Linear(num_inputs + self.code_size, hidden_size)), nn.Tanh(),
          init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
    else:
      self.actor = nn.Sequential(
          init_(nn.Linear(num_inputs + self.code_size, hidden_size)), nn.ReLU(),
          init_(nn.Linear(hidden_size, hidden_size * 2)), nn.ReLU(),
          init_(nn.Linear(hidden_size * 2, hidden_size * 2)), nn.Tanh(),
          init_(nn.Linear(hidden_size * 2, hidden_size)), nn.Tanh())

      self.critic = nn.Sequential(
          init_(nn.Linear(num_inputs + self.code_size, hidden_size)), nn.ReLU(),
          init_(nn.Linear(hidden_size, hidden_size * 2)), nn.ReLU(),
          init_(nn.Linear(hidden_size * 2, hidden_size * 2)), nn.Tanh(),
          init_(nn.Linear(hidden_size * 2, hidden_size)), nn.Tanh())

    self.critic_linear = init_(nn.Linear(hidden_size, 1))

    self.train()

  def forward(self, inputs, random_seed, rnn_hxs, masks):
    x = inputs

    x = torch.cat((x, random_seed), dim=1)

    hidden_critic = self.critic(x)
    hidden_actor = self.actor(x)

    return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
