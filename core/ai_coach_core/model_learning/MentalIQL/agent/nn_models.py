import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from ai_coach_core.model_learning.IQLearn.agent.sac_models import (
    SquashedNormal, GumbelSoftmax)
from aicoach_baselines.option_gail.utils.model_util import (make_module_list,
                                                            make_activation)


# #############################################################################
# SoftQ models
class OptionSoftQNetwork(nn.Module):

  def __init__(self,
               obs_dim,
               action_dim,
               option_dim,
               list_hidden_dims,
               activation,
               gamma=0.99,
               use_tanh: bool = False):
    super().__init__()
    self.use_tanh = use_tanh
    self.gamma = gamma


class SimpleOptionQNetwork(OptionSoftQNetwork):

  def __init__(self,
               obs_dim,
               action_dim,
               option_dim,
               list_hidden_dims,
               activation,
               gamma=0.99,
               use_tanh: bool = False):
    super().__init__(obs_dim, action_dim, option_dim, list_hidden_dims,
                     activation, gamma, use_tanh)

    activation = make_activation(activation)
    # Q1 architecture
    self.Q1 = make_module_list(obs_dim, action_dim, list_hidden_dims,
                               option_dim, activation)

  def forward(self, state, option, *args):
    out = torch.stack([m(state) for m in self.Q1], dim=-2)

    option_idx = option.view(-1, 1, 1).expand(-1, 1, out.shape[-1])
    out = out.gather(dim=-2, index=option_idx)

    if self.use_tanh:
      out = torch.tanh(out) * 1 / (1 - self.gamma)

    return out


class DoubleOptionQNetwork(OptionSoftQNetwork):

  def __init__(self,
               obs_dim,
               action_dim,
               option_dim,
               list_hidden_dims,
               activation,
               gamma=0.99,
               use_tanh: bool = False):
    super().__init__(obs_dim, action_dim, option_dim, list_hidden_dims,
                     activation, gamma, use_tanh)

    activation = make_activation(activation)
    self.net1 = make_module_list(obs_dim, action_dim, list_hidden_dims,
                                 option_dim, activation)
    self.net2 = make_module_list(obs_dim, action_dim, list_hidden_dims,
                                 option_dim, activation)

  def forward(self, state, option, both=False, *args):
    q1 = torch.stack([m(state) for m in self.net1], dim=-2)
    q2 = torch.stack([m(state) for m in self.net2], dim=-2)

    option_idx = option.view(-1, 1, 1).expand(-1, 1, q1.shape[-1])
    q1 = q1.gather(dim=-2, index=option_idx)
    q2 = q2.gather(dim=-2, index=option_idx)

    if self.use_tanh:
      q1 = torch.tanh(q1) * 1 / (1 - self.gamma)
      q2 = torch.tanh(q2) * 1 / (1 - self.gamma)

    if both:
      return q1, q2
    else:
      return torch.minimum(q1, q2)


# #############################################################################
# SACQCritic models


class SACOptionQCritic(nn.Module):

  def __init__(self,
               obs_dim,
               action_dim,
               option_dim,
               list_hidden_dims,
               activation,
               gamma=0.99,
               use_tanh: bool = False):
    super().__init__()
    self.obs_dim = obs_dim
    self.action_dim = action_dim
    self.use_tanh = use_tanh
    self.gamma = gamma


class DoubleOptionQCritic(SACOptionQCritic):

  def __init__(self,
               obs_dim,
               action_dim,
               option_dim,
               list_hidden_dims,
               activation,
               gamma=0.99,
               use_tanh: bool = False):
    super().__init__(obs_dim, action_dim, option_dim, list_hidden_dims,
                     activation, gamma, use_tanh)

    activation = make_activation(activation)
    self.Q1 = make_module_list(obs_dim + action_dim, 1, list_hidden_dims,
                               option_dim, activation)
    self.Q2 = make_module_list(obs_dim + action_dim, 1, list_hidden_dims,
                               option_dim, activation)

  def forward(self, obs, option, action, both=False, *args):

    obs_action = torch.cat([obs, action], dim=-1)
    q1 = torch.cat([m(obs_action) for m in self.Q1], dim=-1)
    q2 = torch.cat([m(obs_action) for m in self.Q2], dim=-1)

    q1 = q1.gather(dim=-1, index=option)
    q2 = q2.gather(dim=-1, index=option)

    if self.use_tanh:
      q1 = torch.tanh(q1) * 1 / (1 - self.gamma)
      q2 = torch.tanh(q2) * 1 / (1 - self.gamma)

    if both:
      return q1, q2
    else:
      return torch.min(q1, q2)


# #############################################################################
# Actor models


class AbstractOptionActor(nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, obs, option):
    raise NotImplementedError

  def rsample(self, obs, option):
    raise NotImplementedError

  def sample(self, obs, option):
    raise NotImplementedError

  def exploit(self, obs, option):
    raise NotImplementedError

  def is_discrete(self):
    raise NotImplementedError


class DiagGaussianOptionActor(AbstractOptionActor):
  """torch.distributions implementation of an diagonal Gaussian policy."""

  def __init__(self,
               obs_dim,
               action_dim,
               option_dim,
               list_hidden_dims,
               activation,
               log_std_bounds,
               bounded=True,
               use_nn_logstd=False,
               clamp_action_logstd=False):
    super().__init__()
    self.use_nn_logstd = use_nn_logstd
    self.clamp_action_logstd = clamp_action_logstd

    output_dim = action_dim
    if self.use_nn_logstd:
      output_dim = 2 * action_dim
    else:
      self.action_logstd = nn.Parameter(
          torch.empty(1, action_dim, dtype=torch.float32).fill_(0.))

    activation = make_activation(activation)

    self.trunk = make_module_list(obs_dim, output_dim, list_hidden_dims,
                                  option_dim, activation)

    self.log_std_bounds = log_std_bounds
    self.bounded = bounded

  def forward(self, obs, option):
    if self.use_nn_logstd:
      out = torch.stack([m(obs) for m in self.trunk], dim=-2)
      option_idx = option.view(-1, 1, 1).expand(-1, 1, out.shape[-1])
      out = out.gather(dim=-2, index=option_idx)
      mu, log_std = out.chunk(2, dim=-1)
    else:
      out = torch.stack([m(obs) for m in self.trunk], dim=-2)
      option_idx = option.view(-1, 1, 1).expand(-1, 1, out.shape[-1])
      mu = out.gather(dim=-2, index=option_idx)
      log_std = self.action_logstd.expand_as(mu)

    # clamp logstd
    if self.clamp_action_logstd:
      log_std = log_std.clamp(self.log_std_bounds[0], self.log_std_bounds[1])
    else:
      # constrain log_std inside [log_std_min, log_std_max]
      log_std = torch.tanh(log_std)
      log_std_min, log_std_max = self.log_std_bounds
      log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
    std = log_std.exp()

    if self.bounded:
      dist = SquashedNormal(mu, std)
    else:
      mu = mu.clamp(-10, 10)
      dist = Normal(mu, std)

    return dist

  def rsample(self, obs, option):
    dist = self.forward(obs, option)
    action = dist.rsample()
    log_prob = dist.log_prob(action).sum(-1, keepdim=True)

    return action, log_prob

  def sample(self, obs, option):
    return self.rsample(obs, option)

  def exploit(self, obs, option):
    return self.forward(obs, option).mean

  def is_discrete(self):
    return False


class DiscreteOptionActor(AbstractOptionActor):
  'cf) https://github.com/openai/spinningup/issues/148 '

  def __init__(self, obs_dim, action_dim, option_dim, list_hidden_dims,
               activation):
    super().__init__()

    output_dim = action_dim
    activation = make_activation(activation)
    self.trunk = make_module_list(obs_dim, output_dim, list_hidden_dims,
                                  option_dim, activation)

  def forward(self, obs, option):
    logits = torch.stack([m(obs) for m in self.trunk], dim=-2)
    option_idx = option.view(-1, 1, 1).expand(-1, 1, logits.shape[-1])
    logits = logits.gather(dim=-2, index=option_idx)
    dist = Categorical(logits=logits)
    return dist

  def action_probs(self, obs, option):
    dist = self.forward(obs, option)
    action_probs = dist.probs
    # avoid numerical instability
    z = (action_probs == 0.0).float() * 1e-10
    log_action_probs = torch.log(action_probs + z)

    return action_probs, log_action_probs

  def exploit(self, obs, option):
    dist = self.forward(obs, option)
    return dist.logits.argmax(dim=-1)

  def sample(self, obs, option):
    dist = self.forward(obs, option)

    samples = dist.sample()
    action_log_probs = dist.log_prob(samples)

    return samples, action_log_probs

  def rsample(self, obs, option):
    'should not be used'
    raise NotImplementedError

  def is_discrete(self):
    return True


class SoftDiscreteOptionActor(AbstractOptionActor):
  'cf) https://github.com/openai/spinningup/issues/148 '

  def __init__(self, obs_dim, action_dim, option_dim, list_hidden_dims,
               activation, temperature):
    super().__init__()

    output_dim = action_dim
    activation = make_activation(activation)
    self.trunk = make_module_list(obs_dim, output_dim, list_hidden_dims,
                                  option_dim, activation)

    self.temperature = torch.tensor(temperature)

  def forward(self, obs, option):
    logits = torch.stack([m(obs) for m in self.trunk], dim=-2)
    option_idx = option.view(-1, 1, 1).expand(-1, 1, logits.shape[-1])
    logits = logits.gather(dim=-2, index=option_idx)

    dist = GumbelSoftmax(self.temperature, logits=logits)
    return dist

  def exploit(self, obs, option):
    dist = self.forward(obs, option)
    return dist.logits.argmax(dim=-1)

  def sample(self, obs, option):
    dist = self.forward(obs, option)

    samples = dist.sample()
    action_log_probs = dist.log_prob(samples).view(-1, 1)

    return samples, action_log_probs

  def rsample(self, obs, option):
    dist = self.forward(obs, option)

    action = dist.rsample()
    log_prob = dist.log_prob(action).view(-1, 1)

    return action, log_prob

  def is_discrete(self):
    return True
