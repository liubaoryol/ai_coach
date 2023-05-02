import math
import torch
import torch.nn.functional as F
from ai_coach_core.model_learning.IQLearn.agent.sac_models import (
    GumbelSoftmax, SquashedNormal)
from torch.distributions import Normal
from ..utils.model_util import make_module, make_module_list, make_activation
from ..utils.config import Config

# this policy uses one-step option, the initial option is fixed as o=dim_c


class OptionPolicyV2(torch.nn.Module):

  def __init__(self, config: Config, dim_s=2, dim_a=2):
    super(OptionPolicyV2, self).__init__()
    self.dim_s = dim_s
    self.dim_a = dim_a
    self.dim_c = config.dim_c
    self.device = torch.device(config.device)
    self.log_clamp = config.log_clamp_policy
    self.is_shared = config.shared_policy
    self.bounded = config.bounded_actor

    activation = make_activation(config.activation)
    n_hidden_pi = config.hidden_policy
    n_hidden_opt = config.hidden_option

    if self.is_shared:
      # output prediction p(ct| st, ct-1) with shape (N x ct-1 x ct)
      self.option_policy = make_module(self.dim_s,
                                       (self.dim_c + 1) * self.dim_c,
                                       n_hidden_opt, activation)
      self.policy = make_module(self.dim_s, self.dim_c * self.dim_a,
                                n_hidden_pi, activation)

      self.a_log_std = torch.nn.Parameter(
          torch.empty(1, self.dim_a, dtype=torch.float32).fill_(0.))
    else:
      self.policy = make_module_list(self.dim_s, self.dim_a, n_hidden_pi,
                                     self.dim_c, activation)
      self.a_log_std = torch.nn.ParameterList([
          torch.nn.Parameter(
              torch.empty(1, self.dim_a, dtype=torch.float32).fill_(0.))
          for _ in range(self.dim_c)
      ])
      # i-th model output prediction p(ct|st, ct-1=i)
      self.option_policy = make_module_list(self.dim_s, self.dim_c,
                                            n_hidden_opt, self.dim_c + 1,
                                            activation)

    self.to(self.device)

  def action_forward(self, st, ct=None):
    # ct: None or long(N x 1) or float(N x dim_c)
    # ct: None for all c, return (N x dim_c x dim_a); else return (N x dim_a)
    # s: N x dim_s, c: N x 1, c should always < dim_c
    if self.is_shared:
      mean = self.policy(st).view(-1, self.dim_c, self.dim_a)
      logstd = self.a_log_std.expand_as(mean[:, 0, :])
    else:
      mean = torch.stack([m(st) for m in self.policy], dim=-2)
      logstd = torch.stack([m.expand_as(mean[:, 0, :]) for m in self.a_log_std],
                           dim=-2)
    if ct is not None:
      # to make backward pass propagate with reparameterized ct
      if ct.shape[-1] > 1 or (self.dim_c == 1 and ct[0][0] != 0):
        ct = ct.view(-1, self.dim_c, 1)
        mean = (mean * ct).sum(dim=-2)
        logstd = (logstd * ct).sum(dim=-2)
      else:
        ind = ct.view(-1, 1, 1).expand(-1, 1, self.dim_a)
        mean = mean.gather(dim=-2, index=ind).squeeze(dim=-2)
        logstd = logstd.gather(dim=-2, index=ind).squeeze(dim=-2)

    # clamp mean and logstd
    mean = mean.clamp(-10, 10)

    logstd = torch.tanh(logstd)
    log_std_min, log_std_max = self.log_clamp
    logstd = log_std_min + 0.5 * (log_std_max - log_std_min) * (logstd + 1)
    std = logstd.exp()

    dist = SquashedNormal(mean, std) if self.bounded else Normal(mean, std)

    return dist

  def _option_logits(self, s):
    if self.is_shared:
      return self.option_policy(s).view(-1, self.dim_c + 1, self.dim_c)
    else:
      return torch.stack([m(s) for m in self.option_policy], dim=-2)

  def get_param(self, low_policy=True):
    if low_policy:
      if self.is_shared:
        return list(self.policy.parameters()) + [self.a_log_std]
      else:
        return list(self.policy.parameters()) + list(
            self.a_log_std.parameters())
    else:
      return list(self.option_policy.parameters())

  # ===================================================================== #

  def option_forward(self, st, ct_1=None):
    # ct_1: long(N x 1) or None
    # ct_1: None: direct output p(ct|st, ct_1): a (N x ct_1 x ct) array
    #             where ct is log-normalized
    logits = self._option_logits(st)
    if ct_1 is not None:
      logits = logits.gather(dim=-2,
                             index=ct_1.view(-1, 1, 1).expand(
                                 -1, 1, self.dim_c)).squeeze(dim=-2)

    dist = GumbelSoftmax(1.0, logits=logits)

    return dist

  def log_trans(self, st, ct_1=None):
    return self.option_forward(st, ct_1).logits

  def log_prob_action(self, st, ct, at):
    # if c is None, return (N x dim_c x 1), else return (N x 1)
    dist = self.action_forward(st, ct)
    if ct is None:
      at = at.view(-1, 1, self.dim_a)

    log_prob = dist.log_prob(at).sum(-1, keepdim=True)
    return log_prob

  def log_prob_option(self, st, ct_1, ct):
    dist = self.option_forward(st, ct_1)
    return dist.log_prob(ct.squeeze(-1))

  def sample_action(self, st, ct, fixed=False):
    dist = self.action_forward(st, ct)
    if fixed:
      action = dist.mean
    else:
      action = dist.sample()

    return action

  def sample_option(self, st, ct_1, fixed=False):
    dist = self.option_forward(st, ct_1)
    if fixed:
      option = dist.logits.argmax(dim=-1, keepdim=True)
    else:
      option = dist.sample().view(*dist.logits.shape[:-1], 1)

    return option

  def policy_log_prob_entropy(self, st, ct, at):
    log_prob = self.log_prob_action(st, ct, at)
    entropy = -log_prob.mean()
    return log_prob, entropy

  def option_log_prob_entropy(self, st, ct_1, ct):
    # c1 can be dim_c, c2 should always < dim_c
    log_opt = self.log_prob_option(st, ct_1, ct)
    entropy = -log_opt.mean()
    return log_opt, entropy

  def log_alpha_beta(self, s_array, a_array):
    log_pis = self.log_prob_action(s_array, None,
                                   a_array).view(-1,
                                                 self.dim_c)  # demo_len x ct
    log_trs = self.log_trans(s_array, None)  # demo_len x (ct_1 + 1) x ct
    log_tr0 = log_trs[0, -1]
    log_trs = log_trs[1:, :-1]  # (demo_len-1) x ct_1 x ct

    log_alpha = [log_tr0 + log_pis[0]]
    for log_tr, log_pi in zip(log_trs, log_pis[1:]):
      log_alpha_t = (log_alpha[-1].unsqueeze(dim=-1) +
                     log_tr).logsumexp(dim=0) + log_pi
      log_alpha.append(log_alpha_t)

    log_beta = [
        torch.zeros(self.dim_c, dtype=torch.float32, device=self.device)
    ]
    for log_tr, log_pi in zip(reversed(log_trs), reversed(log_pis[1:])):
      log_beta_t = ((log_beta[-1] + log_pi).unsqueeze(dim=0) +
                    log_tr).logsumexp(dim=-1)
      log_beta.append(log_beta_t)
    log_beta.reverse()

    log_alpha = torch.stack(log_alpha)
    log_beta = torch.stack(log_beta)
    entropy = -(log_trs * log_trs.exp()).sum(dim=-1)
    return log_alpha, log_beta, log_trs, log_pis, entropy

  def viterbi_path(self, s_array, a_array):
    # c = torch.zeros(s_array.size(0)+1, 1, dtype=torch.long, device=self.device)
    # c[0] = self.dim_c
    # c[1] = torch.randint(0, self.dim_c, (1, 1))
    # for i in range(2, s_array.size(0)+1):
    # if torch.randint(0, 17, (1,)) == 0:
    # c[i] = torch.randint(0, self.dim_c, (1, 1))
    # else:
    # c[i] = c[i-1]
    # return c, torch.zeros(1, dtype=torch.float32, device=self.device)
    with torch.no_grad():
      log_pis = self.log_prob_action(s_array, None, a_array).view(
          -1, 1, self.dim_c)  # demo_len x 1 x ct
      log_trs = self.log_trans(s_array, None)  # demo_len x (ct_1+1) x ct
      log_prob = log_trs[:, :-1] + log_pis
      log_prob0 = log_trs[0, -1] + log_pis[0, 0]
      # forward
      max_path = torch.empty(s_array.size(0),
                             self.dim_c,
                             dtype=torch.long,
                             device=self.device)
      accumulate_logp = log_prob0
      max_path[0] = self.dim_c
      for i in range(1, s_array.size(0)):
        accumulate_logp, max_path[i, :] = (accumulate_logp.unsqueeze(dim=-1) +
                                           log_prob[i]).max(dim=-2)
      # backward
      c_array = torch.zeros(s_array.size(0) + 1,
                            1,
                            dtype=torch.long,
                            device=self.device)
      log_prob_traj, c_array[-1] = accumulate_logp.max(dim=-1)
      for i in range(s_array.size(0), 0, -1):
        c_array[i - 1] = max_path[i - 1][c_array[i]]
    return c_array.detach(), log_prob_traj.detach()
