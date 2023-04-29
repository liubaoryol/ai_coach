import torch
from ai_coach_core.model_learning.IQLearn.agent.sac_models import (
    GumbelSoftmax, SquashedNormal)
from torch.distributions import Normal
from aicoach_baselines.option_gail.utils.model_util import (make_module,
                                                            make_module_list,
                                                            make_activation)

# this policy uses one-step option, the initial option is fixed as o=dim_c


class MentalPolicy(torch.nn.Module):

  def __init__(self,
               dim_s,
               dim_a,
               dim_c,
               device,
               log_std_bounds,
               is_shared,
               activation,
               hidden_policy,
               hidden_option,
               gumbel_temperature,
               bounded_actor=True):
    super(MentalPolicy, self).__init__()
    self.dim_s = dim_s
    self.dim_a = dim_a
    self.dim_c = dim_c
    self.device = torch.device(device)
    self.log_std_bounds = log_std_bounds
    self.is_shared = is_shared
    self.temperature = gumbel_temperature
    activation = make_activation(activation)
    n_hidden_pi = hidden_policy
    n_hidden_opt = hidden_option
    self.bounded = bounded_actor

    if self.is_shared:
      # output prediction p(ct| st, ct-1) with shape (N x ct-1 x ct)
      self.option_policy = make_module(self.dim_s,
                                       (self.dim_c + 1) * self.dim_c,
                                       n_hidden_pi, activation)
      self.policy = make_module(self.dim_s, self.dim_c * self.dim_a,
                                n_hidden_opt, activation)

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
    # ct: None or long(N x 1)
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
      # TODO: backward pass not propagate
      ind = ct.view(-1, 1, 1).expand(-1, 1, self.dim_a)
      mean = mean.gather(dim=-2, index=ind).squeeze(dim=-2)
      logstd = logstd.gather(dim=-2, index=ind).squeeze(dim=-2)

    # clamp mean and logstd
    mean = mean.clamp(-10, 10)

    logstd = torch.tanh(logstd)
    log_std_min, log_std_max = self.log_std_bounds
    logstd = log_std_min + 0.5 * (log_std_max - log_std_min) * (logstd + 1)
    std = logstd.exp()

    dist = SquashedNormal(mean, std) if self.bounded else Normal(mean, std)

    return dist

  def _option_logits(self, s):
    if self.is_shared:
      return self.option_policy(s).view(-1, self.dim_c + 1, self.dim_c)
    else:
      return torch.stack([m(s) for m in self.option_policy], dim=-2)

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

    dist = GumbelSoftmax(self.temperature, logits=logits)

    return dist

  def log_prob_action(self, st, ct, at):
    # if c is None, return (N x dim_c x 1), else return (N x 1)
    dist = self.action_forward(st, ct)
    if ct is None:
      at = at.view(-1, 1, self.dim_a)

    log_prob = dist.log_prob(at).sum(-1, keepdim=True)
    return log_prob

  def log_prob_option(self, st, ct_1, ct):
    dist = self.option_forward(st, ct_1)
    return dist.log_prob(ct)

  def rsample_action(self, st, ct):
    dist = self.action_forward(st, ct)
    action = dist.rsample()
    log_prob = dist.log_prob(action).sum(-1, keepdim=True)

    return action, log_prob

  def sample_action(self, st, ct, fixed=False):
    dist = self.action_forward(st, ct)
    if fixed:
      action = dist.mean
    else:
      action = dist.sample()

    log_prob = dist.log_prob(action).sum(-1, keepdim=True)

    return action, log_prob

  def rsample_option(self, st, ct_1):
    dist = self.option_forward(st, ct_1)
    option = dist.rsample()
    log_prob = dist.log_prob(option).view(*dist.logits.shape[:-1], 1)

    return option, log_prob

  def sample_option(self, st, ct_1, fixed=False):
    dist = self.option_forward(st, ct_1)
    if fixed:
      option = dist.logits.argmax(dim=-1, keepdim=True)
    else:
      option = dist.sample().view(*dist.logits.shape[:-1], 1)

    log_prob = dist.log_prob(option.squeeze(-1)).view(*dist.logits.shape[:-1],
                                                      1)

    return option, log_prob

  def log_alpha_beta(self, s_array, a_array):
    log_pis = self.log_prob_action(s_array, None,
                                   a_array).view(-1,
                                                 self.dim_c)  # demo_len x ct
    log_trs = self.option_forward(s_array,
                                  None).logits  # demo_len x (ct_1 + 1) x ct
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
    with torch.no_grad():
      log_pis = self.log_prob_action(s_array, None, a_array).view(
          -1, 1, self.dim_c)  # demo_len x 1 x ct
      log_trs = self.option_forward(s_array,
                                    None).logits  # demo_len x (ct_1+1) x ct
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
