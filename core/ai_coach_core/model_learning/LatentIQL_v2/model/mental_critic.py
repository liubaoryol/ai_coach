import torch
from aicoach_baselines.option_gail.utils.model_util import (make_module,
                                                            make_module_list,
                                                            make_activation)

# This file should be included by option_ppo.py and never be used otherwise


class MentalCritic(torch.nn.Module):

  def __init__(
      self,
      dim_s,
      dim_a,
      dim_c,
      device,
      is_shared,
      activation,
      hidden_critic,
      gamma=0.99,
      use_tanh: bool = False,
  ):
    super(MentalCritic, self).__init__()
    self.dim_s = dim_s
    self.dim_a = dim_a
    self.dim_c = dim_c
    self.device = torch.device(device)
    self.is_shared = is_shared

    self.gamma = gamma
    self.use_tanh = use_tanh

    activation = make_activation(activation)
    n_hidden_v = hidden_critic

    if self.is_shared:
      self.Q1 = make_module(self.dim_s + self.dim_a,
                            (self.dim_c + 1) * self.dim_c, n_hidden_v,
                            activation)
      self.Q2 = make_module(self.dim_s + self.dim_a,
                            (self.dim_c + 1) * self.dim_c, n_hidden_v,
                            activation)
    else:
      self.Q1 = make_module_list(self.dim_s + self.dim_a, self.dim_c,
                                 n_hidden_v, self.dim_c + 1, activation)
      self.Q2 = make_module_list(self.dim_s + self.dim_a, self.dim_c,
                                 n_hidden_v, self.dim_c + 1, activation)

    self.to(self.device)

  def forward(self, s, ct_1, ct, a, both=False):
    q_input = torch.cat((s, a), dim=-1)
    if self.is_shared:
      q1 = self.Q1(q_input).view(-1, self.dim_c + 1, self.dim_c)
      q2 = self.Q2(q_input).view(-1, self.dim_c + 1, self.dim_c)
    else:
      q1 = torch.stack([m(q_input) for m in self.Q1], dim=-2)
      q2 = torch.stack([m(q_input) for m in self.Q2], dim=-2)

    ind_ct_1 = ct_1.view(-1, 1, 1).expand(-1, 1, self.dim_c)

    q1 = q1.gather(dim=-2, index=ind_ct_1).squeeze(dim=-2)
    q2 = q2.gather(dim=-2, index=ind_ct_1).squeeze(dim=-2)

    if ct.shape[-1] > 1 or (self.dim_c == 1 and ct[0][0] != 0):
      q1 = (q1 * ct).sum(dim=-1, keepdim=True)
      q2 = (q2 * ct).sum(dim=-1, keepdim=True)
    else:
      ind_ct = ct.view(-1, 1)
      q1 = q1.gather(dim=-1, index=ind_ct)
      q2 = q2.gather(dim=-1, index=ind_ct)

    if self.use_tanh:
      q1 = torch.tanh(q1) * 1 / (1 - self.gamma)
      q2 = torch.tanh(q2) * 1 / (1 - self.gamma)

    if both:
      return q1, q2
    else:
      return torch.min(q1, q2)

  # def get_value(self, s, c=None):
  #   # c could be None for directly output value on each c
  #   if self.is_shared:
  #     vs = self.value(s)
  #   else:
  #     vs = torch.cat([v(s) for v in self.value], dim=-1)

  #   if c is None:
  #     return vs
  #   else:
  #     return vs.gather(dim=-1, index=c)
