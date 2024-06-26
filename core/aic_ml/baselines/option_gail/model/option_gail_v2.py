import torch
import torch.nn.functional as F
from .option_policy_v2 import OptionPolicyV2
from .option_discriminator import (OptionDiscriminator, Discriminator,
                                   MoEDiscriminator)
from omegaconf import DictConfig


class OptionGAILV2(torch.nn.Module):

  def __init__(self, config: DictConfig, dim_s=2, dim_a=2):
    super(OptionGAILV2, self).__init__()
    self.dim_a = dim_a
    self.dim_s = dim_s
    self.dim_c = config.dim_c
    self.with_c = config.use_c_in_discriminator
    self.mini_bs = config.mini_batch_size
    self.use_d_info_gail = config.use_d_info_gail
    self.device = torch.device(config.device)

    self.discriminator = OptionDiscriminator(config, dim_s=dim_s, dim_a=dim_a)
    self.policy = OptionPolicyV2(config, dim_s=self.dim_s, dim_a=self.dim_a)

    self.optim = torch.optim.Adam(self.discriminator.parameters(),
                                  weight_decay=1.e-3)
    self.to(self.device)

  def original_gail_reward(self, s, c_1, a, c):
    d = self.discriminator.get_unnormed_d(s, c_1, a, c)
    reward = -F.logsigmoid(d)
    return reward

  def d_info_gail_reward(self, s, c_1, a, c):
    d = self.discriminator.get_unnormed_d(s, c_1, a, c)
    # la, lb, _, _, _ = self.policy.log_alpha_beta(s, a)
    # logpc = (la + lb).log_softmax(dim=-1).gather(dim=-1, index=c)
    reward = -F.logsigmoid(d)
    reward += 0.001 * self.policy.log_prob_option(s, c_1, c)
    return reward

  def gail_reward(self, s, c_1, a, c):
    if not self.use_d_info_gail:
      return self.original_gail_reward(s, c_1, a, c)
    else:
      return self.d_info_gail_reward(s, c_1, a, c)

  def step_original_gan(self, sample_scar, demo_scar, n_step=10):
    sp = torch.cat([s for s, c, a, r in sample_scar], dim=0)
    se = torch.cat([s for s, c, a, r in demo_scar], dim=0)
    c_1p = torch.cat([c[:-1] for s, c, a, r in sample_scar], dim=0)
    c_1e = torch.cat([c[:-1] for s, c, a, r in demo_scar], dim=0)
    cp = torch.cat([c[1:] for s, c, a, r in sample_scar], dim=0)
    ce = torch.cat([c[1:] for s, c, a, r in demo_scar], dim=0)
    ap = torch.cat([a for s, c, a, r in sample_scar], dim=0)
    ae = torch.cat([a for s, c, a, r in demo_scar], dim=0)
    tp = torch.ones(self.mini_bs, 1, dtype=torch.float32, device=self.device)
    te = torch.zeros(self.mini_bs, 1, dtype=torch.float32, device=self.device)

    for _ in range(n_step):
      inds = torch.randperm(sp.size(0), device=self.device)
      for ind_p in inds.split(self.mini_bs):
        sp_b, cp_1b, ap_b, cp_b, tp_b = sp[ind_p], c_1p[ind_p], ap[ind_p], cp[
            ind_p], tp[:ind_p.size(0)]
        ind_e = torch.randperm(se.size(0), device=self.device)[:ind_p.size(0)]
        se_b, ce_1b, ae_b, ce_b, te_b = se[ind_e], c_1e[ind_e], ae[ind_e], ce[
            ind_e], te[:ind_p.size(0)]

        s_array = torch.cat((sp_b, se_b), dim=0)
        a_array = torch.cat((ap_b, ae_b), dim=0)
        c_1array = torch.cat((cp_1b, ce_1b), dim=0)
        c_array = torch.cat((cp_b, ce_b), dim=0)
        t_array = torch.cat((tp_b, te_b), dim=0)
        for _ in range(3):
          src = self.discriminator.get_unnormed_d(s_array, c_1array, a_array,
                                                  c_array)
          loss = F.binary_cross_entropy_with_logits(src, t_array)
          self.optim.zero_grad()
          loss.backward()
          self.optim.step()

    return {'disc_loss': loss.item()}

  def step(self, sample_sar, demo_sar, n_step=10):
    return self.step_original_gan(sample_sar, demo_sar, n_step)

  def convert_demo(self, demo_sa):
    with torch.no_grad():
      out_sample = []
      r_sum_avg = 0.
      for s_array, a_array in demo_sa:
        if self.with_c:
          c_array, _ = self.policy.viterbi_path(s_array, a_array)
        else:
          c_array = torch.zeros(s_array.size(0) + 1,
                                1,
                                dtype=torch.long,
                                device=self.device)
        r_array = self.gail_reward(s_array, c_array[:-1], a_array, c_array[1:])
        out_sample.append((s_array, c_array, a_array, r_array))
        r_sum_avg += r_array.sum().item()
      r_sum_avg /= len(demo_sa)
    return out_sample, r_sum_avg

  def convert_sample(self, sample_scar):
    with torch.no_grad():
      out_sample = []
      r_sum_avg = 0.
      for s_array, c_array, a_array, r_real_array in sample_scar:
        r_fake_array = self.gail_reward(s_array, c_array[:-1], a_array,
                                        c_array[1:])
        out_sample.append((s_array, c_array, a_array, r_fake_array))
        r_sum_avg += r_real_array.sum().item()
      r_sum_avg /= len(sample_scar)
    return out_sample, r_sum_avg
