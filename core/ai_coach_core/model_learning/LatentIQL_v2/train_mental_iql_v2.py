#!/usr/bin/env python3

import numpy as np
import os
import torch
import random
from aicoach_baselines.option_gail.utils.utils import (env_class_and_demo_fn,
                                                       get_dirs)
from aicoach_baselines.option_gail.utils.logger import Logger
from aicoach_baselines.option_gail.utils.config import Config
from aicoach_baselines.option_gail.utils.agent import _SamplerCommon
from .model.mental_critic import MentalCritic
from .model.mental_policy import MentalPolicy
from .model.mental_iql_v2 import MentalIQL_V2
from .agent import Sampler


def reward_validate(sampler: _SamplerCommon,
                    policy: MentalPolicy,
                    n_sample=-8,
                    do_print=True):
  trajs = sampler.collect(policy.state_dict(), n_sample, fixed=True)
  rsums = [tr[-1].sum().item() for tr in trajs]
  steps = [tr[-1].size(0) for tr in trajs]

  info_dict = {
      "r-max": np.max(rsums),
      "r-min": np.min(rsums),
      "r-avg": np.mean(rsums),
      "step-max": np.max(steps),
      "step-min": np.min(steps),
  }
  if do_print:
    print(f"R: [ {info_dict['r-min']:.02f} ~ {info_dict['r-max']:.02f},",
          f"avg: {info_dict['r-avg']:.02f} ],",
          f"L: [ {info_dict['step-min']} ~ {info_dict['step-max']} ]")
  return info_dict


def train_iql(agent: MentalIQL_V2,
              smpl_scar,
              demo_sca,
              mini_bs,
              is_sqil,
              n_step=10):
  sp = torch.cat([s[:-1] for s, c, a, r in smpl_scar], dim=0)
  se = torch.cat([s[:-1] for s, c, a in demo_sca], dim=0)
  snp = torch.cat([s[1:] for s, c, a, r in smpl_scar], dim=0)
  sne = torch.cat([s[1:] for s, c, a in demo_sca], dim=0)

  c_1p = torch.cat([c[:-2] for s, c, a, r in smpl_scar], dim=0)
  c_1e = torch.cat([c[:-2] for s, c, a in demo_sca], dim=0)
  cp = torch.cat([c[1:-1] for s, c, a, r in smpl_scar], dim=0)
  ce = torch.cat([c[1:-1] for s, c, a in demo_sca], dim=0)
  ap = torch.cat([a[:-1] for s, c, a, r in smpl_scar], dim=0)
  ae = torch.cat([a[:-1] for s, c, a in demo_sca], dim=0)

  p_a = torch.zeros_like(smpl_scar[0][2][0])
  a_1p = torch.cat([torch.vstack([p_a, a[:-2]]) for s, c, a, r in smpl_scar],
                   dim=0)
  a_1e = torch.cat([torch.vstack([p_a, a[:-2]]) for s, c, a in demo_sca], dim=0)

  lentrj = sp.size(0)
  rp = torch.zeros((lentrj, 1))
  re = torch.zeros((lentrj, 1))
  dp = torch.zeros((lentrj, 1))
  de = torch.zeros((lentrj, 1))

  for _ in range(n_step):
    inds = torch.randperm(lentrj, device=agent.device)
    for ind_p in inds.split(mini_bs):
      sp_b, cp_1b, ap_b, cp_b = sp[ind_p], c_1p[ind_p], ap[ind_p], cp[ind_p]
      snp_b, ap_1b, rp_b, dp_b = snp[ind_p], a_1p[ind_p], rp[ind_p], dp[ind_p]

      ind_e = torch.randperm(lentrj, device=agent.device)[:ind_p.size(0)]
      se_b, ce_1b, ae_b, ce_b = se[ind_e], c_1e[ind_e], ae[ind_e], ce[ind_e]
      sne_b, ae_1b, re_b, de_b = sne[ind_e], a_1e[ind_e], re[ind_e], de[ind_e]

      policy_batch = (sp_b, cp_1b, ap_1b, snp_b, cp_b, ap_b, rp_b, dp_b)
      expert_batch = (se_b, ce_1b, ae_1b, sne_b, ce_b, ae_b, re_b, de_b)

      agent.iq_update(policy_batch, expert_batch, is_sqil=is_sqil)


def convert_demo(demo_sa, agent: MentalIQL_V2):
  with torch.no_grad():
    out_sample = []
    for s_array, a_array in demo_sa:
      c_array, _ = agent.policy.viterbi_path(s_array, a_array)
      out_sample.append((s_array, c_array, a_array))
  return out_sample


def avg_sample_reward(sample_scar):
  with torch.no_grad():
    r_sum_avg = 0.
    for _, _, _, r_real_array in sample_scar:
      r_sum_avg += r_real_array.sum().item()
    r_sum_avg /= len(sample_scar)
  return r_sum_avg


def sample_batch(agent: MentalIQL_V2, sampler: _SamplerCommon, n_sample: int,
                 demo_sa_array):
  demo_sa_in = sampler.filter_demo(demo_sa_array)
  sample_sxar = sampler.collect(agent.policy.state_dict(),
                                n_sample,
                                fixed=False)
  sample_r = avg_sample_reward(sample_sxar)
  demo_sxa = convert_demo(demo_sa_in, agent)
  return sample_sxar, demo_sxa, sample_r


def learn(config: Config, msg="default"):

  env_type = config.env_type
  use_option = config.use_option
  n_demo = config.n_demo
  n_sample = config.n_sample
  n_thread = config.n_thread
  n_epoch = config.n_epoch
  seed = config.seed
  env_name = config.env_name
  use_state_filter = config.use_state_filter
  batch_size = config.mini_batch_size
  base_dir = config.base_dir
  critic_tau = 0.005
  critic_target_update_frequency = 1
  init_temp = 1e-2
  critic_betas = [0.9, 0.999]
  policy_betas = [0.9, 0.999]
  use_tanh = False
  learn_temp = False
  policy_update_frequency = 1
  alpha_lr = config.optimizer_lr_policy
  alpha_betas = [0.9, 0.999]
  clip_grad_val = 0.5
  is_sqil = False
  bounded_actor = config.bounded_actor
  gumbel_temperature = 1

  random.seed(seed)
  np.random.seed(seed)
  torch.random.manual_seed(seed)

  log_dir, save_dir, sample_name, _ = get_dirs(seed, base_dir, "miql_v2",
                                               env_type, env_name, msg,
                                               use_option)
  with open(os.path.join(save_dir, "config.log"), 'w') as f:
    f.write(str(config))
  logger = Logger(log_dir)
  save_name_f = lambda i: os.path.join(save_dir, f"miql_v2_{i}.torch")

  class_Env, fn_get_demo = env_class_and_demo_fn(env_type)

  env = class_Env(env_name)
  dim_s, dim_a = env.state_action_size()
  demo, _ = fn_get_demo(config, path=sample_name, n_demo=n_demo, display=False)

  critic = MentalCritic(dim_s, dim_a, config.dim_c, config.device,
                        config.shared_critic, config.activation,
                        config.hidden_critic, config.gamma, use_tanh)
  policy = MentalPolicy(dim_s,
                        dim_a,
                        config.dim_c,
                        config.device,
                        config.log_clamp_policy,
                        config.shared_policy,
                        config.activation,
                        config.hidden_policy,
                        config.hidden_option,
                        gumbel_temperature=gumbel_temperature,
                        bounded_actor=bounded_actor)

  agent = MentalIQL_V2(dim_s, dim_a, config.dim_c, batch_size, config.device,
                       config.gamma, critic_tau, config.optimizer_lr_critic,
                       critic_target_update_frequency, init_temp, critic_betas,
                       critic, policy, learn_temp, policy_update_frequency,
                       config.optimizer_lr_policy, policy_betas, alpha_lr,
                       alpha_betas, clip_grad_val)

  sampler = Sampler(seed,
                    env,
                    agent.policy,
                    use_state_filter=use_state_filter,
                    n_thread=n_thread)

  demo_sa_array = tuple(
      (s.to(agent.device), a.to(agent.device)) for s, a, r in demo)

  sample_sxar, demo_sxa, sample_r = sample_batch(agent, sampler, n_sample,
                                                 demo_sa_array)
  info_dict = reward_validate(sampler, agent.policy, do_print=True)

  logger.log_test_info(info_dict, 0)
  print(f"init: r-sample-avg={sample_r} ; {msg}")

  for i in range(n_epoch):
    sample_sxar, demo_sxar = sample_batch(agent, sampler, n_sample,
                                          demo_sa_array)

    train_iql(agent, sample_sxar, demo_sxa, batch_size, is_sqil=is_sqil)

    if (i + 1) % 20 == 0:
      info_dict = reward_validate(sampler, agent.policy, do_print=True)

      torch.save((agent.state_dict(), sampler.state_dict()), save_name_f(i))
      logger.log_test_info(info_dict, i)
      print(f"{i}: r-sample-avg={sample_r}", f"{msg}")
    else:
      print(f"{i}: r-sample-avg={sample_r}, {msg}")
    logger.log_train("r-sample-avg", sample_r, i)
    logger.flush()
