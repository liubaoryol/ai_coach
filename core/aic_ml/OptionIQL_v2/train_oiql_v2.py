#!/usr/bin/env python3

import numpy as np
import os
import torch
import random
from itertools import count
from aic_ml.baselines.option_gail.utils.utils import (env_class, set_seed,
                                                      reward_validate,
                                                      load_n_convert_data)
from aic_ml.baselines.option_gail.utils.logger import Logger
from aic_ml.baselines.option_gail.utils.config import Config
from aic_ml.baselines.option_gail.utils.agent import _SamplerCommon
from .model.option_critic import OptionCritic
from .model.option_policy import OptionPolicy
from .model.option_iql_v2 import OptionIQL_V2
from .agent import Sampler


def train_iql(agent: OptionIQL_V2,
              config,
              smpl_scar,
              demo_sca,
              mini_bs,
              logger,
              learn_step,
              use_target,
              do_soft_update,
              method_loss,
              method_regularize,
              n_step=10):
  agent.reset_optimizers(config)
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

  p_a = torch.zeros_like(smpl_scar[0][2][0], device=agent.device)
  a_1p = torch.cat([torch.vstack([p_a, a[:-2]]) for s, c, a, r in smpl_scar],
                   dim=0)
  a_1e = torch.cat([torch.vstack([p_a, a[:-2]]) for s, c, a in demo_sca], dim=0)

  len_s = sp.size(0)
  len_e = se.size(0)
  rp = torch.zeros((len_s, 1), device=agent.device)
  re = torch.zeros((len_e, 1), device=agent.device)
  dp = torch.zeros((len_s, 1), device=agent.device)
  de = torch.zeros((len_e, 1), device=agent.device)

  for _ in range(n_step):
    inds = torch.randperm(len_s, device=agent.device)
    for ind_p in inds.split(mini_bs):
      sp_b, cp_1b, ap_b, cp_b = sp[ind_p], c_1p[ind_p], ap[ind_p], cp[ind_p]
      snp_b, ap_1b, rp_b, dp_b = snp[ind_p], a_1p[ind_p], rp[ind_p], dp[ind_p]

      ind_e = torch.randperm(len_e, device=agent.device)[:ind_p.size(0)]
      se_b, ce_1b, ae_b, ce_b = se[ind_e], c_1e[ind_e], ae[ind_e], ce[ind_e]
      sne_b, ae_1b, re_b, de_b = sne[ind_e], a_1e[ind_e], re[ind_e], de[ind_e]

      policy_batch = (sp_b, cp_1b, ap_1b, snp_b, cp_b, ap_b, rp_b, dp_b)
      expert_batch = (se_b, ce_1b, ae_1b, sne_b, ce_b, ae_b, re_b, de_b)

      agent.iq_update(policy_batch, expert_batch, logger, learn_step,
                      use_target, do_soft_update, method_loss,
                      method_regularize)


def convert_demo(demo_sa, agent: OptionIQL_V2):
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


def sample_batch(agent: OptionIQL_V2, sampler: _SamplerCommon, n_sample: int,
                 demo_sa_array):
  demo_sa_in = sampler.filter_demo(demo_sa_array)
  sample_sxar = sampler.collect(agent.policy.state_dict(),
                                n_sample,
                                fixed=False)
  sample_r = avg_sample_reward(sample_sxar)
  demo_sxa = convert_demo(demo_sa_in, agent)
  sample_avgstep = (sum([sxar[-1].size(0)
                         for sxar in sample_sxar]) / len(sample_sxar))

  return sample_sxar, demo_sxa, sample_r, sample_avgstep


def learn(config: Config,
          log_dir,
          save_dir,
          demo_path,
          pretrain_name,
          msg="default"):

  env_type = config.env_type
  n_traj = config.n_traj
  n_sample = config.n_sample
  n_thread = config.n_thread
  max_exp_step = config.max_explore_step
  seed = config.seed
  env_name = config.env_name
  use_state_filter = config.use_state_filter
  batch_size = config.mini_batch_size

  use_target = True
  do_soft_update = True
  method_loss = config.method_loss
  method_regularize = config.method_regularize

  set_seed(seed)

  logger = Logger(log_dir)

  class_Env = env_class(env_type)

  env = class_Env(env_name)
  dim_s, dim_a = env.state_action_size()

  # ----- prepare demo
  n_labeled = int(n_traj * config.supervision)
  device = torch.device(config.device)
  dim_c = config.dim_c

  (demo_sa_array, demo_labels, cnt_label, expert_avg,
   expert_std) = load_n_convert_data(demo_path, n_traj, n_labeled, device,
                                     dim_c, seed)

  critic = OptionCritic(config, dim_s, dim_a, config.dim_c)
  policy = OptionPolicy(config, dim_s, dim_a, config.dim_c)

  agent = OptionIQL_V2(config, dim_s, dim_a, config.dim_c, critic, policy)

  sampler = Sampler(seed,
                    env,
                    agent.policy,
                    use_state_filter=use_state_filter,
                    n_thread=n_thread)

  explore_step = 0
  for i in count():
    if explore_step >= max_exp_step:
      break

    sample_sxar, demo_sxa, sample_r, sample_avgstep = sample_batch(
        agent, sampler, n_sample, demo_sa_array)
    logger.log_train("r-sample-avg", sample_r, explore_step)
    logger.log_train("step-sample-avg", sample_avgstep, explore_step)
    print(f"{explore_step}: r-sample-avg={sample_r}, "
          f"step-sample-avg={sample_avgstep} ; {msg}")

    train_iql(agent, config, sample_sxar, demo_sxa, batch_size, logger,
              explore_step, use_target, do_soft_update, method_loss,
              method_regularize)
    explore_step += sum([len(traj[0]) for traj in sample_sxar])

    if (i + 1) % 10 == 0:
      info_dict, _ = reward_validate(sampler, agent.policy, do_print=True)

      # torch.save((agent.state_dict(), sampler.state_dict()),
      #            save_name_f(explore_step))
      logger.log_test_info(info_dict, explore_step)
    logger.flush()
