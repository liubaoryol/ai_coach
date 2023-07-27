#!/usr/bin/env python3

import numpy as np
import os
import torch
import random
from itertools import count
from aicoach_baselines.option_gail.utils.utils import env_class_and_demo_fn
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
              config,
              smpl_scar,
              demo_sca,
              mini_bs,
              logger,
              learn_step,
              is_sqil,
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

      agent.iq_update(policy_batch, expert_batch, logger, learn_step, is_sqil,
                      use_target, do_soft_update, method_loss,
                      method_regularize)


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
  sample_avgstep = (sum([sxar[-1].size(0)
                         for sxar in sample_sxar]) / len(sample_sxar))

  return sample_sxar, demo_sxa, sample_r, sample_avgstep


def learn(config: Config,
          log_dir,
          save_dir,
          sample_name,
          pretrain_name,
          msg="default"):

  env_type = config.env_type
  n_demo = config.n_demo
  n_sample = config.n_sample
  n_thread = config.n_thread
  max_exp_step = config.max_explore_step
  seed = config.seed
  env_name = config.env_name
  use_state_filter = config.use_state_filter
  batch_size = config.mini_batch_size

  is_sqil = False
  use_target = True
  do_soft_update = True
  method_loss = config.method_loss
  method_regularize = config.method_regularize

  random.seed(seed)
  np.random.seed(seed)
  torch.random.manual_seed(seed)

  with open(os.path.join(save_dir, "config.log"), 'w') as f:
    f.write(str(config))
  logger = Logger(log_dir)
  save_name_f = lambda i: os.path.join(save_dir, f"miql_v2_{i}.torch")

  class_Env, fn_get_demo = env_class_and_demo_fn(env_type)

  env = class_Env(env_name)
  dim_s, dim_a = env.state_action_size()
  demo, _ = fn_get_demo(config, path=sample_name, n_demo=n_demo, display=False)

  critic = MentalCritic(config, dim_s, dim_a, config.dim_c)
  policy = MentalPolicy(config, dim_s, dim_a, config.dim_c)

  agent = MentalIQL_V2(config, dim_s, dim_a, config.dim_c, critic, policy)

  sampler = Sampler(seed,
                    env,
                    agent.policy,
                    use_state_filter=use_state_filter,
                    n_thread=n_thread)

  demo_sa_array = tuple(
      (s.to(agent.device), a.to(agent.device)) for s, a, r in demo)

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
              explore_step, is_sqil, use_target, do_soft_update, method_loss,
              method_regularize)
    explore_step += sum([len(traj[0]) for traj in sample_sxar])

    if (i + 1) % 10 == 0:
      info_dict = reward_validate(sampler, agent.policy, do_print=True)

      # torch.save((agent.state_dict(), sampler.state_dict()),
      #            save_name_f(explore_step))
      logger.log_test_info(info_dict, explore_step)
    logger.flush()
