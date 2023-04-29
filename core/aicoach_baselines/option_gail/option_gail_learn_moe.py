#!/usr/bin/env python3

import os
import torch
from .model.option_ppo import MoEPPO
from .model.option_gail import MoEGAIL
from .utils.utils import (reward_validate, set_seed, env_class_and_demo_fn)
from .utils.agent import Sampler
from .utils.logger import Logger
from .utils.config import Config


def make_gail(config: Config, dim_s, dim_a):
  gail = MoEGAIL(config, dim_s=dim_s, dim_a=dim_a)
  ppo = MoEPPO(config, gail.policy)
  return gail, ppo


def train_g(ppo: MoEPPO, sample_sxar, factor_lr, n_step=10):
  ppo.step(sample_sxar, lr_mult=factor_lr, n_step=n_step)


def train_d(gail: MoEGAIL, sample_sxar, demo_sxar, n_step=10):
  return gail.step(sample_sxar, demo_sxar, n_step=n_step)


def sample_batch(gail: MoEGAIL, agent, n_sample, demo_sa_array):
  demo_sa_in = agent.filter_demo(demo_sa_array)
  sample_sar_in = agent.collect(gail.policy.state_dict(), n_sample, fixed=False)
  sample_sar, sample_rsum = gail.convert_sample(sample_sar_in)
  demo_sar, demo_rsum = gail.convert_demo(demo_sa_in)
  return sample_sar, demo_sar, sample_rsum, demo_rsum


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
  n_epoch = config.n_epoch
  seed = config.seed
  env_name = config.env_name
  use_state_filter = config.use_state_filter
  base_dir = config.base_dir

  set_seed(seed)

  with open(os.path.join(save_dir, "config.log"), 'w') as f:
    f.write(str(config))
  logger = Logger(log_dir)
  save_name_f = lambda i: os.path.join(save_dir, f"gail_{i}.torch")

  class_Env, fn_get_demo = env_class_and_demo_fn(env_type)

  env = class_Env(env_name)
  dim_s, dim_a = env.state_action_size()
  demo, _ = fn_get_demo(config, path=sample_name, n_demo=n_demo, display=False)

  gail, ppo = make_gail(config, dim_s=dim_s, dim_a=dim_a)
  sampling_agent = Sampler(seed,
                           env,
                           gail.policy,
                           use_state_filter=use_state_filter,
                           n_thread=n_thread)

  demo_sa_array = tuple(
      (s.to(gail.device), a.to(gail.device)) for s, a, r in demo)

  sample_sxar, demo_sxar, sample_r, demo_r = sample_batch(
      gail, sampling_agent, n_sample, demo_sa_array)
  info_dict, cs_sample = reward_validate(sampling_agent,
                                         gail.policy,
                                         do_print=True)

  logger.log_test_info(info_dict, 0)
  print(f"init: r-sample-avg={sample_r}, d-demo-avg={demo_r} ; {msg}")

  for i in range(n_epoch):
    sample_sar, demo_sxar, sample_r, demo_r = sample_batch(
        gail, sampling_agent, n_sample, demo_sa_array)

    train_d(gail, sample_sar, demo_sxar)
    # factor_lr = lr_factor_func(i, 1000., 1., 0.0001)
    train_g(ppo, sample_sar, factor_lr=1.)
    if (i + 1) % 20 == 0:
      info_dict, cs_sample = reward_validate(sampling_agent,
                                             gail.policy,
                                             do_print=True)

      torch.save((gail.state_dict(), sampling_agent.state_dict()),
                 save_name_f(i))
      logger.log_test_info(info_dict, i)
      print(f"{i}: r-sample-avg={sample_r}, d-demo-avg={demo_r} ; {msg}")
    else:
      print(f"{i}: r-sample-avg={sample_r}, d-demo-avg={demo_r} ; {msg}")
    logger.log_train("r-sample-avg", sample_r, i)
    logger.log_train("r-demo-avg", demo_r, i)
    logger.flush()
