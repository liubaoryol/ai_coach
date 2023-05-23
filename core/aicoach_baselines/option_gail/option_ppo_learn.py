#!/usr/bin/env python3

import os
import torch
from .model.option_ppo import PPO, OptionPPO
from .model.option_policy import OptionPolicy, Policy
from .utils.agent import Sampler
from .utils.utils import (lr_factor_func, sample_batch, reward_validate,
                          set_seed, env_class_and_demo_fn)
from .utils.logger import Logger
from .utils.config import Config


def learn(config: Config, log_dir, save_dir, msg="default"):
  env_type = config.env_type
  use_option = config.use_option
  env_name = config.env_name
  n_sample = config.n_sample
  n_thread = config.n_thread
  n_epoch = config.n_epoch
  seed = config.seed
  use_state_filter = config.use_state_filter

  set_seed(seed)

  with open(os.path.join(save_dir, "config.log"), 'w') as f:
    f.write(str(config))
  logger = Logger(log_dir)

  save_name_f = lambda i: os.path.join(save_dir, f"{i}.torch")

  class_Env, _ = env_class_and_demo_fn(env_type)

  env = class_Env(env_name)
  dim_s, dim_a = env.state_action_size()

  if use_option:
    policy = OptionPolicy(config, dim_s=dim_s, dim_a=dim_a)
    ppo = OptionPPO(config, policy)
  else:
    policy = Policy(config, dim_s=dim_s, dim_a=dim_a)
    ppo = PPO(config, policy)

  sampling_agent = Sampler(seed,
                           env,
                           policy,
                           use_state_filter=use_state_filter,
                           n_thread=n_thread)

  for i in range(n_epoch):
    sample_sxar, sample_r = sample_batch(policy, sampling_agent, n_sample)
    lr_mult = lr_factor_func(i, n_epoch, 1., 0.)
    ppo.step(sample_sxar, lr_mult=lr_mult)
    if (i + 1) % 50 == 0:
      info_dict, cs_sample = reward_validate(sampling_agent, policy)

      torch.save((policy.state_dict(), sampling_agent.state_dict()),
                 save_name_f(i))
      logger.log_test_info(info_dict, i)
    print(f"{i}: r-sample-avg={sample_r} ; {msg}")
    logger.log_train("r-sample-avg", sample_r, i)
    logger.flush()