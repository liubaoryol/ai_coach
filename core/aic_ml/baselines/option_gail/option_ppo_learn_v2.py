#!/usr/bin/env python3

import os
import torch
from itertools import count
from .model.option_ppo_v2 import OptionPPOV2, PPOV2
from .model.option_policy_v2 import OptionPolicyV2, PolicyV2
from .utils.agent import Sampler
from .utils.utils import (lr_factor_func, sample_batch, reward_validate,
                          set_seed, env_class_and_demo_fn)
from .utils.logger import Logger
from omegaconf import DictConfig


def learn(config: DictConfig, log_dir, save_dir):
  env_type = config.env_type
  use_option = config.use_option
  env_name = config.env_name
  n_sample = config.n_sample
  n_thread = config.n_thread
  max_explore_step = config.max_explore_step
  seed = config.seed
  use_state_filter = config.use_state_filter

  msg = f"{config.alg_name}_{config.tag}"
  set_seed(seed)

  with open(os.path.join(save_dir, "config.log"), 'w') as f:
    f.write(str(config))
  logger = Logger(log_dir)

  # save_name_f = lambda i: os.path.join(save_dir, f"{i}.torch")
  best_model_save_name = os.path.join(save_dir, f"{env_name}_best.torch")

  class_Env, _ = env_class_and_demo_fn(env_type)

  env = class_Env(env_name)
  dim_s, dim_a = env.state_action_size()

  if use_option:
    policy = OptionPolicyV2(config, dim_s=dim_s, dim_a=dim_a)
    ppo = OptionPPOV2(config, policy)
  else:
    policy = PolicyV2(config, dim_s=dim_s, dim_a=dim_a)
    ppo = PPOV2(config, policy)

  sampling_agent = Sampler(seed,
                           env,
                           policy,
                           use_state_filter=use_state_filter,
                           n_thread=n_thread)
  n_epoch = int(max_explore_step / n_sample)
  explore_step = 0
  best_reward = -float("inf")
  for i in count():
    if explore_step >= max_explore_step:
      break

    sample_sxar, sample_r, avgsteps = sample_batch(policy, sampling_agent,
                                                   n_sample)
    print(f"{explore_step}: r-sample-avg={sample_r}, step-sample-avg={avgsteps}"
          f" ; {msg}")
    logger.log_train("r-sample-avg", sample_r, explore_step)
    logger.log_train("step-sample-avg", avgsteps, explore_step)

    explore_step += sum([len(traj[0]) for traj in sample_sxar])

    lr_mult = lr_factor_func(i, n_epoch, 1., 0.)
    losses = ppo.step(sample_sxar,
                      lr_mult=lr_mult,
                      n_step=config.n_update_rounds)
    logger.log_loss_info(losses, explore_step)

    if (i + 1) % 5 == 0:
      info_dict, cs_sample = reward_validate(sampling_agent, policy)
      if best_reward < info_dict["r-avg"]:
        best_reward = info_dict["r-avg"]
        torch.save((policy.state_dict(), sampling_agent.state_dict()),
                   best_model_save_name)
      logger.log_test_info(info_dict, explore_step)
    logger.flush()
