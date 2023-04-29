#!/usr/bin/env python3

import os
import torch
from typing import Union
import matplotlib.pyplot as plt
from .model.option_ppo import OptionPPO, PPO
from .model.option_gail import OptionGAIL, GAIL
from .utils.utils import (env_class_and_demo_fn, validate, reward_validate,
                          set_seed)
from .utils.agent import Sampler
from .utils.logger import Logger
from .utils.config import Config
from .utils.pre_train import pretrain


def make_gail(config: Config, dim_s, dim_a):
  use_option = config.use_option

  if use_option:
    gail = OptionGAIL(config, dim_s=dim_s, dim_a=dim_a)
    ppo = OptionPPO(config, gail.policy)
  else:
    gail = GAIL(config, dim_s=dim_s, dim_a=dim_a)
    ppo = PPO(config, gail.policy)
  return gail, ppo


def train_g(ppo: Union[OptionPPO, PPO], sample_sxar, factor_lr, n_step=10):
  if isinstance(ppo, OptionPPO):
    ppo.step(sample_sxar, lr_mult=factor_lr, n_step=n_step)
  else:
    ppo.step(sample_sxar, lr_mult=factor_lr, n_step=n_step)


def train_d(gail: Union[OptionGAIL, GAIL], sample_sxar, demo_sxar, n_step=10):
  return gail.step(sample_sxar, demo_sxar, n_step=n_step)


def sample_batch(gail: Union[OptionGAIL, GAIL], agent, n_sample, demo_sa_array):
  demo_sa_in = agent.filter_demo(demo_sa_array)
  sample_sxar_in = agent.collect(gail.policy.state_dict(),
                                 n_sample,
                                 fixed=False)
  sample_sxar, sample_rsum = gail.convert_sample(sample_sxar_in)
  demo_sxar, demo_rsum = gail.convert_demo(demo_sa_in)
  return sample_sxar, demo_sxar, sample_rsum, demo_rsum


def learn(config: Config,
          log_dir,
          save_dir,
          sample_name,
          pretrain_name,
          msg="default"):

  env_type = config.env_type
  use_pretrain = config.use_pretrain
  use_option = config.use_option
  n_demo = config.n_demo
  n_sample = config.n_sample
  n_thread = config.n_thread
  n_iter = config.n_pretrain_epoch
  n_epoch = config.n_epoch
  seed = config.seed
  log_interval = config.pretrain_log_interval
  env_name = config.env_name
  use_state_filter = config.use_state_filter
  use_d_info_gail = config.use_d_info_gail

  set_seed(seed)

  with open(os.path.join(save_dir, "config.log"), 'w') as f:
    f.write(str(config))
  logger = Logger(log_dir)
  save_name_pre_f = lambda i: os.path.join(save_dir, f"pre_{i}.torch")
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

  if use_pretrain or use_d_info_gail:
    opt_sd = None
    if use_d_info_gail:
      import copy
      opt_sd = copy.deepcopy(gail.policy.policy.state_dict())
    if os.path.isfile(pretrain_name):
      print(f"Loading pre-train model from {pretrain_name}")
      param, filter_state = torch.load(pretrain_name)
      gail.policy.load_state_dict(param)
      sampling_agent.load_state_dict(filter_state)
    else:
      pretrain(gail.policy,
               sampling_agent,
               demo_sa_array,
               save_name_pre_f,
               logger,
               msg,
               n_iter,
               log_interval,
               in_pretrain=True)
    if use_d_info_gail:
      gail.policy.policy.load_state_dict(opt_sd)

  sample_sxar, demo_sxar, sample_r, demo_r = sample_batch(
      gail, sampling_agent, n_sample, demo_sa_array)
  v_l, cs_demo = validate(gail.policy, [(tr[0], tr[-2]) for tr in demo_sxar])
  info_dict, cs_sample = reward_validate(sampling_agent,
                                         gail.policy,
                                         do_print=True)

  LOG_PLOT = False
  if LOG_PLOT and use_option:
    a = plt.figure()
    a.gca().plot(cs_demo[0][1:])
    logger.log_test_fig("expert_c", a, 0)

    a = plt.figure()
    a.gca().plot(cs_sample[0][1:])
    logger.log_test_fig("sample_c", a, 0)

  logger.log_test_info(info_dict, 0)
  print(
      f"init: r-sample-avg={sample_r}, d-demo-avg={demo_r}, log_p={v_l} ; {msg}"
  )

  for i in range(n_epoch):
    sample_sxar, demo_sxar, sample_r, demo_r = sample_batch(
        gail, sampling_agent, n_sample, demo_sa_array)

    train_d(gail, sample_sxar, demo_sxar)
    # factor_lr = lr_factor_func(i, 1000., 1., 0.0001)
    train_g(ppo, sample_sxar, factor_lr=1.)
    if (i + 1) % 20 == 0:
      v_l, cs_demo = validate(gail.policy,
                              [(tr[0], tr[-2]) for tr in demo_sxar])
      logger.log_test("expert_logp", v_l, i)
      info_dict, cs_sample = reward_validate(sampling_agent,
                                             gail.policy,
                                             do_print=True)
      if LOG_PLOT and use_option:
        a = plt.figure()
        a.gca().plot(cs_demo[0][1:])
        logger.log_test_fig("expert_c", a, i)

        a = plt.figure()
        a.gca().plot(cs_sample[0][1:])
        logger.log_test_fig("sample_c", a, i)

      torch.save((gail.state_dict(), sampling_agent.state_dict()),
                 save_name_f(i))
      logger.log_test_info(info_dict, i)
      print(f"{i}: r-sample-avg={sample_r}, d-demo-avg={demo_r}, log_p={v_l} ;",
            f"{msg}")
    else:
      print(f"{i}: r-sample-avg={sample_r}, d-demo-avg={demo_r} ; {msg}")
    logger.log_train("r-sample-avg", sample_r, i)
    logger.log_train("r-demo-avg", demo_r, i)
    logger.flush()
