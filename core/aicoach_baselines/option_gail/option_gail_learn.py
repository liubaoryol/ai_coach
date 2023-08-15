#!/usr/bin/env python3

import os
import torch
from typing import Union
import matplotlib.pyplot as plt
from itertools import count
from .model.option_ppo import OptionPPO, PPO
from .model.option_gail import OptionGAIL, GAIL
from .utils.utils import (env_class_and_demo_fn, validate, reward_validate,
                          set_seed)
from .utils.agent import Sampler
from .utils.logger import Logger
from .utils.config import Config
from .utils.pre_train import pretrain
from ai_coach_core.model_learning.IQLearn.dataset.expert_dataset import (
    ExpertDataset)


def load_n_convert_data(demo_path, n_traj, n_labeled, device, dim_c, seed):
  expert_dataset = ExpertDataset(demo_path, n_traj, 1, seed + 42)
  trajectories = expert_dataset.trajectories

  cnt_label = 0
  demo_labels = []
  demo_sa_array = []
  for epi in range(n_traj):
    n_steps = len(trajectories["rewards"][epi])
    s_array = torch.as_tensor(trajectories["states"][epi],
                              dtype=torch.float32).reshape(n_steps, -1)
    a_array = torch.as_tensor(trajectories["actions"][epi],
                              dtype=torch.float32).reshape(n_steps, -1)
    r_array = torch.as_tensor(trajectories["rewards"][epi],
                              dtype=torch.float32).reshape(n_steps, -1)
    if "latents" in trajectories:
      x_array = torch.zeros(n_steps + 1, 1, dtype=torch.long, device=device)
      x_array[1:] = torch.as_tensor(trajectories["latents"][epi],
                                    dtype=torch.float32).reshape(n_steps, -1)
      x_array[0] = dim_c
    else:
      x_array = None

    demo_sa_array.append((s_array.to(device), a_array.to(device)))
    if epi < n_labeled:
      demo_labels.append(x_array.to(device))
      cnt_label += 1
    else:
      demo_labels.append(None)

  demo_sa_array = tuple(demo_sa_array)
  print("num_labeled:", cnt_label)

  return demo_sa_array, demo_labels, cnt_label


def make_gail(config: Config, dim_s, dim_a, discrete_s, discrete_a):
  use_option = config.use_option

  if use_option:
    gail = OptionGAIL(config,
                      dim_s=dim_s,
                      dim_a=dim_a,
                      discrete_s=discrete_s,
                      discrete_a=discrete_a)
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


def sample_batch(gail: Union[OptionGAIL, GAIL], agent, n_sample, demo_sa_array,
                 demo_labels):
  demo_sa_in = agent.filter_demo(demo_sa_array)
  sample_sxar_in = agent.collect(gail.policy.state_dict(),
                                 n_sample,
                                 fixed=False)
  sample_sxar, sample_rsum = gail.convert_sample(sample_sxar_in)
  demo_sxar, demo_rsum = gail.convert_demo(demo_sa_in, demo_labels)
  sample_avgstep = (sum([sxar[-1].size(0)
                         for sxar in sample_sxar]) / len(sample_sxar))
  return sample_sxar, demo_sxar, sample_rsum, demo_rsum, sample_avgstep


def learn(config: Config,
          log_dir,
          save_dir,
          demo_path,
          pretrain_name,
          msg="default"):

  env_type = config.env_type
  use_pretrain = config.use_pretrain
  use_option = config.use_option
  n_traj = config.n_traj
  n_sample = config.n_sample
  n_thread = config.n_thread
  n_iter = config.n_pretrain_epoch
  max_exp_step = config.max_explore_step
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
  discrete_s, discrete_a = env.is_discrete_state_action()

  # ----- prepare demo
  n_labeled = int(n_traj * config.supervision)
  device = torch.device(config.device)
  dim_c = config.dim_c

  demo_sa_array, demo_labels, cnt_label = load_n_convert_data(
      demo_path, n_traj, n_labeled, device, dim_c, seed)

  # sample.append((s_array, a_array, r_array, x_array))

  # demo, _ = fn_get_demo(config, path=sample_name, n_traj=n_traj, display=False)
  # demo_sa_array = tuple(
  #     (s.to(gail.device), a.to(gail.device)) for s, a, r in demo)

  best_model_save_name = os.path.join(
      save_dir, f"{env_name}_n{n_traj}_l{cnt_label}_best.torch")

  gail, ppo = make_gail(config,
                        dim_s=dim_s,
                        dim_a=dim_a,
                        discrete_s=discrete_s,
                        discrete_a=discrete_a)
  sampling_agent = Sampler(seed,
                           env,
                           gail.policy,
                           use_state_filter=use_state_filter,
                           n_thread=n_thread)

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

  explore_step = 0
  LOG_PLOT = False
  best_reward = -float("inf")
  for i in count():
    if explore_step >= max_exp_step:
      break

    sample_sxar, demo_sxar, sample_r, demo_r, sample_avgstep = sample_batch(
        gail, sampling_agent, n_sample, demo_sa_array, demo_labels)

    logger.log_train("r-sample-avg", sample_r, explore_step)
    logger.log_train("r-demo-avg", demo_r, explore_step)
    logger.log_train("step-sample-avg", sample_avgstep, explore_step)
    print(f"{explore_step}: r-sample-avg={sample_r}, d-demo-avg={demo_r}, "
          f"step-sample-avg={sample_avgstep} ; {msg}")

    train_d(gail, sample_sxar, demo_sxar)
    # factor_lr = lr_factor_func(i, 1000., 1., 0.0001)
    train_g(ppo, sample_sxar, factor_lr=1.)

    explore_step += sum([len(traj[0]) for traj in sample_sxar])
    if (i + 1) % 5 == 0:
      v_l, cs_demo = validate(gail.policy,
                              [(tr[0], tr[-2]) for tr in demo_sxar])
      logger.log_test("expert_logp", v_l, explore_step)
      info_dict, cs_sample = reward_validate(sampling_agent,
                                             gail.policy,
                                             do_print=True)
      if LOG_PLOT and use_option:
        a = plt.figure()
        a.gca().plot(cs_demo[0][1:])
        logger.log_test_fig("expert_c", a, explore_step)

        a = plt.figure()
        a.gca().plot(cs_sample[0][1:])
        logger.log_test_fig("sample_c", a, explore_step)

      if best_reward <= info_dict["r-avg"]:
        best_reward = info_dict["r-avg"]
        torch.save((gail.state_dict(), sampling_agent.state_dict()),
                   best_model_save_name)
      # torch.save((gail.state_dict(), sampling_agent.state_dict()),
      #            save_name_f(explore_step))
      logger.log_test_info(info_dict, explore_step)

    logger.flush()
