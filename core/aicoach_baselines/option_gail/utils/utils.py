import torch
from ..model.option_policy import OptionPolicy, Policy
from ..model.option_policy_v2 import OptionPolicyV2
import numpy as np
from typing import Union
import os
import random


def sample_batch(policy: Union[OptionPolicy, Policy], agent, n_step):
  sample = agent.collect(policy.state_dict(), n_step, fixed=False)
  rsum = sum([sxar[-1].sum().item() for sxar in sample]) / len(sample)
  avgsteps = sum([sxar[-1].size(0) for sxar in sample]) / len(sample)
  return sample, rsum, avgsteps


def validate(policy: Union[OptionPolicy, Policy], sa_array):
  with torch.no_grad():
    log_pi = 0.
    cs = []
    for s_array, a_array in sa_array:
      if isinstance(policy, OptionPolicy) or isinstance(policy, OptionPolicyV2):
        c_array, logp = policy.viterbi_path(s_array, a_array)
        log_pi += logp.item()
        cs.append(c_array.detach().cpu().squeeze(dim=-1).numpy())
      else:
        log_pi += policy.log_prob_action(s_array, a_array).sum().item()
        cs.append([0.])
    log_pi /= len(sa_array)
  return log_pi, cs


def reward_validate(agent,
                    policy: Union[OptionPolicy, Policy],
                    n_sample=-8,
                    do_print=True):
  trajs = agent.collect(policy.state_dict(), n_sample, fixed=True)
  rsums = [tr[-1].sum().item() for tr in trajs]
  steps = [tr[-1].size(0) for tr in trajs]
  if isinstance(policy, OptionPolicy) or isinstance(policy, OptionPolicyV2):
    css = [
        tr[1].cpu().squeeze(dim=-1).numpy()
        for _, tr in sorted(zip(rsums, trajs), key=lambda d: d[0], reverse=True)
    ]
  else:
    css = None

  info_dict = {
      "r-max": np.max(rsums),
      "r-min": np.min(rsums),
      "r-avg": np.mean(rsums),
      "step-max": np.max(steps),
      "step-avg": np.mean(steps),
      "step-min": np.min(steps),
  }
  if do_print:
    print(f"R: [ {info_dict['r-min']:.02f} ~ {info_dict['r-max']:.02f},",
          f"avg: {info_dict['r-avg']:.02f} ],",
          f"L: [ {info_dict['step-min']} ~ {info_dict['step-max']}, ",
          f"avg: {info_dict['step-avg']:.02f} ]")
  return info_dict, css


def lr_factor_func(i_iter, end_iter, start=1., end=0.):
  if i_iter <= end_iter:
    return start - (start - end) * i_iter / end_iter
  else:
    return end


def env_class_and_demo_fn(env_type):
  if env_type == "mujoco":
    from .mujoco_env import MujocoEnv as RLEnv, get_demo
    return RLEnv, get_demo
  else:
    raise KeyError(f"Unknown envir type {env_type}")


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.random.manual_seed(seed)
