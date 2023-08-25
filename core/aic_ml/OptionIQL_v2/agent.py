import torch
import time
from copy import deepcopy
from torch.multiprocessing import Process, Pipe, Lock, Value
from .model.option_policy import OptionPolicy
from aic_baselines.option_gail.utils.state_filter import StateFilter
from aic_baselines.option_gail.utils.utils import set_seed
from aic_baselines.option_gail.utils.agent import (_Sampler, _SamplerCommon,
                                                       _SamplerSS)

__all__ = ["Sampler"]

# rlbench: 4096, 1t, 135s; 2t, 79s; 4t, 51s; 6t, 51s
# mujoco: 5000, 1t, 7.2s; 2t, 5.6s; 4t, 4.2s; 6t, 4.2s


def option_loop(env, policy: OptionPolicy, state_filter, fixed):
  with torch.no_grad():
    a_array = []
    c_array = []
    s_array = []
    r_array = []
    s, done = env.reset(random=not fixed), False
    ct = torch.empty(1, 1, dtype=torch.long,
                     device=policy.device).fill_(policy.dim_c)
    c_array.append(ct)
    while not done:
      st = torch.as_tensor(state_filter(s, fixed),
                           dtype=torch.float32,
                           device=policy.device).unsqueeze(0)
      ct = policy.sample_option(st, ct, fixed=fixed)[0].detach()
      at = policy.sample_action(st, ct, fixed=fixed)[0].detach()
      s_array.append(st)
      c_array.append(ct)
      a_array.append(at)
      s, r, done = env.step(at.cpu().squeeze(dim=0).numpy())
      r_array.append(r)
    a_array = torch.cat(a_array, dim=0)
    c_array = torch.cat(c_array, dim=0)
    s_array = torch.cat(s_array, dim=0)
    r_array = torch.as_tensor(r_array,
                              dtype=torch.float32,
                              device=policy.device).unsqueeze(dim=-1)
  return s_array, c_array, a_array, r_array


def Sampler(seed,
            env,
            policy,
            use_state_filter: bool = True,
            n_thread=4) -> _SamplerCommon:
  if isinstance(policy, OptionPolicy):
    loop_func = option_loop
  class_m = _Sampler if n_thread > 1 else _SamplerSS
  return class_m(seed, env, policy, use_state_filter, n_thread, loop_func)


if __name__ == "__main__":
  from torch.multiprocessing import set_start_method
  set_start_method("spawn")
