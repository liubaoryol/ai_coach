#!/usr/bin/env python3

import os
import math
import torch
from typing import Union
import torch.nn.functional as F
from .model.option_policy import (OptionPolicy, Policy)
from .utils.utils import (validate, reward_validate, set_seed, env_class,
                          load_n_convert_data)

from .utils.logger import Logger
from .utils.state_filter import StateFilter
from .utils.agent import Sampler
import numpy as np
from omegaconf import DictConfig


def policy_loss(optimizer, policy: Policy, sa_array, n_step=10):
  l_avg = 0.
  for _ in range(n_step):
    optimizer.zero_grad()
    ss_array = torch.cat([s_array for s_array, a_array in sa_array], dim=0)
    as_array = torch.cat([a_array for s_array, a_array in sa_array], dim=0)
    loss = F.mse_loss(policy.sample_action(ss_array, fixed=False), as_array)
    loss.backward()
    optimizer.step()
    l_avg += loss.item() / 10.
  return l_avg


def operate_sac(sa_array, dim_c, device):
  with torch.no_grad():
    sac_array = []
    for s_array, a_array in sa_array:
      demo_len = s_array.size(0)
      s = s_array.unsqueeze(dim=1).repeat(1, dim_c,
                                          1)  # demo_len x dim_c x dim_s
      a = a_array.unsqueeze(dim=1).repeat(1, dim_c,
                                          1)  # demo_len x dim_c x dim_a
      c = torch.arange(dim_c, dtype=torch.long,
                       device=device).view(1, dim_c).repeat(demo_len + 1, 1)
      c[0] = -1
      sac_array.append((s, a, c))
  return sac_array


def calculate_log_pi_tr(policy, s, a, c):
  log_pis = policy.log_prob_action(s.view(-1, policy.dim_s), c[1:].view(-1, 1),
                                   a.view(-1,
                                          policy.dim_a)).view(-1, policy.dim_c)
  log_trs = policy.log_trans(s.view(-1, policy.dim_s),
                             c[:-1].view(-1, 1)).view(-1, policy.dim_c,
                                                      policy.dim_c)
  return log_pis, log_trs


def calculate_log_ab(log_pis, log_trs, dim_c, device):
  log_alpha = [
      torch.empty(dim_c, dtype=torch.float32,
                  device=device).fill_(-math.log(dim_c))
  ]
  for log_tr, log_pi in zip(log_trs, log_pis):
    log_alpha_t = (log_alpha[-1].unsqueeze(dim=-1) +
                   log_tr).logsumexp(dim=0) + log_pi
    log_alpha.append(log_alpha_t)
  log_alpha = log_alpha[1:]

  log_beta = [torch.zeros(dim_c, dtype=torch.float32, device=device)]
  for log_tr, log_pi in zip(reversed(log_trs), reversed(log_pis)):
    log_beta_t = ((log_beta[-1] + log_pi).unsqueeze(dim=0) +
                  log_tr).logsumexp(dim=-1)
    log_beta.append(log_beta_t)
  log_beta.reverse()
  log_beta = log_beta[1:]

  log_alpha = torch.stack(log_alpha)
  log_beta = torch.stack(log_beta)
  return log_alpha, log_beta


def policy_loss_option_MLE(optimizer,
                           opolicy: OptionPolicy,
                           sa_array,
                           factor_ent=1.,
                           n_step=10):
  l_avg = 0.
  for _ in range(n_step):
    # np.random.shuffle(sa_array)
    for s, a in sa_array:
      optimizer.zero_grad()

      log_pis = opolicy.log_prob_action(s, None,
                                        a).view(-1,
                                                opolicy.dim_c)  # demo_len x ct
      log_trs = opolicy.log_trans(s, None)  # demo_len x (ct_1 + 1) x ct
      log_tr0 = log_trs[0, -1]
      log_trs = log_trs[1:, :-1]  # (demo_len-1) x ct_1 x ct

      log_alpha = [log_tr0 + log_pis[0]]
      for log_tr, log_pi in zip(log_trs, log_pis[1:]):
        log_alpha_t = (log_alpha[-1].unsqueeze(dim=-1) +
                       log_tr).logsumexp(dim=0) + log_pi
        log_alpha.append(log_alpha_t)

      log_alpha = torch.stack(log_alpha)
      entropy = -(log_trs * log_trs.exp()).sum(dim=-1).mean()
      log_p = (log_alpha.softmax(dim=-1).detach() * log_alpha).sum()
      loss = -log_p - factor_ent * entropy
      loss.backward()
      optimizer.step()
      l_avg += loss.item() / 10. / len(sa_array)
  return l_avg


def policy_loss_option_MAP(optimizer,
                           opolicy: OptionPolicy,
                           sa_array,
                           n_step=10):
  l_avg = 0.
  sac_pi_tr = []
  sac = operate_sac(sa_array, opolicy.dim_c, opolicy.device)
  with torch.no_grad():
    for s, a, c in sac:
      log_pis, log_trs = calculate_log_pi_tr(opolicy, s, a, c)
      log_alpha, log_beta = calculate_log_ab(log_pis, log_trs, opolicy.dim_c,
                                             opolicy.device)
      pi = (log_alpha + log_beta).softmax(dim=-1).detach()
      marg_a = torch.cat((torch.zeros_like(log_alpha[0:1]), log_alpha[:-1]),
                         dim=0)
      tr = (marg_a.unsqueeze(dim=-1) + (log_beta + log_pis).unsqueeze(dim=-2) +
            log_trs).softmax(dim=-1).detach()
      sac_pi_tr.append((s, a, c, pi, tr))

  for _ in range(n_step):
    np.random.shuffle(sac_pi_tr)
    for s, a, c, pi, tr in sac_pi_tr:
      optimizer.zero_grad()
      log_pis, log_trs = calculate_log_pi_tr(opolicy, s, a, c)

      L_RC = -(pi * log_pis).sum()
      L_TR = -(tr * log_trs).sum()
      loss = L_RC + L_TR
      loss.backward()
      optimizer.step()
      l_avg += loss.item() / 10. / len(sac_pi_tr)
  return l_avg


def policy_loss_option_MAP_5(optimizer,
                             opolicy: OptionPolicy,
                             sa_array,
                             n_part=5,
                             factor_ent=1.,
                             n_step=10):
  l_avg = 0.
  sac_array = operate_sac(sa_array, opolicy.dim_c, opolicy.device)
  sac_ab_parts = []
  with torch.no_grad():
    log_alpha_0 = torch.empty(
        opolicy.dim_c, dtype=torch.float32,
        device=opolicy.device).fill_(-math.log(opolicy.dim_c))
    for s, a, c in sac_array:
      log_pis, log_trs = calculate_log_pi_tr(opolicy, s, a, c)
      log_alpha, log_beta = calculate_log_ab(log_pis, log_trs, opolicy.dim_c,
                                             opolicy.device)
      hc = ((log_alpha * log_beta).softmax(dim=-1) *
            (log_alpha * log_beta).log_softmax(dim=-1)).sum(dim=-1)
      ind = hc[:-1].sort()[1][:n_part].sort()[0]
      last_i = 0
      sac_ab_parts.append([])
      for i in ind:
        sac_ab_parts[-1].append(
            (s[last_i:i + 1], a[last_i:i + 1], c[last_i:i + 2],
             log_alpha_0.detach() if last_i == 0 else log_alpha[last_i -
                                                                1].detach(),
             log_beta[i].detach()))
        last_i = i
      sac_ab_parts[-1].append(
          (s[last_i:], a[last_i:], c[last_i:], log_alpha[last_i - 1].detach(),
           log_beta[-1].detach()))
  for _ in range(n_step):
    np.random.shuffle(sac_ab_parts)
    for parts in sac_ab_parts:
      optimizer.zero_grad()
      loss = 0.
      for s, a, c, log_alpha, log_beta in parts:
        log_pis, log_trs = calculate_log_pi_tr(opolicy, s, a, c)

        for log_tr, log_pi in zip(log_trs, log_pis):
          log_alpha = (log_alpha.unsqueeze(dim=-1) +
                       log_tr).logsumexp(dim=0) + log_pi

        log_p = ((log_alpha + log_beta).softmax(dim=-1).detach() *
                 (log_alpha + log_beta)).sum()
        entropy = -(log_trs * log_trs.exp().view(
            -1, opolicy.dim_c, opolicy.dim_c)).sum(dim=-1).mean()
        loss = loss - log_p - factor_ent * entropy
      loss.backward()
      optimizer.step()
      l_avg += loss.item() / 10. / len(sac_ab_parts)
  return l_avg


def pretrain(policy: Union[OptionPolicy, Policy],
             sampler,
             sa_array,
             save_name_f,
             logger,
             msg,
             n_iter,
             log_interval,
             loss_type="MLE",
             in_pretrain=True):
  is_option = isinstance(policy, OptionPolicy)
  optimizer = torch.optim.Adam(policy.parameters(), weight_decay=1.e-3)

  log_test = logger.log_pretrain if in_pretrain else logger.log_test
  log_train = logger.log_pretrain if in_pretrain else logger.log_train
  # log_test_fig = (logger.log_pretrain_fig
  #                 if in_pretrain else logger.log_test_fig)
  log_test_info = (logger.log_pretrain_info
                   if in_pretrain else logger.log_test_info)

  sa_array = sampler.filter_demo(sa_array)

  for i in range(n_iter):
    if is_option:
      if loss_type == "MLE":
        loss = policy_loss_option_MLE(optimizer,
                                      policy,
                                      sa_array,
                                      factor_ent=20. * math.exp(-i / 5.) + 1.)
      elif loss_type == "MAP":
        loss = policy_loss_option_MAP(optimizer, policy, sa_array)
      elif loss_type == "MAP_5":
        loss = policy_loss_option_MAP_5(optimizer,
                                        policy,
                                        sa_array,
                                        factor_ent=20. * math.exp(-i / 5.) + 1.,
                                        n_part=5)
      else:
        raise ValueError()
    else:
      loss = policy_loss(optimizer, policy, sa_array)

    if (i + 1) % log_interval == 0:
      v_l, cs_expert = validate(policy, sa_array)
      log_test("expert_logp", v_l, i)
      info_dict, cs_sample = reward_validate(sampler,
                                             policy,
                                             n_sample=-8,
                                             do_print=True)

      log_test_info(info_dict, i)

      torch.save((policy.state_dict(), StateFilter().state_dict()),
                 save_name_f(i))
      print(f"pre-{i} ; loss={loss} ; log_p={v_l} ; {msg}")
    else:
      print(f"pre-{i} ; loss={loss} ; {msg}")
    log_train("loss", loss, i)
    logger.flush()


def make_policy(config: DictConfig, dim_s, dim_a):
  use_option = config.use_option

  if use_option:
    policy = OptionPolicy(config, dim_s=dim_s, dim_a=dim_a)
  else:
    policy = Policy(config, dim_s=dim_s, dim_a=dim_a)
  return policy


def learn(config: DictConfig, log_dir, save_dir, demo_path, pretrain_name):

  use_option = config.use_option
  env_name = config.env_name
  env_type = config.env_type
  n_traj = config.n_traj
  n_iter = config.n_pretrain_epoch
  seed = config.seed
  log_interval = config.pretrain_log_interval
  use_state_filter = config.use_state_filter
  loss_type = config.loss_type
  base_dir = config.base_dir

  msg = f"{config.alg_name}_{config.tag}"

  set_seed(seed)

  with open(os.path.join(save_dir, "config.log"), 'w') as f:
    f.write(str(config))
  logger = Logger(log_dir)
  save_name_f = lambda i: os.path.join(save_dir, f"{i}.torch")

  class_RLEnv = env_class(env_type)

  env = class_RLEnv(env_name)
  dim_s, dim_a = env.state_action_size()
  policy = make_policy(config, dim_s=dim_s, dim_a=dim_a)

  # ----- prepare demo
  n_labeled = int(n_traj * config.supervision)
  device = torch.device(config.device)
  dim_c = config.dim_c

  (demo_sa_array, demo_labels, cnt_label, expert_avg,
   expert_std) = load_n_convert_data(demo_path, n_traj, n_labeled, device,
                                     dim_c, seed)

  filter_state = StateFilter(False)

  sampler = Sampler(seed,
                    env,
                    policy,
                    use_state_filter=use_state_filter,
                    n_thread=2)
  sampler.load_state_dict(filter_state)

  pretrain(policy,
           sampler,
           demo_sa_array,
           save_name_f,
           logger,
           msg,
           n_iter,
           log_interval,
           loss_type,
           in_pretrain=False)
