import torch
import torch.nn as nn
import numpy as np
from aic_baselines.option_gail.utils.config import Config
from .nn_models import (SimpleOptionQNetwork, DoubleOptionQCritic,
                        SingleOptionQCritic, DiagGaussianOptionActor)
# from .option_softq import OptionSoftQ
# from .option_sac import OptionSAC
from .option_iql import IQLOptionSAC, IQLOptionSoftQ


def get_tx_pi_config(config: Config):
  tx_prefix = "miql_tx_"
  config_tx = Config()
  for key in config:
    if key[:len(tx_prefix)] == tx_prefix:
      config_tx[key[len(tx_prefix):]] = config[key]

  pi_prefix = "miql_pi_"
  config_pi = Config()
  for key in config:
    if key[:len(pi_prefix)] == pi_prefix:
      config_pi[key[len(pi_prefix):]] = config[key]

  config_pi["gamma"] = config_tx["gamma"] = config.gamma
  config_pi["device"] = config_tx["device"] = config.device

  return config_tx, config_pi


class MentalIQL:

  def __init__(self, config: Config, obs_dim, action_dim, lat_dim, discrete_obs,
               discrete_act):
    self.discrete_obs = discrete_obs
    self.obs_dim = obs_dim
    self.action_dim = action_dim
    self.lat_dim = lat_dim
    self.discrete_act = discrete_act

    self.update_strategy = config.miql_update_strategy
    self.update_tx_after_pi = config.miql_tx_after_pi
    self.alter_update_n_pi_tx = config.miql_alter_update_n_pi_tx
    self.order_update_pi_ratio = config.miql_order_update_pi_ratio
    self.tx_batch_size = min(config.miql_tx_tx_batch_size,
                             config.mini_batch_size)

    self.device = torch.device(config.device)
    self.PREV_LATENT = lat_dim
    self.PREV_ACTION = (float("nan") if discrete_act else np.zeros(
        self.action_dim, dtype=np.float32))
    self.internal_step = 0
    self.pi_update_count = 0
    self.tx_update_count = 0

    config_tx, config_pi = get_tx_pi_config(config)

    self.tx_agent = IQLOptionSoftQ(config_tx, obs_dim, lat_dim, lat_dim + 1,
                                   discrete_obs, SimpleOptionQNetwork,
                                   self._get_tx_iq_vars)

    if discrete_act:
      self.pi_agent = IQLOptionSoftQ(config_pi, obs_dim, action_dim, lat_dim,
                                     discrete_obs, SimpleOptionQNetwork,
                                     self._get_pi_iq_vars)
    else:
      if config.miql_pi_single_critic:
        critic_base = SingleOptionQCritic
      else:
        critic_base = DoubleOptionQCritic
      actor = DiagGaussianOptionActor(
          obs_dim, action_dim, lat_dim, config_pi.hidden_policy,
          config_pi.activation, config_pi.log_std_bounds,
          config_pi.bounded_actor, config_pi.use_nn_logstd,
          config_pi.clamp_action_logstd)
      self.pi_agent = IQLOptionSAC(config_pi, obs_dim, action_dim, lat_dim,
                                   discrete_obs, critic_base, actor,
                                   self._get_pi_iq_vars)

    self.train()

  def train(self, training=True):
    self.training = training
    self.tx_agent.train(training)
    self.pi_agent.train(training)

  def reset_optimizers(self):
    self.tx_agent.reset_optimizers()
    self.pi_agent.reset_optimizers()

  def _get_tx_iq_vars(self, batch):
    prev_lat, _, state, latent, _, next_state, _, _, done = batch
    vec_v_args = (state, prev_lat)
    vec_next_v_args = (next_state, latent)
    vec_actions = (latent, )
    return vec_v_args, vec_next_v_args, vec_actions, done

  def _get_pi_iq_vars(self, batch):
    _, _, state, latent, action, next_state, next_latent, _, done = batch
    vec_v_args = (state, latent)
    vec_next_v_args = (next_state, next_latent)
    vec_actions = (action, )
    return vec_v_args, vec_next_v_args, vec_actions, done

  def pi_update(self, policy_batch, expert_batch, logger, step):
    PI_IS_SQIL = False
    if self.discrete_act:
      pi_use_target, pi_soft_update = False, False
    else:
      pi_use_target, pi_soft_update = True, True

    pi_loss = self.pi_agent.iq_update(policy_batch, expert_batch, logger,
                                      self.pi_update_count, PI_IS_SQIL,
                                      pi_use_target, pi_soft_update,
                                      self.pi_agent.method_loss,
                                      self.pi_agent.method_regularize)
    self.pi_update_count += 1
    return pi_loss

  def tx_update(self, policy_batch, expert_batch, logger, step):
    TX_IS_SQIL, TX_USE_TARGET, TX_DO_SOFT_UPDATE = False, False, False
    tx_loss = self.tx_agent.iq_update(policy_batch[:self.tx_batch_size],
                                      expert_batch[:self.tx_batch_size], logger,
                                      self.tx_update_count, TX_IS_SQIL,
                                      TX_USE_TARGET, TX_DO_SOFT_UPDATE,
                                      self.tx_agent.method_loss,
                                      self.tx_agent.method_regularize)
    self.tx_update_count += 1
    return tx_loss

  def miql_update(self, policy_batch, expert_batch, num_updates_per_cycle,
                  logger, step):
    # update pi first and then tx
    ALWAYS_UPDATE_BOTH = 1
    UPDATE_IN_ORDER = 2
    UPDATE_ALTERNATIVELY = 3

    if self.internal_step >= num_updates_per_cycle:
      self.internal_step = 0

    self.internal_step += 1

    num_pi_update, num_tx_update = self.alter_update_n_pi_tx

    if self.update_tx_after_pi:
      fn_update_1, fn_update_2 = self.tx_update, self.pi_update
      ratio_1st = (1 - self.order_update_pi_ratio)
      num_1st = num_tx_update
    else:
      fn_update_1, fn_update_2 = self.pi_update, self.tx_update
      ratio_1st = self.order_update_pi_ratio
      num_1st = num_pi_update

    loss_1, loss_2 = {}, {}
    if self.update_strategy == ALWAYS_UPDATE_BOTH:
      loss_1 = fn_update_1(policy_batch, expert_batch, logger, step)
      loss_2 = fn_update_2(policy_batch, expert_batch, logger, step)
    elif self.update_strategy == UPDATE_IN_ORDER:
      if self.internal_step < ratio_1st * num_updates_per_cycle:
        loss_1 = fn_update_1(policy_batch, expert_batch, logger, step)
      else:
        loss_2 = fn_update_2(policy_batch, expert_batch, logger, step)
    elif self.update_strategy == UPDATE_ALTERNATIVELY:
      alternating_step = self.internal_step % (num_pi_update + num_tx_update)
      if alternating_step < num_1st:
        loss_1 = fn_update_1(policy_batch, expert_batch, logger, step)
      else:
        loss_2 = fn_update_2(policy_batch, expert_batch, logger, step)
    else:
      raise NotImplementedError

    return (loss_1, loss_2) if self.update_tx_after_pi else (loss_2, loss_1)

  def choose_action(self, state, prev_option, prev_action, sample=False):
    'for compatibility with OptionIQL evaluate function'
    option = self.tx_agent.choose_action(state, prev_option, sample)
    action = self.pi_agent.choose_action(state, option, sample)
    return option, action

  def choose_policy_action(self, state, option, sample=False):
    return self.pi_agent.choose_action(state, option, sample)

  def choose_mental_state(self, state, prev_option, sample=False):
    return self.tx_agent.choose_action(state, prev_option, sample)

  def save(self, path):
    self.tx_agent.save(path, "_tx")
    self.pi_agent.save(path, "_pi")
  
  def load(self, path):
    self.tx_agent.load(path + "_tx")
    self.pi_agent.load(path + "_pi")

  def infer_mental_states(self, state, action):
    '''
    return: options with the length of len_demo
    '''
    len_demo = len(state)

    with torch.no_grad():
      log_pis = self.pi_agent.log_probs(state, action).view(
          -1, 1, self.lat_dim)  # len_demo x 1 x ct
      log_trs = self.tx_agent.log_probs(state, None)  # len_demo x (ct_1+1) x ct
      log_prob = log_trs[:, :-1] + log_pis
      log_prob0 = log_trs[0, -1] + log_pis[0, 0]
      # forward
      max_path = torch.empty(len_demo,
                             self.lat_dim,
                             dtype=torch.long,
                             device=self.device)
      accumulate_logp = log_prob0
      max_path[0] = self.lat_dim
      for i in range(1, len_demo):
        accumulate_logp, max_path[i, :] = (accumulate_logp.unsqueeze(dim=-1) +
                                           log_prob[i]).max(dim=-2)
      # backward
      c_array = torch.zeros(len_demo + 1,
                            1,
                            dtype=torch.long,
                            device=self.device)
      log_prob_traj, c_array[-1] = accumulate_logp.max(dim=-1)
      for i in range(len_demo, 0, -1):
        c_array[i - 1] = max_path[i - 1][c_array[i]]
    return (c_array[1:].detach().cpu().numpy(),
            log_prob_traj.detach().cpu().numpy())
