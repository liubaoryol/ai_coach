import torch
import torch.nn as nn
import numpy as np
from aicoach_baselines.option_gail.utils.config import Config
from .nn_models import (SimpleOptionQNetwork, DoubleOptionQCritic,
                        DiagGaussianOptionActor)
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

    self.device = torch.device(config.device)
    self.PREV_LATENT = lat_dim
    self.PREV_ACTION = (float("nan") if discrete_act else np.zeros(
        self.action_dim, dtype=np.float32))

    self.demo_latent_infer_interval = config.demo_latent_infer_interval

    config_tx, config_pi = get_tx_pi_config(config)

    self.tx_agent = IQLOptionSoftQ(config_tx, obs_dim, lat_dim, lat_dim + 1,
                                   discrete_obs, SimpleOptionQNetwork,
                                   self._get_tx_iq_vars)

    if discrete_act:
      self.pi_agent = IQLOptionSoftQ(config_pi, obs_dim, action_dim, lat_dim,
                                     discrete_obs, SimpleOptionQNetwork,
                                     self._get_pi_iq_vars)
    else:
      actor = DiagGaussianOptionActor(
          obs_dim, action_dim, lat_dim, config_pi.hidden_policy,
          config_pi.activation, config_pi.log_std_bounds,
          config_pi.bounded_actor, config_pi.use_nn_logstd,
          config_pi.clamp_action_logstd)
      self.pi_agent = IQLOptionSAC(config_pi, obs_dim, action_dim, lat_dim,
                                   discrete_obs, DoubleOptionQCritic, actor,
                                   self._get_pi_iq_vars)

    self.train()

  def train(self, training=True):
    self.training = training
    self.tx_agent.train(training)
    self.pi_agent.train(training)

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
    PI_IS_SQIL, PI_USE_TARGET, PI_DO_SOFT_UPDATE = False, True, True

    pi_loss = self.pi_agent.iq_update(policy_batch, expert_batch, logger, step,
                                      PI_IS_SQIL, PI_USE_TARGET,
                                      PI_DO_SOFT_UPDATE,
                                      self.pi_agent.method_loss,
                                      self.pi_agent.method_regularize)
    return pi_loss

  def tx_update(self, policy_batch, expert_batch, logger, step):
    TX_IS_SQIL, TX_USE_TARGET, TX_DO_SOFT_UPDATE = False, True, True
    tx_loss = self.tx_agent.iq_update(policy_batch, expert_batch, logger, step,
                                      TX_IS_SQIL, TX_USE_TARGET,
                                      TX_DO_SOFT_UPDATE,
                                      self.tx_agent.method_loss,
                                      self.tx_agent.method_regularize)
    return tx_loss

  def miql_update(self, policy_batch, expert_batch, logger, step):
    # update pi first and then tx
    ALWAYS_UPDATE_BOTH = 1
    UPDATE_IN_ORDER = 2
    UPDATE_ALTERNATIVELY = 3

    RATIO_PI_UPDATE = 0.7

    update_method = ALWAYS_UPDATE_BOTH
    internal_step = step % self.demo_latent_infer_interval

    tx_loss, pi_loss = {}, {}
    if update_method == ALWAYS_UPDATE_BOTH:
      pi_loss = self.pi_update(policy_batch, expert_batch, logger, step)
      tx_loss = self.tx_update(policy_batch, expert_batch, logger, step)
    elif update_method == UPDATE_IN_ORDER:
      if internal_step < RATIO_PI_UPDATE * self.demo_latent_infer_interval:
        pi_loss = self.pi_update(policy_batch, expert_batch, logger, step)
      else:
        tx_loss = self.tx_update(policy_batch, expert_batch, logger, step)
    elif update_method == UPDATE_ALTERNATIVELY:
      NUM_PI_UPDATE = 10
      NUM_TX_UPDATE = 5
      alternating_step = internal_step % (NUM_PI_UPDATE + NUM_TX_UPDATE)
      if alternating_step < NUM_PI_UPDATE:
        pi_loss = self.pi_update(policy_batch, expert_batch, logger, step)
      else:
        tx_loss = self.tx_update(policy_batch, expert_batch, logger, step)

    return tx_loss, pi_loss

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
