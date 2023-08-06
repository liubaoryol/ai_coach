import torch
import torch.nn as nn
from aicoach_baselines.option_gail.utils.config import Config
from .nn_models import (SimpleOptionQNetwork, DoubleOptionQCritic,
                        DiagGaussianOptionActor)
from .option_softq import OptionSoftQ
from .option_sac import OptionSAC


def get_tx_pi_config(config: Config):
  # TODO: implement
  config_tx = None
  config_pi = None

  return config_tx, config_pi


class MentalIQL:

  def __init__(self, config: Config, obs_dim, action_dim, lat_dim, discrete_obs,
               discrete_act, discrete_lat):
    self.discrete_obs = discrete_obs
    self.obs_dim = obs_dim
    self.action_dim = action_dim
    self.lat_dim = lat_dim

    self.device = torch.device(config.device)

    config_tx, config_pi = get_tx_pi_config(config)

    self.tx_agent = OptionSoftQ(config_tx, obs_dim, lat_dim, lat_dim + 1,
                                discrete_obs, SimpleOptionQNetwork)

    if discrete_act:
      self.pi_agent = OptionSoftQ(config_pi, obs_dim, action_dim, lat_dim,
                                  discrete_obs, SimpleOptionQNetwork)
    else:
      # TODO: implement
      actor = DiagGaussianOptionActor(obs_dim, action_dim, lat_dim,
                                      config.hidden_policy, config.activation,
                                      config.log_std_bounds,
                                      config.bounded_actor,
                                      config.use_nn_logstd,
                                      config.clamp_action_logstd)
      self.pi_agent = OptionSAC(config_pi, obs_dim, action_dim, lat_dim,
                                discrete_obs, DoubleOptionQCritic, actor)

  def miql_update(self):
    self.tx_agent

  def choose_action(self, state, option, sample=False):
    return self.pi_agent.choose_action(state, option, sample)

  def choose_mental_state(self, state, prev_option, sample=False):
    return self.tx_agent.choose_action(state, prev_option, sample)
