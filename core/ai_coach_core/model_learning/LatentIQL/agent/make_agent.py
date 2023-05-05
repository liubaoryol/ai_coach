from typing import Type
import gym
from gym.spaces import Discrete, Box
from .mental_models import (SoftDiscreteMentalActor, DiagGaussianMentalActor,
                            SoftDiscreteMentalThinker, MentalDoubleQCritic)
from .mental_iql import MentalIQL
from aicoach_baselines.option_gail.utils.config import Config


def make_miql_agent(config: Config, env: gym.Env):
  'discrete observation may not work well'

  latent_dim = config.dim_c
  if isinstance(env.observation_space, Discrete):
    obs_dim = env.observation_space.n
    discrete_obs = True
  else:
    obs_dim = env.observation_space.shape[0]
    discrete_obs = False

  if not (isinstance(env.action_space, Discrete)
          or isinstance(env.action_space, Box)):
    raise RuntimeError(
        "Invalid action space: Only Discrete and Box action spaces supported")

  if isinstance(env.action_space, Discrete):
    action_dim = env.action_space.n
    actor = SoftDiscreteMentalActor(config, obs_dim, action_dim, latent_dim)
  else:
    action_dim = env.action_space.shape[0]
    actor = DiagGaussianMentalActor(config, obs_dim, action_dim, latent_dim)

  thinker = SoftDiscreteMentalThinker(config, obs_dim, action_dim, latent_dim)
  critic = MentalDoubleQCritic(config, obs_dim, action_dim, latent_dim)

  agent = MentalIQL(config, obs_dim, action_dim, latent_dim, discrete_obs,
                    critic, actor, thinker)

  return agent
