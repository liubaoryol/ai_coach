from typing import Type
import gym
from gym.spaces import Discrete, Box
from .mental_models import (SoftDiscreteMentalActor, DiagGaussianMentalActor,
                            SoftDiscreteMentalThinker, MentalDoubleQCritic)
from .mental_iql import MentalIQL


def make_miql_agent(env: gym.Env,
                    batch_size,
                    device_name,
                    lat_dim,
                    gamma: float = 0.99,
                    critic_tau: float = 0.005,
                    critic_lr: float = 3e-4,
                    critic_target_update_frequency: int = 1,
                    init_temp: float = 1e-2,
                    critic_betas=[0.9, 0.999],
                    use_tanh: bool = False,
                    learn_temp: bool = False,
                    actor_update_frequency: int = 1,
                    actor_lr: float = 3e-4,
                    actor_betas=[0.9, 0.999],
                    thinker_lr: float = 3e-4,
                    thinker_betas=[0.9, 0.999],
                    alpha_lr: float = 3e-4,
                    alpha_betas=[0.9, 0.999],
                    list_critic_hidden_dims=[256, 256],
                    list_actor_hidden_dims=[256, 256],
                    list_thinker_hidden_dims=[256, 256],
                    log_std_bounds=[-5, 2],
                    gumbel_temperature=0.5,
                    clip_grad_val=None,
                    bounded_actor=True,
                    use_prev_action=True):
  'discrete observation may not work well'

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
    actor = SoftDiscreteMentalActor(obs_dim, action_dim, lat_dim,
                                    list_actor_hidden_dims, gumbel_temperature)
  else:
    action_dim = env.action_space.shape[0]
    actor = DiagGaussianMentalActor(obs_dim, action_dim, lat_dim,
                                    list_actor_hidden_dims, log_std_bounds,
                                    bounded_actor)

  thinker = SoftDiscreteMentalThinker(obs_dim, action_dim, lat_dim,
                                      list_thinker_hidden_dims,
                                      gumbel_temperature, use_prev_action)
  critic = MentalDoubleQCritic(obs_dim, action_dim, lat_dim,
                               list_critic_hidden_dims, gamma, use_tanh,
                               use_prev_action)

  agent = MentalIQL(obs_dim, action_dim, lat_dim, batch_size, discrete_obs,
                    device_name, gamma, critic_tau, critic_lr,
                    critic_target_update_frequency, init_temp, critic_betas,
                    critic, actor, thinker, learn_temp, actor_update_frequency,
                    actor_lr, actor_betas, thinker_lr, thinker_betas, alpha_lr,
                    alpha_betas, clip_grad_val)

  return agent
