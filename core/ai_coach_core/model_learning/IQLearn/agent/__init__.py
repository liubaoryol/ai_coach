import gym
from gym.spaces import Discrete, MultiDiscrete, Box
from .sac_models import DiscreteActor, DiagGaussianActor, SoftDiscreteActor
from ai_coach_core.model_learning.IQLearn.agent.sac import SAC
from ai_coach_core.model_learning.IQLearn.agent.sac_discrete import SAC_Discrete
from ai_coach_core.model_learning.IQLearn.agent.softq import SoftQ
from .softq_models import SimpleQNetwork, SingleQCriticDiscrete
from .sac_models import DoubleQCritic
from aicoach_baselines.option_gail.utils.config import Config


def make_softq_agent(config: Config, env: gym.Env):
  # gamma: 0.99
  # critic_tau = 0.1
  # critic_target_update_frequency = 4
  q_net_base = SimpleQNetwork

  if isinstance(env.observation_space, Discrete):
    obs_dim = env.observation_space.n
    discrete_obs = True
  else:
    obs_dim = env.observation_space.shape[0]
    discrete_obs = False

  if not isinstance(env.action_space, Discrete):
    raise RuntimeError(
        "Invalid action space: only discrete action is supported")

  action_dim = env.action_space.n

  obs_dim = int(obs_dim)
  action_dim = int(action_dim)
  agent = SoftQ(config, obs_dim, action_dim, discrete_obs, q_net_base)

  return agent


def make_sac_agent(config: Config, env: gym.Env):
  'discrete observation may not work well'

  critic_base = DoubleQCritic
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
    actor = SoftDiscreteActor(config, obs_dim, action_dim)
  else:
    action_dim = env.action_space.shape[0]
    actor = DiagGaussianActor(config, obs_dim, action_dim)

  agent = SAC(config, obs_dim, action_dim, discrete_obs, critic_base, actor)

  return agent


def make_sacd_agent(config: Config, env: gym.Env):
  # batch_size,
  # device_name,
  # critic_base,
  # gamma: float = 0.99,
  # critic_tau: float = 0.005,
  # critic_lr: float = 3e-4,
  # critic_target_update_frequency: int = 1,
  # init_temp: float = 1e-2,
  # critic_betas=[0.9, 0.999],
  # use_tanh: bool = False,
  # learn_temp: bool = False,
  # actor_update_frequency: int = 1,
  # actor_lr: float = 3e-4,
  # actor_betas=[0.9, 0.999],
  # alpha_lr: float = 3e-4,
  # alpha_betas=[0.9, 0.999],
  # list_critic_hidden_dims=[256, 256],
  # list_actor_hidden_dims=[256, 256],
  # clip_grad_val=None):
  'discrete observation may not work well'
  critic_base = SingleQCriticDiscrete

  if isinstance(env.observation_space, Discrete):
    obs_dim = env.observation_space.n
    discrete_obs = True
  else:
    obs_dim = env.observation_space.shape[0]
    discrete_obs = False

  if not (isinstance(env.action_space, Discrete)):
    raise RuntimeError(
        "Invalid action space: Only Discrete action spaces supported")

  action_dim = env.action_space.n
  actor = DiscreteActor(config, obs_dim, action_dim)

  agent = SAC_Discrete(config, obs_dim, action_dim, discrete_obs, critic_base,
                       actor)

  return agent
