import gym
from gym.spaces import Discrete, MultiDiscrete, Box
from .sac_models import DiscreteActor, DiagGaussianActor
from ai_coach_core.model_learning.IQLearn.agent.sac import SAC
from ai_coach_core.model_learning.IQLearn.agent.sac_discrete import SAC_Discrete
from ai_coach_core.model_learning.IQLearn.agent.softq import SoftQ


def make_softq_agent(env: gym.Env,
                     batch_size,
                     device_name,
                     critic_base,
                     gamma: float = 0.99,
                     critic_tau: float = 0.1,
                     critic_lr: float = 3e-4,
                     critic_target_update_frequency: int = 4,
                     init_temp: float = 1e-2,
                     critic_betas=[0.9, 0.999],
                     use_tanh: bool = False,
                     double_q: bool = False):

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
  agent = SoftQ(obs_dim, action_dim, batch_size, discrete_obs, device_name,
                gamma, critic_tau, critic_lr, critic_target_update_frequency,
                init_temp, critic_betas, critic_base, double_q, use_tanh)

  return agent


def make_sac_agent(env: gym.Env,
                   batch_size,
                   device_name,
                   critic_base,
                   gamma: float = 0.99,
                   critic_tau: float = 0.1,
                   critic_lr: float = 3e-4,
                   critic_target_update_frequency: int = 4,
                   init_temp: float = 1e-2,
                   critic_betas=[0.9, 0.999],
                   use_tanh: bool = False,
                   learn_temp: bool = False,
                   actor_update_frequency: int = 1,
                   actor_lr: float = 3e-4,
                   actor_betas=[0.9, 0.999],
                   alpha_lr: float = 3e-4,
                   alpha_betas=[0.9, 0.999],
                   critic_hidden_dim=256,
                   critic_hidden_depth=2,
                   actor_hidden_dim=256,
                   actor_hidden_depth=2,
                   log_std_bounds=[-5, 2],
                   gumbel_temperature=0.5):
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
    actor = DiscreteActor(obs_dim, action_dim, actor_hidden_dim,
                          actor_hidden_depth, gumbel_temperature)
  else:
    action_dim = env.action_space.shape[0]
    actor = DiagGaussianActor(obs_dim, action_dim, actor_hidden_dim,
                              actor_hidden_depth, log_std_bounds)

  agent = SAC(obs_dim, action_dim, batch_size, discrete_obs, device_name, gamma,
              critic_tau, critic_lr, critic_target_update_frequency, init_temp,
              critic_betas, use_tanh, critic_base, actor, learn_temp,
              actor_update_frequency, actor_lr, actor_betas, alpha_lr,
              alpha_betas, critic_hidden_dim, critic_hidden_depth)

  return agent


def make_sacd_agent(env: gym.Env,
                    batch_size,
                    device_name,
                    critic_base,
                    gamma: float = 0.99,
                    critic_tau: float = 0.1,
                    critic_lr: float = 3e-4,
                    critic_target_update_frequency: int = 4,
                    init_temp: float = 1e-2,
                    critic_betas=[0.9, 0.999],
                    use_tanh: bool = False,
                    learn_temp: bool = False,
                    actor_update_frequency: int = 1,
                    actor_lr: float = 3e-4,
                    actor_betas=[0.9, 0.999],
                    alpha_lr: float = 3e-4,
                    alpha_betas=[0.9, 0.999],
                    critic_hidden_dim=256,
                    critic_hidden_depth=2,
                    actor_hidden_dim=256,
                    actor_hidden_depth=2,
                    log_std_bounds=[-5, 2],
                    gumbel_temperature=0.5):
  'discrete observation may not work well'

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
  actor = DiscreteActor(obs_dim, action_dim, actor_hidden_dim,
                        actor_hidden_depth, gumbel_temperature)

  agent = SAC_Discrete(obs_dim, action_dim, batch_size, discrete_obs,
                       device_name, gamma, critic_tau, critic_lr,
                       critic_target_update_frequency, init_temp, critic_betas,
                       use_tanh, critic_base, actor, learn_temp,
                       actor_update_frequency, actor_lr, actor_betas, alpha_lr,
                       alpha_betas, critic_hidden_dim, critic_hidden_depth)

  return agent


# def make_agent(env: gym.Env,
#                batch_size,
#                device_name,
#                critic_base,
#                actor_base=None,
#                gamma: float = 0.99,
#                critic_tau: float = 0.1,
#                critic_lr: float = 3e-4,
#                critic_target_update_frequency: int = 4,
#                init_temp: float = 1e-2,
#                critic_betas=[0.9, 0.999],
#                use_tanh: bool = False,
#                double_q: bool = False,
#                learn_temp: bool = False,
#                actor_update_frequency: int = 1,
#                actor_lr: float = 3e-4,
#                actor_betas=[0.9, 0.999],
#                alpha_lr: float = 3e-4,
#                alpha_betas=[0.9, 0.999],
#                critic_hidden_dim=256,
#                critic_hidden_depth=2,
#                actor_hidden_dim=256,
#                actor_hidden_depth=2,
#                log_std_bounds=[-5, 2]):
#   if isinstance(env.observation_space, Discrete):
#     obs_dim = env.observation_space.n
#     discrete_obs = True
#   else:
#     obs_dim = env.observation_space.shape[0]
#     discrete_obs = False

#   if (isinstance(env.action_space, Discrete)
#       or isinstance(env.action_space, MultiDiscrete)):
#     print('--> Using Soft-Q agent')
#     if isinstance(env.action_space, Discrete):
#       action_dim = env.action_space.n
#     else:
#       action_dim = sum(env.action_space.nvec)
#     obs_dim = int(obs_dim)
#     action_dim = int(action_dim)
#     agent = SoftQ(obs_dim, action_dim, batch_size, discrete_obs, device_name,
#                   gamma, critic_tau, critic_lr, critic_target_update_frequency,
#                   init_temp, critic_betas, critic_base, double_q, use_tanh)
#   else:
#     print('--> Using SAC agent')
#     action_dim = env.action_space.shape[0]
#     action_range = [
#         float(env.action_space.low.min()),
#         float(env.action_space.high.max())
#     ]
#     obs_dim = obs_dim
#     action_dim = action_dim
#     agent = SAC(obs_dim, action_dim, action_range, batch_size, device_name,
#                 gamma, critic_tau, critic_lr, critic_target_update_frequency,
#                 init_temp, critic_betas, use_tanh, critic_base, actor_base,
#                 learn_temp, actor_update_frequency, actor_lr, actor_betas,
#                 alpha_lr, alpha_betas, critic_hidden_dim, critic_hidden_depth,
#                 actor_hidden_dim, actor_hidden_depth, log_std_bounds)

#   return agent
