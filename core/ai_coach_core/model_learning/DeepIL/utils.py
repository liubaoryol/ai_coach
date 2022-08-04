import numpy as np
import torch
import time
import gym.spaces

from tqdm import tqdm
from .buffer import Buffer
from .algo.base import Expert
from .env import NormalizedEnv


def state_action_size(env: NormalizedEnv):
  state_size = 0
  discrete_state = False
  if isinstance(env.observation_space, gym.spaces.Box):
    state_size = env.observation_space.shape[0]
    discrete_state = False
  elif isinstance(env.observation_space, gym.spaces.Discrete):
    state_size = env.observation_space.n
    discrete_state = True
  else:
    raise NotImplementedError

  action_size = 0
  discrete_action = False
  if isinstance(env.action_space, gym.spaces.Box):
    action_size = env.action_space.shape[0]
    discrete_action = False
  elif isinstance(env.action_space, gym.spaces.Discrete):
    action_size = env.action_space.n
    discrete_action = True
  else:
    raise NotImplementedError

  return state_size, discrete_state, action_size, discrete_action


def collect_demo(env: NormalizedEnv,
                 latent_size: int,
                 discrete_latent: bool,
                 algo: Expert,
                 buffer_size: int,
                 device: torch.device,
                 p_rand: float,
                 seed: int = 0):
  """
    Collect demonstrations using the well-trained policy

    Parameters
    ----------
    env: NormalizedEnv
        environment to collect demonstrations
    algo: Expert
        well-trained algorithm used to collect demonstrations
    buffer_size: int
        size of the buffer, also the number of s-a pairs in the demonstrations
    device: torch.device
        cpu or cuda
    std: float
        standard deviation add to the policy
    p_rand: float
        with probability of p_rand, the policy will act randomly
    seed: int
        random seed

    Returns
    -------
    buffer: Buffer
        buffer of demonstrations
    mean_return: float
        average episode reward
    """
  env.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)

  (state_size, discrete_state, action_size,
   discrete_action) = state_action_size(env)

  buffer = Buffer(buffer_size=buffer_size,
                  state_size=state_size,
                  latent_size=latent_size,
                  action_size=action_size,
                  discrete_state=discrete_state,
                  discrete_latent=discrete_latent,
                  discrete_action=discrete_action,
                  device=device)

  total_return = 0.0
  num_steps = []
  num_episodes = 0

  state = env.reset()
  t = 0
  latent = algo.get_latent(0, state)
  episode_return = 0.0
  episode_steps = 0

  for _ in tqdm(range(1, buffer_size + 1)):
    t += 1

    if np.random.rand() < p_rand:
      action = env.action_space.sample()
    else:
      action = algo.exploit(state, latent)

    next_state, reward, done, _ = env.step(action)
    if algo.cb_reward:
      reward = algo.cb_reward(state, latent, action, reward)
    next_latent = algo.get_latent(t, next_state, latent, action, state)
    mask = True if t == env.max_episode_steps else done
    buffer.append(state, latent, action, reward, mask, next_state, next_latent)
    episode_return += reward
    episode_steps += 1

    state = next_state
    latent = next_latent

    if done or t == env.max_episode_steps:
      num_episodes += 1
      total_return += episode_return
      state = env.reset()
      t = 0
      latent = algo.get_latent(0, state)
      episode_return = 0.0
      num_steps.append(episode_steps)
      episode_steps = 0

  mean_return = total_return / num_episodes
  print(f'Mean return of the expert is {mean_return}')
  print(f'Max episode steps is {np.max(num_steps)}')
  print(f'Min episode steps is {np.min(num_steps)}')

  return buffer, mean_return


def evaluation(env: NormalizedEnv,
               algo: Expert,
               episodes: int,
               render: bool,
               seed: int = 0,
               delay: float = 0.03):
  """
    Evaluate the well-trained policy

    Parameters
    ----------
    env: NormalizedEnv
        environment to evaluate the policy
    algo: Expert
        well-trained policy to be evaluated
    episodes: int
        number of episodes used in evaluation
    render: bool
        render the environment or not
    seed: int
        random seed
    delay: float
        number of seconds to delay while rendering,
          in case the agent moves too fast

    Returns
    -------
    mean_return: float
        average episode reward
    """
  env.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)

  total_return = 0.0
  num_episodes = 0
  num_steps = []

  state = env.reset()
  t = 0
  latent = algo.get_latent(0, state)
  episode_return = 0.0
  episode_steps = 0

  while num_episodes < episodes:
    t += 1

    action = algo.exploit(state, latent)
    next_state, reward, done, _ = env.step(action)
    if algo.cb_reward:
      reward = algo.cb_reward(state, latent, action, reward)
    print(latent, reward)
    next_latent = algo.get_latent(t, next_state, latent, action, state)
    episode_return += reward
    episode_steps += 1
    state = next_state
    latent = next_latent

    if render:
      env.render()
      time.sleep(delay)

    if done or t == env.max_episode_steps:
      num_episodes += 1
      total_return += episode_return
      state = env.reset()
      t = 0
      latent = algo.get_latent(0, state)
      episode_return = 0.0
      num_steps.append(episode_steps)
      episode_steps = 0

  mean_return = total_return / num_episodes
  print(f'Mean return of the policy is {mean_return}')
  print(f'Max episode steps is {np.max(num_steps)}')
  print(f'Min episode steps is {np.min(num_steps)}')

  return mean_return
