import os
import torch
import gym
from gym import spaces
from ai_coach_core.slm_lab_test.agent.net.mlp import MLPNet
from ai_coach_core.slm_lab_test.agent.net.q_net import QMLPNet
from ai_coach_core.slm_lab_test.agent.algorithm.sac import SoftActorCritic
from ai_coach_core.slm_lab_test.agent.memory.replay import Replay
from ai_coach_core.slm_lab_test.agent.algorithm import policy_util
from ai_coach_core.model_learning.IQLearn.utils.utils import make_env
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from ai_coach_core.model_learning.IQLearn.utils.logger import Logger
import datetime
import time
from itertools import count
from collections import deque
from tqdm import tqdm


def get_observable_dim(observation_space):
  '''Get the observable dim for an agent in env'''
  state_dim = observation_space.shape
  if len(state_dim) == 1:
    state_dim = state_dim[0]
  return state_dim


def get_action_dim(action_space):
  '''Get the action dim for an action_space for agent to use'''
  if isinstance(action_space, spaces.Box):
    assert len(action_space.shape) == 1
    action_dim = action_space.shape[0]
  elif isinstance(action_space, (spaces.Discrete, spaces.MultiBinary)):
    action_dim = action_space.n
  elif isinstance(action_space, spaces.MultiDiscrete):
    action_dim = action_space.nvec.tolist()
  else:
    raise ValueError('action_space not recognized')
  return action_dim


def evaluate(alg: SoftActorCritic, env: gym.Env, num_episodes=10):
  """Evaluates the policy.
    Args:
      actor: A policy to evaluate.
      env: Environment to evaluate the policy on.
      num_episodes: A number of episodes to average the policy on.
    Returns:
      Averaged reward and a total number of steps.
    """
  total_timesteps = []
  total_returns = []

  while len(total_returns) < num_episodes:
    state = env.reset()
    done = False

    while not done:
      with torch.no_grad():
        action = alg.act(state)
      next_state, reward, done, info = env.step(action)
      state = next_state

      if 'episode' in info.keys():
        total_returns.append(info['episode']['r'])
        total_timesteps.append(info['episode']['l'])

  return total_returns, total_timesteps


def run_rl(env: gym.Env, memory: Replay, max_steps, algorithm: SoftActorCritic):
  '''Run the main RL loop until clock.max_frame'''
  state = env.reset()
  done = False
  steps = 0
  num_epi = 0
  while True:
    if done:  # before starting another episode
      if steps < max_steps:  # reset and continue
        num_epi += 1
        state = env.reset()
        done = False
    # self.try_ckpt(self.agent, self.env)
    if steps >= max_steps:  # finish
      break

    steps += 1
    with torch.no_grad():
      action = algorithm.act(state, steps)
    next_state, reward, done, info = env.step(action)
    # self.agent.update(state, action, reward, next_state, done)
    # log
    memory.update(state, action, reward, next_state, done)
    algorithm.to_train = algorithm.to_train or (
        memory.seen_size > algorithm.training_start_step
        and memory.head % algorithm.training_frequency == 0)

    losses = algorithm.train()
    algorithm.update()
    state = next_state


if __name__ == "__main__":
  env_name = 'LunarLander-v2'
  env_kwargs = {}

  batch_size = 256
  max_size = 100000
  use_cer = False
  max_steps = 300000

  action_pdtype = "GumbelSoftmax"
  gamma = 0.99
  training_frequency = 1
  agent_name = 'sac'
  log_interval = 500
  eval_interval = 1000

  num_eval_episode = 10
  eps_steps = 1000
  eps_window = 10

  # device
  device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
  cuda_deterministic = False

  # set seeds
  seed = 0
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  device = torch.device(device_name)
  if device.type == 'cuda' and torch.cuda.is_available() and cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  env = make_env(env_name, env_kwargs)
  eval_env = make_env(env_name, env_make_kwargs=env_kwargs)

  env.seed(seed)
  eval_env.seed(seed + 10)

  net_kwargs = {
      "net_class": MLPNet,
      "qnet_class": QMLPNet,
      "in_dim": get_observable_dim(env.observation_space),
      "out_dim": get_action_dim(env.action_space),
      "device": device_name,
      "hid_layers": [64, 64, 32],
      "hid_layers_activation": 'relu',
      "out_layer_activation": None,
      "init_fn": 'orthogonal_',
      "clip_grad_val": 0.5,
      "loss_spec": {
          "name": "MSELoss"
      },
      "optim_spec": {
          'name': 'Adam',
          'lr': 0.005
      },
      "lr_scheduler_spec": {},
      "update_type": 'polyak',
      "update_frequency": 1,
      "polyak_coef": 0.005
  }

  # Setup logging
  LOG_DIR = os.path.join(os.path.dirname(__file__), "logs/")
  if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

  ts_str = datetime.datetime.fromtimestamp(
      time.time()).strftime("%Y-%m-%d_%H-%M-%S")
  log_dir = os.path.join(LOG_DIR, env_name, agent_name, ts_str)
  writer = SummaryWriter(log_dir=log_dir)
  print(f'--> Saving logs at: {log_dir}')
  logger = Logger(log_dir,
                  log_frequency=log_interval,
                  writer=writer,
                  save_tb=True,
                  agent=agent_name)

  memory = Replay(batch_size, max_size, use_cer)
  alg = SoftActorCritic(env, memory, net_kwargs, action_pdtype, batch_size,
                        gamma, training_frequency)

  episode_reward = 0
  learn_steps = 0
  rewards_window = deque(maxlen=eps_window)
  best_eval_returns = -np.inf
  # run_rl(env, memory, max_steps, alg)
  started = False

  progress_bar = tqdm(total=max_steps)
  for epoch in count():
    state = env.reset()
    episode_reward = 0
    done = False

    start_time = time.time()
    for episode_step in range(eps_steps):

      with torch.no_grad():
        if learn_steps < alg.training_start_step:
          action = policy_util.random(state, env).cpu().squeeze().numpy()
        else:
          action = alg.act(state)
      next_state, reward, done, info = env.step(action)
      episode_reward += reward

      if learn_steps % eval_interval == 0 and started:
        eval_returns, eval_timesteps = evaluate(alg,
                                                eval_env,
                                                num_episodes=num_eval_episode)
        returns = np.mean(eval_returns)
        # learn_steps += 1  # To prevent repeated eval at timestep 0
        logger.log('eval/episode_reward', returns, learn_steps)
        # logger.log('eval/episode', epoch, learn_steps)
        logger.dump(learn_steps, ty='eval', save=False)
        # print('EVAL\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, returns))

        if returns > best_eval_returns:
          # Store best eval returns
          best_eval_returns = returns

      # only store done true when episode finishes without hitting timelimit (allow infinite bootstrap)
      done_no_lim = done
      if info.get('TimeLimit.truncated', False):
        done_no_lim = 0
      memory.update(state, action, reward, next_state, done)
      alg.to_train = alg.to_train or (memory.seen_size > alg.training_start_step
                                      and memory.head % alg.training_frequency
                                      == 0)
      started = started or alg.to_train

      losses = alg.train(learn_steps)
      alg.update()

      learn_steps += 1

      progress_bar.update()
      if learn_steps == max_steps:
        print('Finished!')
        exit()

      # losses = agent.update(online_memory_replay, logger, learn_steps)

      if learn_steps % log_interval == 0:
        writer.add_scalar('loss/loss_sum', losses[0], global_step=learn_steps)
        writer.add_scalar('loss/actor_loss', losses[1], global_step=learn_steps)
        writer.add_scalar('loss/critic_loss',
                          losses[2],
                          global_step=learn_steps)
        writer.add_scalar('loss/alpha_loss', losses[3], global_step=learn_steps)
      #   for key, loss in losses.items():
      #     writer.add_scalar(key, loss, global_step=learn_steps)

      if done:
        break
      state = next_state

    rewards_window.append(episode_reward)
    logger.log('train/episode', epoch, learn_steps)
    logger.log('train/episode_reward', episode_reward, learn_steps)
    logger.log('train/episode_step', episode_step, learn_steps)
    logger.dump(learn_steps, save=False)

  progress_bar.close()