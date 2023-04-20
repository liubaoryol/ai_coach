from typing import Optional
import os
import random
import numpy as np
import time
import datetime
import torch
from collections import deque
from ai_coach_core.model_learning.IQLearn.utils.utils import make_env, eval_mode

from ai_coach_core.model_learning.IQLearn.agent import (make_softq_agent,
                                                        make_sac_agent,
                                                        make_sacd_agent)
from ai_coach_core.model_learning.IQLearn.agent.softq_models import (
    SimpleQNetwork, SingleQCriticDiscrete)
from ai_coach_core.model_learning.IQLearn.agent.sac_models import (
    DoubleQCritic, SingleQCritic)
from ai_coach_core.model_learning.IQLearn.dataset.memory import Memory
from torch.utils.tensorboard import SummaryWriter
from ai_coach_core.model_learning.IQLearn.utils.logger import Logger
from itertools import count
import types
import ai_coach_core.gym


def save(agent,
         epoch,
         save_interval,
         env_name,
         agent_name,
         is_sqil: bool,
         output_dir='results',
         suffix=""):
  if epoch % save_interval == 0:
    if is_sqil:
      name = f'sqil_{env_name}'
    else:
      name = f'iq_{env_name}'

    if not os.path.exists(output_dir):
      os.mkdir(output_dir)
    file_path = os.path.join(output_dir, f'{agent_name}_{name}' + suffix)
    agent.save(file_path)


if __name__ == "__main__":
  env_name = 'LunarLander-v2'
  env_kwargs = {}
  seed = 0
  batch_size = 256
  LOG_DIR = os.path.join(os.path.dirname(__file__), "logs/")
  if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
  output_dir = os.path.join(os.path.dirname(__file__), "output/")
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  replay_mem = 100000
  initial_mem = batch_size
  eps_steps = 1000
  eps_window = 10
  num_iterations = 500000
  agent_name = "sac"
  log_interval = 500
  eval_interval = 1000

  num_eval_episode = 10
  save_interval = 10

  gumbel_temperature = 1.0
  list_hidden_dims = [64, 64, 32]
  clip_grad_val = 0.5
  start_with_random_sample = False

  # device
  device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
  print(device_name)
  cuda_deterministic = False

  # set seeds
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  device = torch.device(device_name)
  if device.type == 'cuda' and torch.cuda.is_available() and cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  env = make_env(env_name, env_make_kwargs=env_kwargs)
  eval_env = make_env(env_name, env_make_kwargs=env_kwargs)

  # Seed envs
  env.seed(seed)
  eval_env.seed(seed + 10)

  REPLAY_MEMORY = int(replay_mem)
  INITIAL_MEMORY = int(initial_mem)
  EPISODE_STEPS = int(eps_steps)
  EPISODE_WINDOW = int(eps_window)
  LEARN_STEPS = int(num_iterations)

  if agent_name == "softq":
    q_net_base = SimpleQNetwork
    use_target = False
    do_soft_update = False
    agent = make_softq_agent(env,
                             batch_size,
                             device_name,
                             q_net_base,
                             critic_target_update_frequency=4,
                             critic_tau=0.1,
                             list_hidden_dims=list_hidden_dims)
  elif agent_name == "sac":
    critic_base = DoubleQCritic
    use_target = True
    do_soft_update = True
    agent = make_sac_agent(env,
                           batch_size,
                           device_name,
                           critic_base,
                           critic_target_update_frequency=1,
                           critic_tau=0.005,
                           gumbel_temperature=gumbel_temperature,
                           learn_temp=False,
                           critic_lr=0.005,
                           actor_lr=0.005,
                           alpha_lr=0.005,
                           list_critic_hidden_dims=list_hidden_dims,
                           list_actor_hidden_dims=list_hidden_dims,
                           clip_grad_val=clip_grad_val)
  elif agent_name == "sacd":
    critic_base = SingleQCriticDiscrete
    use_target = True
    do_soft_update = True
    agent = make_sacd_agent(env,
                            batch_size,
                            device_name,
                            critic_base,
                            critic_target_update_frequency=1,
                            critic_tau=0.005,
                            list_critic_hidden_dims=list_hidden_dims,
                            list_actor_hidden_dims=list_hidden_dims)

  online_memory_replay = Memory(REPLAY_MEMORY, seed + 1)

  # Setup logging
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

  # track mean reward and scores
  scores_window = deque(maxlen=EPISODE_WINDOW)  # last N scores
  rewards_window = deque(maxlen=EPISODE_WINDOW)  # last N rewards
  best_eval_returns = -np.inf

  begin_learn = False
  episode_reward = 0
  learn_steps = 0

  for epoch in count():
    state = env.reset()
    episode_reward = 0
    done = False

    start_time = time.time()
    for episode_step in range(EPISODE_STEPS):

      with eval_mode(agent):
        if not begin_learn and start_with_random_sample:
          action = env.action_space.sample()
        else:
          action = agent.choose_action(state, sample=True)
      next_state, reward, done, info = env.step(action)
      episode_reward += reward

      if learn_steps % eval_interval == 0 and begin_learn:
        eval_returns, eval_timesteps = evaluate(agent,
                                                eval_env,
                                                num_episodes=num_eval_episode)
        returns = np.mean(eval_returns)
        # learn_steps += 1  # To prevent repeated eval at timestep 0
        logger.log('eval/episode_reward', returns, learn_steps)
        logger.dump(learn_steps, ty='eval')
        # print('EVAL\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, returns))

        if returns > best_eval_returns:
          # Store best eval returns
          best_eval_returns = returns
          save(agent,
               epoch,
               1,
               env_name,
               agent_name,
               False,
               output_dir=output_dir,
               suffix="_best")

      # only store done true when episode finishes without hitting timelimit (allow infinite bootstrap)
      done_no_lim = done
      if info.get('TimeLimit.truncated', False):
        done_no_lim = 0
      online_memory_replay.add((state, next_state, action, reward, done_no_lim))

      learn_steps += 1
      if online_memory_replay.size() > INITIAL_MEMORY:
        # Start learning
        if begin_learn is False:
          print('Learn begins!')
          begin_learn = True

        if learn_steps == LEARN_STEPS:
          print('Finished!')
          exit()

        losses = agent.update(online_memory_replay, logger, learn_steps)

        if learn_steps % log_interval == 0:
          for key, loss in losses.items():
            writer.add_scalar(key, loss, global_step=learn_steps)

      if done:
        break
      state = next_state

    rewards_window.append(episode_reward)
    logger.log('train/episode', epoch, learn_steps)
    logger.log('train/episode_reward', episode_reward, learn_steps)
    logger.log('train/episode_step', episode_step, learn_steps)
    # logger.log('train/duration', time.time() - start_time, learn_steps)
    logger.dump(learn_steps, save=begin_learn)
    save(agent,
         epoch,
         save_interval,
         env_name,
         agent_name,
         False,
         output_dir=output_dir)
