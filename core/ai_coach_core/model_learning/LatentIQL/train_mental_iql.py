from typing import Optional
import os
import random
import numpy as np
import time
import datetime
import torch
from itertools import count
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from ai_coach_core.model_learning.IQLearn.utils.utils import make_env, eval_mode
from ai_coach_core.model_learning.IQLearn.dataset.expert_dataset import (
    ExpertDataset)
from ai_coach_core.model_learning.IQLearn.utils.logger import Logger
from .agent import make_miql_agent
from .agent.mental_models import MentalDoubleQCritic
from .helper.mental_memory import MentalMemory
from .helper.utils import get_expert_batch, evaluate, save


def train_mental_iql(env_name,
                     env_kwargs,
                     seed,
                     batch_size,
                     num_latent,
                     demo_path,
                     num_trajs,
                     log_dir,
                     output_dir,
                     replay_mem,
                     eps_window,
                     num_learn_steps,
                     initial_mem=None,
                     output_suffix="",
                     log_interval=500,
                     eval_interval=2000,
                     gumbel_temperature: float = 1.0,
                     list_hidden_dims=[256, 256],
                     clip_grad_val=None,
                     learn_alpha=False,
                     learning_rate=0.005,
                     load_path: Optional[str] = None,
                     bounded_actor=True,
                     method_loss="value"):
  agent_name = "miql"
  # constants
  num_episodes = 10
  save_interval = 10
  is_sqil = False
  if initial_mem is None:
    initial_mem = batch_size

  # device
  device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
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

  replay_mem = int(replay_mem)
  initial_mem = int(initial_mem)
  eps_window = int(eps_window)
  num_learn_steps = int(num_learn_steps)

  use_target = True
  do_soft_update = True
  agent = make_miql_agent(env,
                          batch_size,
                          device_name,
                          num_latent,
                          MentalDoubleQCritic,
                          critic_tau=0.005,
                          gumbel_temperature=gumbel_temperature,
                          learn_temp=learn_alpha,
                          critic_lr=learning_rate,
                          actor_lr=learning_rate,
                          thinker_lr=learning_rate,
                          alpha_lr=learning_rate,
                          list_critic_hidden_dims=list_hidden_dims,
                          list_actor_hidden_dims=list_hidden_dims,
                          list_thinker_hidden_dims=list_hidden_dims,
                          clip_grad_val=clip_grad_val,
                          bounded_actor=bounded_actor)

  if load_path is not None:
    if os.path.isfile(load_path):
      print("=> loading pretrain '{}'".format(load_path))
      agent.load(load_path)
    else:
      print("[Attention]: Did not find checkpoint {}".format(load_path))

  # Load expert data
  expert_dataset = ExpertDataset(demo_path, num_trajs, 1, seed + 42)
  print(f'--> Expert memory size: {len(expert_dataset)}')

  online_memory_replay = MentalMemory(replay_mem, seed + 1)

  # Setup logging
  ts_str = datetime.datetime.fromtimestamp(
      time.time()).strftime("%Y-%m-%d_%H-%M-%S")
  log_dir = os.path.join(log_dir, env_name, agent_name, ts_str)
  writer = SummaryWriter(log_dir=log_dir)
  print(f'--> Saving logs at: {log_dir}')
  logger = Logger(log_dir,
                  log_frequency=log_interval,
                  writer=writer,
                  save_tb=True,
                  agent=agent_name)

  # track mean reward and scores
  rewards_window = deque(maxlen=eps_window)  # last N rewards
  best_eval_returns = -np.inf

  begin_learn = False
  episode_reward = 0
  learn_steps = 0
  NAN = float("nan")

  for epoch in count():
    state = env.reset()
    prev_lat = NAN
    prev_act = (NAN if agent.actor.is_discrete() else np.zeros(
        env.action_space.shape))
    episode_reward = 0
    done = False

    for episode_step in count():
      with eval_mode(agent):
        # if not begin_learn:
        #   action = env.action_space.sample()
        # else:
        latent, action = agent.choose_action(state,
                                             prev_lat,
                                             prev_act,
                                             sample=True)
      next_state, reward, done, info = env.step(action)
      episode_reward += reward

      if learn_steps % eval_interval == 0 and begin_learn:
        eval_returns, eval_timesteps = evaluate(agent,
                                                eval_env,
                                                num_episodes=num_episodes)
        returns = np.mean(eval_returns)
        # learn_steps += 1  # To prevent repeated eval at timestep 0
        logger.log('eval/episode', epoch, learn_steps)
        logger.log('eval/episode_reward', returns, learn_steps)
        logger.dump(learn_steps, ty='eval')

        if returns > best_eval_returns:
          # Store best eval returns
          best_eval_returns = returns
          save(agent,
               epoch,
               1,
               env_name,
               agent_name,
               is_sqil,
               output_dir=output_dir,
               suffix=output_suffix + "_best")

      # only store done true when episode finishes without hitting timelimit
      done_no_lim = done
      if info.get('TimeLimit.truncated', False):
        done_no_lim = 0
      online_memory_replay.add((state, prev_lat, prev_act, next_state, latent,
                                action, reward, done_no_lim))

      learn_steps += 1
      if online_memory_replay.size() >= initial_mem:
        # Start learning
        if begin_learn is False:
          print('Learn begins!')
          begin_learn = True

        if learn_steps == num_learn_steps:
          print('Finished!')
          return

        ######
        # infer mental states of expert data
        num_samples = 1
        expert_traj = expert_dataset.sample_episodes(num_samples)
        expert_batch = get_expert_batch(agent, expert_traj, num_latent,
                                        agent.device)

        ######
        # IQ-Learn Modification
        losses = agent.iq_update(online_memory_replay, expert_batch, logger,
                                 learn_steps, is_sqil, use_target,
                                 do_soft_update, method_loss)
        ######

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
    logger.dump(learn_steps, save=begin_learn)
    save(agent,
         epoch,
         save_interval,
         env_name,
         agent_name,
         is_sqil,
         output_dir=output_dir,
         suffix=output_suffix)
