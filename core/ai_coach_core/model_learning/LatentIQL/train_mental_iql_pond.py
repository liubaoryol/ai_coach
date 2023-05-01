from typing import Optional
import os
import random
import numpy as np
import time
import datetime
import torch
from itertools import count
from torch.utils.tensorboard import SummaryWriter
from ai_coach_core.model_learning.IQLearn.utils.utils import make_env, eval_mode
from ai_coach_core.model_learning.IQLearn.dataset.expert_dataset import (
    ExpertDataset)
from ai_coach_core.model_learning.IQLearn.utils.logger import Logger
from .agent.make_agent import make_miql_agent
from .helper.mental_memory import MentalMemory
from .helper.utils import get_expert_batch, evaluate, save


def train_mental_iql_pond(env_name,
                          env_kwargs,
                          seed,
                          batch_size,
                          num_latent,
                          demo_path,
                          num_trajs,
                          log_dir,
                          output_dir,
                          replay_mem,
                          max_explore_step,
                          initial_mem=None,
                          output_suffix="",
                          log_interval=500,
                          eval_epoch_interval=5,
                          gumbel_temperature: float = 1.0,
                          list_critic_hidden_dims=[256, 256],
                          list_actor_hidden_dims=[256, 256],
                          list_thinker_hidden_dims=[256, 256],
                          num_critic_update=1,
                          num_actor_update=1,
                          clip_grad_val=None,
                          learn_alpha=False,
                          critic_lr=0.005,
                          actor_lr=0.005,
                          thinker_lr=0.005,
                          alpha_lr=0.005,
                          load_path: Optional[str] = None,
                          bounded_actor=True,
                          method_loss="value",
                          method_regularize=True,
                          use_prev_action=True):
  agent_name = "miql"
  # constants
  num_episodes = 10
  save_interval = 10
  is_sqil = False
  if initial_mem is None:
    initial_mem = replay_mem

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

  use_target = True
  do_soft_update = True
  agent = make_miql_agent(env,
                          batch_size,
                          device_name,
                          num_latent,
                          critic_tau=0.005,
                          gumbel_temperature=gumbel_temperature,
                          learn_temp=learn_alpha,
                          critic_lr=critic_lr,
                          actor_lr=actor_lr,
                          thinker_lr=thinker_lr,
                          alpha_lr=alpha_lr,
                          list_critic_hidden_dims=list_critic_hidden_dims,
                          list_actor_hidden_dims=list_actor_hidden_dims,
                          list_thinker_hidden_dims=list_thinker_hidden_dims,
                          num_critic_update=num_critic_update,
                          num_actor_update=num_actor_update,
                          clip_grad_val=clip_grad_val,
                          bounded_actor=bounded_actor,
                          use_prev_action=use_prev_action)

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
  best_eval_returns = -np.inf

  learn_steps = 0
  NAN = float("nan")
  N_UPDATE_STEPS = 10

  for epoch in count():
    if learn_steps >= max_explore_step:
      print('Finished!')
      return

    # #################### collect data
    explore_step_cur = 0
    avg_episode_reward = 0
    online_memory_replay.clear()
    for n_epi in count():
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

        # only store done true when episode finishes without hitting timelimit
        done_no_lim = done
        if info.get('TimeLimit.truncated', False):
          done_no_lim = 0
        online_memory_replay.add((state, prev_lat, prev_act, next_state, latent,
                                  action, reward, done_no_lim))
        explore_step_cur += 1
        if done or online_memory_replay.size() == replay_mem:
          break

        state = next_state

      if online_memory_replay.size() < replay_mem:
        avg_episode_reward += episode_reward
      else:
        avg_episode_reward = 0 if n_epi == 0 else avg_episode_reward / n_epi
        break
    logger.log('train/episode_reward', avg_episode_reward, learn_steps)
    logger.dump(learn_steps, save=(learn_steps > 0))
    learn_steps += explore_step_cur

    # #################### prepare expert data
    expert_traj = expert_dataset.trajectories
    expert_data = get_expert_batch(agent, expert_traj, num_latent, agent.device)
    sample_data = online_memory_replay.get_all_samples(agent.device)

    for _ in range(N_UPDATE_STEPS):
      inds = torch.randperm(online_memory_replay.size(), device=agent.device)
      for ind_p in inds.split(batch_size):
        so, spl, spa, sno, sl, sa, sr, sd = (sample_data[0][ind_p],
                                             sample_data[1][ind_p],
                                             sample_data[2][ind_p],
                                             sample_data[3][ind_p],
                                             sample_data[4][ind_p],
                                             sample_data[5][ind_p],
                                             sample_data[6][ind_p],
                                             sample_data[7][ind_p])
        sample_batch = (so, spl, spa, sno, sl, sa, sr, sd)
        ind_e = torch.randperm(len(expert_data[0]),
                               device=agent.device)[:ind_p.size(0)]
        eo, epl, epa, eno, el, ea, er, ed = (expert_data[0][ind_e],
                                             expert_data[1][ind_e],
                                             expert_data[2][ind_e],
                                             expert_data[3][ind_e],
                                             expert_data[4][ind_e],
                                             expert_data[5][ind_e],
                                             expert_data[6][ind_e],
                                             expert_data[7][ind_e])
        expert_batch = (eo, epl, epa, eno, el, ea, er, ed)

        # IQ-Learn
        losses = agent.iq_update(sample_batch, expert_batch, logger,
                                 learn_steps, is_sqil, use_target,
                                 do_soft_update, method_loss, method_regularize)

    for key, loss in losses.items():
      writer.add_scalar(key, loss, global_step=learn_steps)

    save(agent,
         epoch,
         save_interval,
         env_name,
         agent_name,
         is_sqil,
         output_dir=output_dir,
         suffix=output_suffix)

    if (epoch + 1) % eval_epoch_interval == 0:
      eval_returns, eval_timesteps = evaluate(agent,
                                              eval_env,
                                              num_episodes=num_episodes)
      returns = np.mean(eval_returns)
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
