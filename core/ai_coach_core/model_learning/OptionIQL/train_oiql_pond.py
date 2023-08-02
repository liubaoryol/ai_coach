import os
import random
import numpy as np
import torch
from itertools import count
from torch.utils.tensorboard import SummaryWriter
from ai_coach_core.model_learning.IQLearn.utils.utils import make_env, eval_mode
from ai_coach_core.model_learning.IQLearn.dataset.expert_dataset import (
    ExpertDataset)
from ai_coach_core.model_learning.IQLearn.utils.logger import Logger
from .agent.make_agent import make_oiql_agent, make_osac_agent
from .agent.option_iql import OptionIQL
from .agent.option_sac import OptionSAC
from .helper.option_memory import OptionMemory
from .helper.utils import get_expert_batch, evaluate, save, get_samples
from aicoach_baselines.option_gail.utils.config import Config


def train_osac_pond(config: Config,
                    log_dir,
                    output_dir,
                    log_interval=500,
                    eval_interval=5000,
                    env_kwargs={}):
  return trainer_impl(config, None, None, log_dir, output_dir, "osac",
                      log_interval, eval_interval, env_kwargs)


def train_oiql_pond(config: Config,
                    demo_path,
                    num_trajs,
                    log_dir,
                    output_dir,
                    log_interval=500,
                    eval_interval=5000,
                    env_kwargs={}):
  return trainer_impl(config, demo_path, num_trajs, log_dir, output_dir, "oiql",
                      log_interval, eval_interval, env_kwargs)


def step_iq_update(config: Config, agent: OptionIQL, sample_data, expert_data,
                   logger, explore_steps, is_sqil):
  use_target = True
  do_soft_update = True

  # #################### update
  agent.reset_optimizers(config)
  for _ in range(config.n_update_rounds):
    inds = torch.randperm(len(sample_data[0]), device=agent.device)
    for ind_p in inds.split(config.mini_batch_size):
      so, spl, spa, sno, sl, sa, sr, sd = (sample_data[0][ind_p],
                                           sample_data[1][ind_p],
                                           sample_data[2][ind_p],
                                           sample_data[3][ind_p],
                                           sample_data[4][ind_p],
                                           sample_data[5][ind_p],
                                           sample_data[6][ind_p],
                                           sample_data[7][ind_p])
      sample_batch = (so, spl, spa, sno, sl, sa, sr, sd)
      expert_batch = get_samples(ind_p.size(0), expert_data)

      # IQ-Learn
      losses = agent.iq_update(sample_batch, expert_batch, logger,
                               explore_steps, is_sqil, use_target,
                               do_soft_update, config.method_loss,
                               config.method_regularize)
  return losses


def step_sac_update(config: Config, agent: OptionSAC,
                    online_memory_replay: OptionMemory, logger, explore_steps):
  # #################### update
  # agent.reset_optimizers(config)
  for dummy_i in range(config.n_update_rounds):
    for dummy_j in range(online_memory_replay.size() // config.mini_batch_size):
      losses = agent.update(online_memory_replay, logger, explore_steps)

  return losses


def trainer_impl(config: Config,
                 demo_path,
                 num_trajs,
                 log_dir,
                 output_dir,
                 agent_name,
                 log_interval=500,
                 eval_interval=5000,
                 env_kwargs={}):
  env_name = config.env_name
  seed = config.seed
  num_latent = config.dim_c
  replay_mem = config.n_sample
  max_explore_step = config.max_explore_step
  initial_mem = replay_mem
  output_suffix = ""
  load_path = None
  is_sqil = False

  alg_type = 'rl'
  imitation = (agent_name == "oiql")
  if imitation:
    fn_make_agent = make_oiql_agent
    alg_type = 'sqil' if is_sqil else 'iq'
  elif agent_name == "osac":
    fn_make_agent = make_osac_agent
  else:
    raise NotImplementedError

  # constants
  num_episodes = 10
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

  agent = fn_make_agent(config, env)

  if load_path is not None:
    if os.path.isfile(load_path):
      print("=> loading pretrain '{}'".format(load_path))
      agent.load(load_path)
    else:
      print("[Attention]: Did not find checkpoint {}".format(load_path))

  if imitation:
    # Load expert data
    expert_dataset = ExpertDataset(demo_path, num_trajs, 1, seed + 42)
    print(f'--> Expert memory size: {len(expert_dataset)}')

  online_memory_replay = OptionMemory(replay_mem, seed + 1, use_deque=False)

  # Setup logging
  log_dir = os.path.join(log_dir, agent_name)
  writer = SummaryWriter(log_dir=log_dir)
  print(f'--> Saving logs at: {log_dir}')
  logger = Logger(log_dir,
                  log_frequency=log_interval,
                  writer=writer,
                  save_tb=True,
                  agent=agent_name)

  # track mean reward and scores
  best_eval_returns = -np.inf

  explore_steps = 0
  cnt_evals = 0
  n_total_epi = 0

  for epoch in count():
    if explore_steps >= max_explore_step:
      print('Finished!')
      return

    # #################### collect data
    explore_step_cur = 0
    avg_episode_reward = 0
    online_memory_replay.clear()
    for n_epi in count():
      state = env.reset()
      prev_lat, prev_act = agent.prev_latent, agent.prev_action
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
        if done:
          break

        state = next_state

      avg_episode_reward += episode_reward
      if online_memory_replay.size() >= replay_mem:
        break

    avg_episode_reward = avg_episode_reward / (n_epi + 1)
    avg_epi_step = explore_step_cur / (n_epi + 1)
    n_total_epi += n_epi + 1

    logger.log('train/episode', n_total_epi, explore_steps)
    logger.log('train/episode_reward', avg_episode_reward, explore_steps)
    logger.log('train/episode_step', avg_epi_step, explore_steps)
    logger.dump(explore_steps, save=(explore_steps > 0))
    explore_steps += explore_step_cur

    # #################### update
    if imitation:
      # prepare expert data
      expert_data = get_expert_batch(agent, expert_dataset.trajectories,
                                     num_latent, agent.device)
      sample_data = online_memory_replay.get_all_samples(agent.device)
      losses = step_iq_update(config, agent, sample_data, expert_data, logger,
                              explore_steps, is_sqil)
    else:
      losses = step_sac_update(config, agent, online_memory_replay, logger,
                               explore_steps)

    for key, loss in losses.items():
      writer.add_scalar("loss/" + key, loss, global_step=explore_steps)

    cnt_evals += explore_step_cur
    if cnt_evals >= eval_interval:
      cnt_evals = 0
      eval_returns, eval_timesteps = evaluate(agent,
                                              eval_env,
                                              num_episodes=num_episodes)
      returns = np.mean(eval_returns)
      logger.log('eval/episode_step', np.mean(eval_timesteps), explore_steps)
      logger.log('eval/episode_reward', returns, explore_steps)
      logger.dump(explore_steps, ty='eval')

      if returns > best_eval_returns:
        # Store best eval returns
        best_eval_returns = returns
        save(agent,
             epoch,
             1,
             env_name,
             agent_name,
             alg_type,
             output_dir=output_dir,
             suffix=output_suffix + "_best")
