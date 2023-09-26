import os
import random
import numpy as np
import torch
from itertools import count
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from aic_ml.baselines.IQLearn.utils.utils import make_env, eval_mode
from aic_ml.baselines.IQLearn.dataset.expert_dataset import (ExpertDataset)
from aic_ml.baselines.IQLearn.utils.logger import Logger
from aic_ml.OptionIQL.helper.option_memory import (OptionMemory)
from aic_ml.OptionIQL.helper.utils import (get_expert_batch, evaluate, save,
                                           get_samples)
from .agent.make_agent import MentalIQL
from .agent.make_agent import make_miql_agent
from omegaconf import DictConfig


def collect_data(online_memory_replay, env, agent, num_data):
  'return: explore_step_cur, n_epi, avg_epi_step, avg_episode_reward'
  explore_step_cur = 0
  avg_episode_reward = 0
  online_memory_replay.clear()
  for n_epi in count():
    episode_reward = 0
    done = False

    state = env.reset()
    prev_lat, prev_act = agent.PREV_LATENT, agent.PREV_ACTION
    latent = agent.choose_mental_state(state, prev_lat, sample=True)

    for episode_step in count():
      with eval_mode(agent):
        # if not begin_learn:
        #   action = env.action_space.sample()
        # else:
        action = agent.choose_policy_action(state, latent, sample=True)

        next_state, reward, done, info = env.step(action)
        next_latent = agent.choose_mental_state(next_state, latent, sample=True)

      episode_reward += reward

      # only store done true when episode finishes without hitting timelimit
      done_no_lim = done
      if info.get('TimeLimit.truncated', False):
        done_no_lim = 0
      online_memory_replay.add((prev_lat, prev_act, state, latent, action,
                                next_state, next_latent, reward, done_no_lim))
      explore_step_cur += 1
      if done:
        break
      state = next_state
      prev_lat = latent
      prev_act = action
      latent = next_latent

    avg_episode_reward += episode_reward
    if online_memory_replay.size() >= num_data:
      break

  avg_episode_reward = avg_episode_reward / (n_epi + 1)
  avg_epi_step = explore_step_cur / (n_epi + 1)

  return explore_step_cur, n_epi, avg_epi_step, avg_episode_reward


def infer_mental_states_all_demo(agent: MentalIQL, expert_traj):
  num_samples = len(expert_traj["states"])
  list_mental_states = []
  for i_e in range(num_samples):
    expert_states = expert_traj["states"][i_e]
    expert_actions = expert_traj["actions"][i_e]
    mental_array, _ = agent.infer_mental_states(expert_states, expert_actions)
    list_mental_states.append(mental_array)

  return list_mental_states


def infer_last_next_mental_state(agent: MentalIQL, expert_traj,
                                 list_mental_states):
  num_samples = len(expert_traj["states"])
  list_last_next_mental_state = []
  for i_e in range(num_samples):
    last_next_state = expert_traj["next_states"][i_e][-1]
    last_mental_state = list_mental_states[i_e][-1]
    last_next_mental_state = agent.choose_mental_state(last_next_state,
                                                       last_mental_state, False)
    list_last_next_mental_state.append(last_next_mental_state)

  return list_last_next_mental_state


def train(config: DictConfig,
          demo_path,
          num_trajs,
          log_dir,
          output_dir,
          log_interval=500,
          eval_interval=5000,
          env_kwargs={}):

  env_name = config.env_name
  seed = config.seed
  batch_size = config.mini_batch_size

  n_pi_samples = int(config.miql_order_update_pi_ratio * config.n_sample)
  n_tx_samples = int(config.n_sample - n_pi_samples)

  max_explore_step = int(config.max_explore_step)
  output_suffix = ""
  num_episodes = 10

  fn_make_agent = make_miql_agent
  alg_type = 'iq'

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

  agent = fn_make_agent(config, env)

  if config.miql_tx_after_pi:
    fn_update_1, fn_update_2 = agent.tx_update, agent.pi_update
    n_1st_samples, n_2nd_samples = n_tx_samples, n_pi_samples
    str_1st_loss, str_2nd_loss = "tx_loss/", "pi_loss/"
  else:
    fn_update_1, fn_update_2 = agent.pi_update, agent.tx_update
    n_1st_samples, n_2nd_samples = n_pi_samples, n_tx_samples
    str_1st_loss, str_2nd_loss = "pi_loss/", "tx_loss/"

  n_1st_updates = int(n_1st_samples / batch_size * config.n_update_rounds)
  n_2nd_updates = int(n_2nd_samples / batch_size * config.n_update_rounds)
  replay_mem = max(n_pi_samples, n_tx_samples)

  # Load expert data
  expert_dataset = ExpertDataset(demo_path, num_trajs, 1, seed + 42)
  print(f'--> Expert memory size: {len(expert_dataset)}')

  online_memory_replay = OptionMemory(replay_mem, seed + 1)

  # Setup logging
  writer = SummaryWriter(log_dir=log_dir)
  print(f'--> Saving logs at: {log_dir}')
  logger = Logger(log_dir,
                  log_frequency=log_interval,
                  writer=writer,
                  save_tb=True)

  # track mean reward and scores
  best_eval_returns = -np.inf

  episode_reward = 0
  explore_steps = 0
  expert_data = None
  cnt_evals = 0
  n_total_epi = 0

  for epoch in count():
    if explore_steps >= max_explore_step:
      print('Finished!')
      return

    # #################### infer mental state and prepare data
    mental_states = infer_mental_states_all_demo(agent,
                                                 expert_dataset.trajectories)
    mental_states_after_end = infer_last_next_mental_state(
        agent, expert_dataset.trajectories, mental_states)
    exb = get_expert_batch(expert_dataset.trajectories,
                           mental_states,
                           agent.device,
                           agent.PREV_LATENT,
                           agent.PREV_ACTION,
                           mental_states_after_end=mental_states_after_end)
    expert_data = (exb["prev_latents"], exb["prev_actions"], exb["states"],
                   exb["latents"], exb["actions"], exb["next_states"],
                   exb["next_latents"], exb["rewards"], exb["dones"])

    # #################### update 1
    # collect data
    explore_step_cur_1, n_epi, avg_epi_step, avg_episode_reward = collect_data(
        online_memory_replay, env, agent, n_1st_samples)

    n_total_epi += n_epi + 1
    explore_steps += explore_step_cur_1

    logger.log('train/episode', n_total_epi, explore_steps)
    logger.log('train/episode_reward', avg_episode_reward, explore_steps)
    logger.log('train/episode_step', avg_epi_step, explore_steps)

    online_data = online_memory_replay.get_all_samples(agent.device)

    for _ in range(n_1st_updates):
      expert_batch = get_samples(batch_size, expert_data)
      policy_batch = get_samples(batch_size, online_data)

      losses = fn_update_1(policy_batch, expert_batch, logger, explore_steps)

    for key, loss in losses.items():
      writer.add_scalar(str_1st_loss + key, loss, global_step=explore_steps)

    # #################### update 2
    # collect data
    explore_step_cur_2, n_epi, avg_epi_step, avg_episode_reward = collect_data(
        online_memory_replay, env, agent, n_2nd_samples)

    n_total_epi += n_epi + 1
    explore_steps += explore_step_cur_2

    logger.log('train/episode', n_total_epi, explore_steps)
    logger.log('train/episode_reward', avg_episode_reward, explore_steps)
    logger.log('train/episode_step', avg_epi_step, explore_steps)
    logger.dump(explore_steps, save=(explore_steps > 0))

    online_data = online_memory_replay.get_all_samples(agent.device)

    for _ in range(n_2nd_updates):
      expert_batch = get_samples(batch_size, expert_data)
      policy_batch = get_samples(batch_size, online_data)

      losses = fn_update_2(policy_batch, expert_batch, logger, explore_steps)

    for key, loss in losses.items():
      writer.add_scalar(str_2nd_loss + key, loss, global_step=explore_steps)

    # #################### eval
    cnt_evals += (explore_step_cur_1 + explore_step_cur_2)
    if cnt_evals >= eval_interval:
      cnt_evals = 0
      eval_returns, eval_timesteps, successes = evaluate(
          agent, eval_env, num_episodes=num_episodes)
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
             alg_type,
             output_dir=output_dir,
             suffix=output_suffix + "_best")
