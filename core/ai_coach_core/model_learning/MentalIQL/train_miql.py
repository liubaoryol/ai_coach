import os
import random
import numpy as np
import torch
from itertools import count
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from ai_coach_core.model_learning.IQLearn.utils.utils import make_env, eval_mode
from ai_coach_core.model_learning.IQLearn.dataset.expert_dataset import (
    ExpertDataset)
from ai_coach_core.model_learning.IQLearn.utils.logger import Logger
from ai_coach_core.model_learning.OptionIQL.helper.option_memory import (
    OptionMemory)
from ai_coach_core.model_learning.OptionIQL.helper.utils import (
    get_expert_batch, evaluate, save, get_samples)
from aicoach_baselines.option_gail.utils.config import Config
from .agent.make_agent import MentalIQL
from .agent.make_agent import make_miql_agent


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


def train(config: Config,
          demo_path,
          num_trajs,
          log_dir,
          output_dir,
          log_interval=500,
          eval_interval=5000,
          env_kwargs={}):

  agent_name = "miql"
  env_name = config.env_name
  seed = config.seed
  batch_size = config.mini_batch_size
  replay_mem = config.n_sample
  max_explore_step = config.max_explore_step
  initial_mem = replay_mem
  output_suffix = ""
  eps_window = 10
  is_sqil = False
  num_episodes = 10

  fn_make_agent = make_miql_agent
  alg_type = 'sqil' if is_sqil else 'iq'
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
  max_explore_step = int(max_explore_step)

  agent = fn_make_agent(config, env)

  # Load expert data
  expert_dataset = ExpertDataset(demo_path, num_trajs, 1, seed + 42)
  print(f'--> Expert memory size: {len(expert_dataset)}')

  online_memory_replay = OptionMemory(replay_mem, seed + 1)

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
  rewards_window = deque(maxlen=eps_window)  # last N rewards
  epi_step_window = deque(maxlen=eps_window)
  cnt_steps = 0

  begin_learn = False
  episode_reward = 0
  explore_steps = 0
  expert_data = None

  for epoch in count():
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

      if explore_steps % eval_interval == 0 and begin_learn:
        eval_returns, eval_timesteps = evaluate(agent,
                                                eval_env,
                                                num_episodes=num_episodes)
        returns = np.mean(eval_returns)
        logger.log('eval/episode_reward', returns, explore_steps)
        logger.log('eval/episode_step', np.mean(eval_timesteps), explore_steps)
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

      # only store done true when episode finishes without hitting timelimit
      done_no_lim = done
      if info.get('TimeLimit.truncated', False):
        done_no_lim = 0
      online_memory_replay.add((prev_lat, prev_act, state, latent, action,
                                next_state, next_latent, reward, done_no_lim))

      explore_steps += 1
      if online_memory_replay.size() >= initial_mem:
        # Start learning
        if begin_learn is False:
          print('Learn begins!')
          begin_learn = True

        if explore_steps == max_explore_step:
          print('Finished!')
          return

        # ##### sample batch
        # infer mental states of expert data
        if (expert_data is None
            or explore_steps % config.demo_latent_infer_interval == 0):
          mental_states = infer_mental_states_all_demo(
              agent, expert_dataset.trajectories)
          mental_states_after_end = infer_last_next_mental_state(
              agent, expert_dataset.trajectories, mental_states)
          exb = get_expert_batch(
              expert_dataset.trajectories,
              mental_states,
              agent.device,
              agent.PREV_LATENT,
              agent.PREV_ACTION,
              mental_states_after_end=mental_states_after_end)
          expert_data = (exb["prev_latents"], exb["prev_actions"],
                         exb["states"], exb["latents"], exb["actions"],
                         exb["next_states"], exb["next_latents"],
                         exb["rewards"], exb["dones"])

        ######
        # IQ-Learn Modification
        expert_batch = get_samples(batch_size, expert_data)
        policy_batch = online_memory_replay.get_samples(batch_size,
                                                        agent.device)

        tx_losses, pi_losses = agent.miql_update(
            policy_batch, expert_batch, config.demo_latent_infer_interval,
            logger, explore_steps)

        if explore_steps % log_interval == 0:
          for key, loss in tx_losses.items():
            writer.add_scalar("tx_loss/" + key, loss, global_step=explore_steps)
          for key, loss in pi_losses.items():
            writer.add_scalar("pi_loss/" + key, loss, global_step=explore_steps)

      if done:
        break
      state = next_state
      prev_lat = latent
      prev_act = action
      latent = next_latent

    rewards_window.append(episode_reward)
    epi_step_window.append(episode_step + 1)
    cnt_steps += episode_step + 1
    if cnt_steps >= log_interval:
      cnt_steps = 0
      logger.log('train/episode', epoch, explore_steps)
      logger.log('train/episode_reward', np.mean(rewards_window), explore_steps)
      logger.log('train/episode_step', np.mean(epi_step_window), explore_steps)
      logger.dump(explore_steps, save=begin_learn)
