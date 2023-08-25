import os
import random
import numpy as np
import torch
from itertools import count
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from aic_ml.IQLearn.utils.utils import make_env, eval_mode
from aic_ml.IQLearn.dataset.expert_dataset import (ExpertDataset)
from aic_ml.IQLearn.utils.logger import Logger
from .agent.make_agent import make_oiql_agent, make_osac_agent
from .helper.option_memory import OptionMemory
from .helper.utils import (get_expert_batch, evaluate, save, get_samples,
                           infer_mental_states)
from aic_baselines.option_gail.utils.config import Config
from aic_ml.MentalIQL.train_miql import (load_expert_data_w_labels)


def train_osac_stream(config: Config,
                      log_dir,
                      output_dir,
                      log_interval=500,
                      eval_interval=5000,
                      env_kwargs={}):
  return trainer_impl(config, None, None, log_dir, output_dir, "osac",
                      log_interval, eval_interval, env_kwargs)


def train_oiql_stream(config: Config,
                      demo_path,
                      num_trajs,
                      log_dir,
                      output_dir,
                      log_interval=500,
                      eval_interval=5000,
                      env_kwargs={}):
  return trainer_impl(config, demo_path, num_trajs, log_dir, output_dir, "oiql",
                      log_interval, eval_interval, env_kwargs)


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
  batch_size = config.mini_batch_size
  num_latent = config.dim_c
  replay_mem = config.n_sample
  num_learn_steps = config.max_explore_step
  output_suffix = ""
  load_path = None
  method_loss = config.method_loss
  method_regularize = config.method_regularize
  eps_window = 10

  is_sqil = False
  alg_type = "rl"
  imitation = (agent_name == "oiql")
  if imitation:
    fn_make_agent = make_oiql_agent
    alg_type = 'sqil' if is_sqil else 'iq'
  elif agent_name == "osac":
    fn_make_agent = make_osac_agent
  else:
    raise NotImplementedError

  # constants
  num_episodes = 8

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
  eps_window = int(eps_window)
  num_learn_steps = int(num_learn_steps)

  use_target = True
  do_soft_update = True
  agent = fn_make_agent(config, env)

  if load_path is not None:
    if os.path.isfile(load_path):
      print("=> loading pretrain '{}'".format(load_path))
      agent.load(load_path)
    else:
      print("[Attention]: Did not find checkpoint {}".format(load_path))

  if imitation:
    # # Load expert data
    # expert_dataset = ExpertDataset(demo_path, num_trajs, 1, seed + 42)
    # print(f'--> Expert memory size: {len(expert_dataset)}')
    n_labeled = int(num_trajs * config.supervision)
    expert_dataset, traj_labels, cnt_label = load_expert_data_w_labels(
        demo_path, num_trajs, n_labeled, seed)
    output_suffix = f"_n{num_trajs}_l{cnt_label}"
    batch_size = min(batch_size, len(expert_dataset))

  online_memory_replay = OptionMemory(replay_mem, seed + 1)

  initial_mem = min(batch_size * 5, replay_mem)
  initial_mem = int(initial_mem)
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
  learn_steps = 0
  expert_data = None

  for epoch in count():
    state = env.reset()
    prev_lat, prev_act = agent.PREV_LATENT, agent.PREV_ACTION
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
        # logger.log('eval/episode', epoch, learn_steps)
        logger.log('eval/episode_reward', returns, learn_steps)
        logger.log('eval/episode_step', np.mean(eval_timesteps), learn_steps)
        logger.dump(learn_steps, ty='eval')

        if returns >= best_eval_returns:
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

        if imitation:
          # ##### sample batch
          # infer mental states of expert data
          if (expert_data is None
              or learn_steps % config.demo_latent_infer_interval == 0):
            inferred_latents = infer_mental_states(agent,
                                                   expert_dataset.trajectories,
                                                   num_latent, traj_labels)
            exb = get_expert_batch(expert_dataset.trajectories,
                                   inferred_latents, agent.device,
                                   agent.PREV_LATENT, agent.PREV_ACTION)
            expert_data = (exb["states"], exb["prev_latents"],
                           exb["prev_actions"], exb["next_states"],
                           exb["latents"], exb["actions"], exb["rewards"],
                           exb["dones"])

          expert_batch = get_samples(batch_size, expert_data)
          policy_batch = online_memory_replay.get_samples(
              batch_size, agent.device)

          ######
          # IQ-Learn Modification
          losses = agent.iq_update(policy_batch, expert_batch, logger,
                                   learn_steps, is_sqil, use_target,
                                   do_soft_update, method_loss,
                                   method_regularize)
        else:
          losses = agent.update(online_memory_replay, logger, learn_steps)

        if learn_steps % log_interval == 0:
          for key, loss in losses.items():
            writer.add_scalar("loss/" + key, loss, global_step=learn_steps)

      if done:
        break
      state = next_state
      prev_lat = latent
      prev_act = action

    rewards_window.append(episode_reward)
    epi_step_window.append(episode_step + 1)
    cnt_steps += episode_step + 1
    if cnt_steps >= log_interval:
      cnt_steps = 0
      logger.log('train/episode', epoch, learn_steps)
      logger.log('train/episode_reward', np.mean(rewards_window), learn_steps)
      logger.log('train/episode_step', np.mean(epi_step_window), learn_steps)
      logger.dump(learn_steps, save=begin_learn)
