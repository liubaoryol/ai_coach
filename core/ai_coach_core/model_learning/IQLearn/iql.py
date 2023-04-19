from typing import Optional
import os
import random
import numpy as np
import time
import datetime
import torch
from collections import deque
from .utils.utils import (make_env, eval_mode, average_dicts,
                          get_concat_samples, evaluate, soft_update,
                          hard_update)

from .agent import make_agent
from .agent.softq_models import SimpleQNetwork
from .dataset.memory import Memory
from torch.utils.tensorboard import SummaryWriter
from .utils.logger import Logger
from itertools import count
import types
from .iq import iq_loss


def run_iql(env_name,
            env_kwargs,
            seed,
            batch_size,
            demo_path,
            num_trajs,
            log_dir,
            output_dir,
            replay_mem,
            initial_mem,
            eps_steps,
            eps_window,
            num_learn_steps,
            output_suffix="",
            log_interval=1000,
            eval_interval=2000,
            load_path: Optional[str] = None):
  # constants
  num_seed_steps = 0
  num_episodes = 10
  save_interval = 10
  is_sqil = False
  only_expert_states = False
  use_target = False
  agent_name = "softq"  # either softq or sac

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

  REPLAY_MEMORY = int(replay_mem)
  INITIAL_MEMORY = int(initial_mem)
  EPISODE_STEPS = int(eps_steps)
  EPISODE_WINDOW = int(eps_window)
  LEARN_STEPS = int(num_learn_steps)

  q_net_base = SimpleQNetwork
  agent = make_agent(env, batch_size, device_name, q_net_base)

  if load_path is not None:
    if os.path.isfile(load_path):
      print("=> loading pretrain '{}'".format(load_path))
      agent.load(load_path)
    else:
      print("[Attention]: Did not find checkpoint {}".format(load_path))

  # Load expert data
  subsample_freq = 1
  expert_memory_replay = Memory(REPLAY_MEMORY // 2, seed)
  expert_memory_replay.load(demo_path,
                            num_trajs=num_trajs,
                            sample_freq=subsample_freq,
                            seed=seed + 42)
  print(f'--> Expert memory size: {expert_memory_replay.size()}')

  online_memory_replay = Memory(REPLAY_MEMORY // 2, seed + 1)

  # Setup logging
  ts_str = datetime.datetime.fromtimestamp(
      time.time()).strftime("%Y-%m-%d_%H-%M-%S")
  log_dir = os.path.join(log_dir, env_name, ts_str)
  writer = SummaryWriter(log_dir=log_dir)
  print(f'--> Saving logs at: {log_dir}')
  logger = Logger(log_dir,
                  log_frequency=log_interval,
                  writer=writer,
                  save_tb=True,
                  agent=agent_name)

  steps = 0

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

      # if steps < args.num_seed_steps:
      #   # Seed replay buffer with random actions
      #   action = env.action_space.sample()
      # else:
      with eval_mode(agent):
        action = agent.choose_action(state, sample=True)
      next_state, reward, done, _ = env.step(action)
      episode_reward += reward
      steps += 1

      if learn_steps % eval_interval == 0 and begin_learn:
        eval_returns, eval_timesteps = evaluate(agent,
                                                eval_env,
                                                num_episodes=num_episodes)
        returns = np.mean(eval_returns)
        # learn_steps += 1  # To prevent repeated eval at timestep 0
        logger.log('eval/episode_reward', returns, learn_steps)
        logger.log('eval/episode', epoch, learn_steps)
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
               is_sqil,
               output_dir=output_dir,
               suffix=output_suffix + "_best")

      # only store done true when episode finishes without hitting timelimit (allow infinite bootstrap)
      done_no_lim = done
      if str(env.__class__.__name__).find(
          'TimeLimit') >= 0 and episode_step + 1 == env._max_episode_steps:
        done_no_lim = 0
      online_memory_replay.add((state, next_state, action, reward, done_no_lim))

      if online_memory_replay.size() > INITIAL_MEMORY:
        # Start learning
        if begin_learn is False:
          print('Learn begins!')
          begin_learn = True

        learn_steps += 1
        if learn_steps == LEARN_STEPS:
          print('Finished!')
          # wandb.finish()
          return

        ######
        # IQ-Learn Modification
        agent.iq_update = types.MethodType(iq_update, agent)
        agent.iq_update_critic = types.MethodType(iq_update_critic, agent)
        losses = agent.iq_update(online_memory_replay, expert_memory_replay,
                                 logger, learn_steps, only_expert_states,
                                 is_sqil, use_target)
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
    logger.log('train/duration', time.time() - start_time, learn_steps)
    logger.dump(learn_steps, save=begin_learn)
    save(agent,
         epoch,
         save_interval,
         env_name,
         agent_name,
         is_sqil,
         output_dir=output_dir,
         suffix=output_suffix)


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


# Minimal IQ-Learn objective
def iq_learn_update(self,
                    policy_batch,
                    expert_batch,
                    logger,
                    step,
                    only_expert_states=False,
                    is_sqil=False,
                    use_target=False,
                    method_alpha=0.5):
  # args = self.args

  (policy_obs, policy_next_obs, policy_action, policy_reward,
   policy_done) = policy_batch
  (expert_obs, expert_next_obs, expert_action, expert_reward,
   expert_done) = expert_batch

  if only_expert_states:
    expert_batch = (expert_obs, expert_next_obs, policy_action, expert_reward,
                    expert_done)

  obs, next_obs, action, reward, done, is_expert = get_concat_samples(
      policy_batch, expert_batch, is_sqil)

  loss_dict = {}

  ######
  # IQ-Learn minimal implementation with X^2 divergence (~15 lines)
  # Calculate 1st term of loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
  current_Q = self.critic(obs, action)
  y = (1 - done) * self.gamma * self.getV(next_obs)
  if use_target:
    with torch.no_grad():
      y = (1 - done) * self.gamma * self.get_targetV(next_obs)

  reward = (current_Q - y)[is_expert]
  loss = -(reward).mean()

  # 2nd term for our loss (use expert and policy states): E_(ρ)[Q(s,a) - γV(s')]
  value_loss = (self.getV(obs) - y).mean()
  loss += value_loss

  # Use χ2 divergence (adds a extra term to the loss)
  chi2_loss = 1 / (4 * method_alpha) * (reward**2).mean()
  loss += chi2_loss
  ######

  self.critic_optimizer.zero_grad()
  loss.backward()
  self.critic_optimizer.step()
  return loss


def iq_update_critic(self,
                     policy_batch,
                     expert_batch,
                     logger,
                     step,
                     only_expert_states=False,
                     is_sqil=False,
                     use_target=False):
  (policy_obs, policy_next_obs, policy_action, policy_reward,
   policy_done) = policy_batch
  (expert_obs, expert_next_obs, expert_action, expert_reward,
   expert_done) = expert_batch

  if only_expert_states:
    # Use policy actions instead of experts actions for IL with only observations
    expert_batch = (expert_obs, expert_next_obs, policy_action, expert_reward,
                    expert_done)

  batch = get_concat_samples(policy_batch, expert_batch, is_sqil)
  obs, next_obs, action = batch[0:3]

  agent = self
  current_V = self.getV(obs)
  if use_target:
    with torch.no_grad():
      next_V = self.get_targetV(next_obs)
  else:
    next_V = self.getV(next_obs)

  if "DoubleQ" in self.__class__.__name__:
    current_Q1, current_Q2 = self.critic(obs, action, both=True)
    q1_loss, loss_dict1 = iq_loss(agent, current_Q1, current_V, next_V, batch)
    q2_loss, loss_dict2 = iq_loss(agent, current_Q2, current_V, next_V, batch)
    critic_loss = 1 / 2 * (q1_loss + q2_loss)
    # merge loss dicts
    loss_dict = average_dicts(loss_dict1, loss_dict2)
  else:
    current_Q = self.critic(obs, action)
    critic_loss, loss_dict = iq_loss(agent, current_Q, current_V, next_V, batch)

  logger.log('train/critic_loss', critic_loss, step)

  # Optimize the critic
  self.critic_optimizer.zero_grad()
  critic_loss.backward()
  # step critic
  self.critic_optimizer.step()
  return loss_dict


def iq_update(self,
              policy_buffer,
              expert_buffer,
              logger,
              step,
              only_expert_states=False,
              is_sqil=False,
              use_target=False):
  policy_batch = policy_buffer.get_samples(self.batch_size, self.device)
  expert_batch = expert_buffer.get_samples(self.batch_size, self.device)

  losses = self.iq_update_critic(policy_batch, expert_batch, logger, step,
                                 only_expert_states, is_sqil, use_target)

  # args
  vdice_actor = False
  offline = False
  num_actor_updates = 1
  do_soft_update = False

  if self.actor and step % self.actor_update_frequency == 0:
    if not vdice_actor:

      if offline:
        obs = expert_batch[0]
      else:
        # Use both policy and expert observations
        obs = torch.cat([policy_batch[0], expert_batch[0]], dim=0)

      if num_actor_updates:
        for i in range(num_actor_updates):
          actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step)

      losses.update(actor_alpha_losses)

  if step % self.critic_target_update_frequency == 0:
    if do_soft_update:
      soft_update(self.critic_net, self.critic_target_net, self.critic_tau)
    else:
      hard_update(self.critic_net, self.critic_target_net)
  return losses