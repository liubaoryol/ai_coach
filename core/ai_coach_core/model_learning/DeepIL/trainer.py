import os
import numpy as np
import pandas as pd

from time import time, sleep
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm
from .algo.base import Algorithm
from .env import NormalizedEnv


class Trainer:
  """
    Trainer for all the algorithms

    Parameters
    ----------
    env: NormalizedEnv
        environment for training
    env_test: NormalizedEnv
        environment for testing
    algo: Algorithm
        the algorithm to be trained
    log_dir: str
        path to save logs
    seed: int
        random seed
    num_steps: int
        number of steps to train
    eval_interval: int
        time interval between evaluations
    num_eval_episodes: int
        number of episodes for evaluation
    """
  def __init__(self,
               env: NormalizedEnv,
               env_test: NormalizedEnv,
               algo: Algorithm,
               log_dir: str,
               seed: int = 0,
               num_steps: int = 10**5,
               num_pretrain_steps: int = 10**4,
               eval_interval: int = 10**3,
               num_eval_episodes: int = 5):
    super().__init__()

    # Env to collect samples.
    self.env = env
    self.env.seed(seed)

    # Env for evaluation.
    self.env_test = env_test
    self.env_test.seed(2**31 - seed)

    self.algo = algo
    self.log_dir = log_dir

    # Log setting.
    self.summary_dir = os.path.join(log_dir, 'summary')
    self.writer = SummaryWriter(log_dir=self.summary_dir)
    self.model_dir = os.path.join(log_dir, 'model')
    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)

    # Other parameters.
    self.num_pretrain_steps = num_pretrain_steps
    self.num_steps = num_steps
    self.eval_interval = eval_interval
    self.num_eval_episodes = num_eval_episodes

  def pretrain(self):
    self.algo.set_pretrain(True)
    self.start_time = time()
    for step in tqdm(range(1, self.num_pretrain_steps + 1)):
      self.algo.update(self.writer)
      # Evaluate regularly.
      if step % self.eval_interval == 0:
        self.evaluate(step)
        self.algo.save_models(os.path.join(self.model_dir, f'pre_step{step}'))

  def train(self):
    """Start training"""
    self.algo.set_pretrain(False)
    # Time to start training.
    self.start_time = time()
    # Episode's timestep.
    t = 0
    # Initialize the environment.
    state = self.env.reset()
    # Initialize the latent.
    latent, _ = self.algo.explore_latent(0, state)

    for step in tqdm(range(1, self.num_steps + 1)):
      t += 1
      action, log_pi = self.algo.explore(state, latent)
      next_state, reward, done, _ = self.env.step(action)
      if self.algo.cb_reward:
        reward = self.algo.cb_reward(state, latent, action, reward)
      next_latent, _ = self.algo.explore_latent(t, next_state, latent, action,
                                                state)
      done |= (t == self.env.max_episode_steps) or self.algo.is_max_time(t)

      self.algo.add_to_buffer(state, latent, action, reward, done, log_pi,
                              next_state, next_latent)

      if done:
        t = 0
        next_state = self.env.reset()
        next_latent, _ = self.algo.explore_latent(0, next_state)

      state = next_state
      latent = next_latent

      # Update the algorithm whenever ready.
      if self.algo.is_update(step):
        self.algo.update(self.writer)

      # Evaluate regularly.
      if step % self.eval_interval == 0:
        self.evaluate(step)
        self.algo.save_models(os.path.join(self.model_dir, f'step{step}'))

    # Wait for the logging to be finished.
    sleep(30)

    # save rewards as csv
    summary = event_accumulator.EventAccumulator(self.summary_dir)
    summary.Reload()
    returns = pd.DataFrame(summary.Scalars('return/test'))
    returns.to_csv(os.path.join(self.log_dir, 'rewards.csv'), index=False)
    sleep(30)

  def evaluate(self, step: int):
    """
        Evaluate the algorithm

        Parameters
        ----------
        step: int
            current training step
        """
    returns = []
    # mean_return = 0.0

    for _ in range(self.num_eval_episodes):
      state = self.env_test.reset()
      episode_return = 0.0
      done = False
      t = 0
      latent = self.algo.get_latent(0, state)

      while t < self.env.max_episode_steps:
        t += 1
        action = self.algo.exploit(state, latent)
        next_state, reward, done, _ = self.env_test.step(action)
        if self.algo.cb_reward:
          reward = self.algo.cb_reward(state, latent, action, reward)

        latent = self.algo.get_latent(t, next_state, latent, action, state)
        state = next_state
        episode_return += reward
        if done or not self.algo.is_max_time(t):
          break

      returns.append(episode_return)
      # mean_return += episode_return / self.num_eval_episodes

    self.writer.add_scalar('return/test', np.mean(returns), step)
    tqdm.write(
        f'Num steps: {step:<6}, '
        f'Return: {np.mean(returns):<5.1f}, '
        f'Min/Max Return: {np.min(returns):<5.1f}/{np.max(returns):<5.1f}, '
        f'Time: {self.time}')

  @property
  def time(self):
    """Return current training time"""
    return str(timedelta(seconds=int(time() - self.start_time)))
