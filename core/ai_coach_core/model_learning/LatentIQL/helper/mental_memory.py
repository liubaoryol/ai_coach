from collections import deque
import numpy as np
import random
import torch

from ai_coach_core.model_learning.IQLearn.utils.atari_wrapper import LazyFrames


class MentalMemory(object):

  def __init__(self, memory_size: int, seed: int = 0) -> None:
    random.seed(seed)
    self.memory_size = memory_size
    self.buffer = deque(maxlen=self.memory_size)

  def add(self, experience) -> None:
    'experience: obs, prev_lat, prev_act, next_obs, latent, action'
    self.buffer.append(experience)

  def size(self):
    return len(self.buffer)

  def sample(self, batch_size: int, continuous: bool = True):
    if batch_size > len(self.buffer):
      batch_size = len(self.buffer)
    if continuous:
      rand = random.randint(0, len(self.buffer) - batch_size)
      return [self.buffer[i] for i in range(rand, rand + batch_size)]
    else:
      indexes = np.random.choice(np.arange(len(self.buffer)),
                                 size=batch_size,
                                 replace=False)
      return [self.buffer[i] for i in indexes]

  def clear(self):
    self.buffer.clear()

  def save(self, path):
    b = np.asarray(self.buffer)
    print(b.shape)
    np.save(path, b)

  def get_samples(self, batch_size, device):
    batch = self.sample(batch_size, False)

    (batch_obs, batch_prev_lat, batch_prev_act, batch_next_obs, batch_latent,
     batch_action, batch_reward, batch_done) = zip(*batch)

    # Scale obs for atari. TODO: Use flags
    if isinstance(batch_obs[0], LazyFrames):
      # Use lazyframes for improved memory storage (same as original DQN)
      batch_obs = np.array(batch_obs) / 255.0
    if isinstance(batch_next_obs[0], LazyFrames):
      batch_next_obs = np.array(batch_next_obs) / 255.0
    batch_obs = np.array(batch_obs)
    batch_next_obs = np.array(batch_next_obs)
    batch_action = np.array(batch_action)
    batch_prev_lat = np.array(batch_prev_lat)
    batch_prev_act = np.array(batch_prev_act)
    batch_latent = np.array(batch_latent)

    batch_obs = torch.as_tensor(batch_obs, dtype=torch.float, device=device)
    batch_next_obs = torch.as_tensor(batch_next_obs,
                                     dtype=torch.float,
                                     device=device)
    batch_action = torch.as_tensor(batch_action,
                                   dtype=torch.float,
                                   device=device)
    batch_prev_lat = torch.as_tensor(batch_prev_lat,
                                     dtype=torch.float,
                                     device=device)
    batch_prev_act = torch.as_tensor(batch_prev_act,
                                     dtype=torch.float,
                                     device=device)
    batch_latent = torch.as_tensor(batch_latent,
                                   dtype=torch.float,
                                   device=device)
    batch_reward = torch.as_tensor(batch_reward,
                                   dtype=torch.float,
                                   device=device).unsqueeze(1)
    batch_done = torch.as_tensor(batch_done, dtype=torch.float,
                                 device=device).unsqueeze(1)

    return (batch_obs, batch_prev_lat, batch_prev_act, batch_next_obs,
            batch_latent, batch_action, batch_reward, batch_done)
