from dataclasses import dataclass
import torch
from torch import nn
import numpy as np
import gym_custom  # noqa: F401


@dataclass
class LatentConfig:
  latent_size: int
  discrete_latent: bool
  actor_units: tuple
  critic_units: tuple
  disc_units: tuple
  trans_units: tuple
  hidden_activation: nn.Module

  def get_init_latent(self, states: torch.Tensor) -> torch.Tensor:
    return torch.randint(low=0,
                         high=self.latent_size,
                         size=(len(states), 1),
                         dtype=torch.float)

  def get_reward(self, state: np.array, latent: np.array, action: np.array,
                 reward: float):
    return reward

  def get_latent(self, t: int, state: np.array, prev_latent: np.array,
                 prev_action: np.array, prev_state: np.array):
    pass


class LatentConfigEnv2(LatentConfig):
  def get_init_latent(self, states: torch.Tensor) -> torch.Tensor:
    return torch.rand(size=(len(states), self.latent_size), dtype=torch.float)


class CircleWorldLatentConfig(LatentConfig):
  scaler = 0.5
  half_sz = 5

  def get_dir(self, state):
    center = np.array([0, self.half_sz])
    dir = state - center
    len_dir = np.linalg.norm(dir)
    if len_dir != 0:
      dir /= len_dir
    return dir

  def get_pos(self, dir):
    neg_y = np.array([0.0, -1.0])
    sin = np.cross(neg_y, dir)
    cos = np.dot(dir, neg_y)
    return sin, cos

  def get_reward(self, state: np.array, latent: np.array, action: np.array,
                 reward: float):
    next_state = state + self.scaler * action
    next_state[0] = min(self.half_sz, max(-self.half_sz, next_state[0]))
    next_state[1] = min(2 * self.half_sz, max(0, next_state[1]))

    if latent[0] == 1:
      reward = -reward

    dir = self.get_dir(state)
    sin_p, cos_p = self.get_pos(dir)

    dir = self.get_dir(next_state)
    sin_n, cos_n = self.get_pos(dir)
    if (sin_n * sin_p < 0 and cos_p > 0 and cos_n > 0 and cos_p < 0.95
        and cos_n < 0.95):
      reward = -1

    if latent[0] == 0:
      if sin_p < 0 and cos_p >= 0.95 and cos_n < 0.95:
        return -1
    else:  # prev_latent[0] == 1
      if sin_p > 0 and cos_p >= 0.95 and cos_n < 0.95:
        return -1

    return reward

  def get_latent(self, t: int, state: np.array, prev_latent: np.array,
                 prev_action: np.array, prev_state: np.array):
    if t == 0:
      state = torch.tensor(state, dtype=torch.float).unsqueeze_(0)
      return self.get_init_latent(state).cpu().numpy()[0]
    else:
      dir = self.get_dir(prev_state)
      sin_p, cos_p = self.get_pos(dir)

      dir = self.get_dir(state)
      _, cos_n = self.get_pos(dir)

      if prev_latent[0] == 0:
        if sin_p < 0 and cos_p < 0.95 and cos_n >= 0.95:
          return np.array([1], dtype=np.float32)
      else:  # prev_latent[0] == 1
        if sin_p > 0 and cos_p < 0.95 and cos_n >= 0.95:
          return np.array([0], dtype=np.float32)

      return prev_latent


env1_latent = LatentConfig(4,
                           True,
                           actor_units=(64, 64),
                           critic_units=(64, 64),
                           disc_units=(100, 100),
                           trans_units=(64, 64),
                           hidden_activation=nn.Tanh())
env2_latent = LatentConfigEnv2(1,
                               False,
                               actor_units=(64, 64),
                               critic_units=(64, 64),
                               disc_units=(100, 100),
                               trans_units=(64, 64),
                               hidden_activation=nn.Tanh())
circleworld_latent = CircleWorldLatentConfig(2,
                                             True,
                                             actor_units=(128, 128),
                                             critic_units=(128, 128),
                                             disc_units=(128, 128),
                                             trans_units=(128, 128),
                                             hidden_activation=nn.ReLU())

LATENT_CONFIG = {
    "circleworld-v0": circleworld_latent,
}
