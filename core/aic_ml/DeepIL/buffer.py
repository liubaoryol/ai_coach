import os
import numpy as np
import torch
import random
from typing import Tuple


class SerializedBuffer:
  """
    Serialized buffer, containing [states, latents, actions, rewards, done, 
                                                      next_states, next_latents]
      and trajectories, often used as demonstrations

    Parameters
    ----------
    path: str
        path to the saved buffer
    device: torch.device
        cpu or cuda
    label_ratio: float
        ratio of labeled data
    sparse_sample: bool
        if true, sample the buffer with the largest gap.
        Often used when the buffer is not shuffled
    use_mean: bool
        if true, use the mean reward of the trajectory s-a pairs
          to sort trajectories
    """
  def __init__(self,
               path: str,
               device: torch.device,
               label_ratio: float = 1,
               sparse_sample: bool = True,
               use_mean: bool = False):
    tmp = torch.load(path)
    self.buffer_size = self._n = tmp['state'].size(0)
    self.device = device

    self.states = tmp['state'].clone().to(self.device)
    self.latents = tmp['latent'].clone().to(self.device)
    self.actions = tmp['action'].clone().to(self.device)
    self.rewards = tmp['reward'].clone().to(self.device)
    self.dones = tmp['done'].clone().to(self.device)
    self.next_states = tmp['next_state'].clone().to(self.device)
    self.next_latents = tmp['next_latent'].clone().to(self.device)

    self.traj_states = []
    self.traj_latents = []
    self.traj_actions = []
    self.traj_rewards = []
    self.traj_next_states = []
    self.traj_next_latents = []

    all_traj_states = []
    all_traj_latents = []
    all_traj_actions = []
    all_traj_rewards = []
    all_traj_next_states = []
    all_traj_next_latents = []

    self.n_traj = 0
    traj_states = torch.Tensor([]).to(self.device)
    traj_latents = torch.Tensor([]).to(self.device)
    traj_actions = torch.Tensor([]).to(self.device)
    traj_rewards = 0
    traj_next_states = torch.Tensor([]).to(self.device)
    traj_next_latents = torch.Tensor([]).to(self.device)
    traj_length = 0
    for i, done in enumerate(self.dones):
      traj_states = torch.cat((traj_states, self.states[i].unsqueeze(0)), dim=0)
      traj_latents = torch.cat((traj_latents, self.latents[i].unsqueeze(0)),
                               dim=0)
      traj_actions = torch.cat((traj_actions, self.actions[i].unsqueeze(0)),
                               dim=0)
      traj_rewards += self.rewards[i]
      traj_next_states = torch.cat(
          (traj_next_states, self.next_states[i].unsqueeze(0)), dim=0)
      traj_next_latents = torch.cat(
          (traj_next_latents, self.next_latents[i].unsqueeze(0)), dim=0)
      traj_length += 1
      if done == 1:
        all_traj_states.append(traj_states)
        all_traj_latents.append(traj_latents)
        all_traj_actions.append(traj_actions)
        if use_mean:
          all_traj_rewards.append(traj_rewards / traj_length)
        else:
          all_traj_rewards.append(traj_rewards)
        all_traj_next_states.append(traj_next_states)
        all_traj_next_latents.append(traj_next_latents)
        traj_states = torch.Tensor([]).to(self.device)
        traj_latents = torch.Tensor([]).to(self.device)
        traj_actions = torch.Tensor([]).to(self.device)
        traj_rewards = 0
        traj_next_states = torch.Tensor([]).to(self.device)
        traj_next_latents = torch.Tensor([]).to(self.device)
        self.n_traj += 1
        traj_length = 0

    i_traj = random.sample(range(self.n_traj), int(label_ratio * self.n_traj))
    n_labeled_traj = int(label_ratio * self.n_traj)
    if sparse_sample:
      i_traj = [
          i * int(self.n_traj / n_labeled_traj) for i in range(n_labeled_traj)
      ]
    self.n_traj = n_labeled_traj
    self.label_ratio = label_ratio

    for i in i_traj:
      self.traj_states.append(all_traj_states[i])
      self.traj_latents.append(all_traj_latents[i])
      self.traj_actions.append(all_traj_actions[i])
      self.traj_rewards.append(all_traj_rewards[i])
      self.traj_next_states.append(all_traj_next_states[i])
      self.traj_next_latents.append(all_traj_next_latents[i])

  def sample(
      self, batch_size: int
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
             torch.Tensor, torch.Tensor, torch.Tensor]:
    """
        Sample data from the buffer

        Parameters
        ----------
        batch_size: int
            batch size

        Returns
        -------
        states: torch.Tensor
        latents: torch.Tensor
        actions: torch.Tensor
        rewards: torch.Tensor
        dones: torch.Tensor
        next_states: torch.Tensor
        next_latents: torch.Tensor
        """
    idxes = np.random.randint(low=0, high=self._n, size=batch_size)
    return (self.states[idxes], self.latents[idxes], self.actions[idxes],
            self.rewards[idxes], self.dones[idxes], self.next_states[idxes],
            self.next_latents[idxes])

  def get(
      self
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
             torch.Tensor, torch.Tensor, torch.Tensor]:
    """
        Get all data in the buffer

        Returns
        -------
        states: torch.Tensor
        latents: torch.Tensor
        actions: torch.Tensor
        rewards: torch.Tensor
        dones: torch.Tensor
        next_states: torch.Tensor
        next_latents: torch.Tensor
        """
    return (self.states, self.latents, self.actions, self.rewards, self.dones,
            self.next_states, self.next_latents)

  def sample_traj(self,
                  batch_size: int) -> Tuple[list, list, list, list, list, list]:
    """
        Sample trajectories from the buffer

        Parameters
        ----------
        batch_size: int
            number of trajectories in a batch

        Returns
        -------
        sample_states: a list of torch.Tensor
            each tensor is the states in one trajectory
        sample_latents: a list of torch.Tensor
            each tensor is the latent states in one trajectory
        sample_actions: a list of torch.Tensor
            each tensor is the actions in one trajectory
        sample_rewards: a list of torch.Tensor
            each tensor is the rewards in one trajectory
        sample_next_states: a list of torch.Tensor
            each tensor is the next_states in one trajectory
        sample_next_latents: a list of torch.Tensor
            each tensor is the next_latents in one trajectory
        """
    idxes = np.random.randint(low=0, high=self.n_traj, size=batch_size)
    sample_states = []
    sample_latents = []
    sample_actions = []
    sample_rewards = []
    sample_next_states = []
    sample_next_latents = []

    for i in idxes:
      sample_states.append(self.traj_states[i])
      sample_latents.append(self.traj_latents[i])
      sample_actions.append(self.traj_actions[i])
      sample_rewards.append(self.traj_rewards[i])
      sample_next_states.append(self.traj_next_states[i])
      sample_next_latents.append(self.traj_next_latents[i])

    return (sample_states, sample_latents, sample_actions, sample_rewards,
            sample_next_states, sample_next_latents)


class Buffer(SerializedBuffer):
  """
    Buffer used while collecting demonstrations

    Parameters
    ----------
    buffer_size: int
        size of the buffer
    state_size: int
        size of the state space
    latent_size: int
        size of the latent state space
    action_size: int
        size of the action space
    device: torch.device
        cpu or cuda
    """
  def __init__(self, buffer_size: int, state_size: int, latent_size: int,
               action_size: int, discrete_state: bool, discrete_latent: bool,
               discrete_action: bool, device: torch.device):
    self._n = 0
    self._p = 0
    self.buffer_size = buffer_size
    self.device = device

    state_size = 1 if discrete_state else state_size
    latent_size = 1 if discrete_latent else latent_size
    action_size = 1 if discrete_action else action_size

    self.states = torch.empty((buffer_size, state_size),
                              dtype=torch.float,
                              device=device)
    self.latents = torch.empty((buffer_size, latent_size),
                               dtype=torch.float,
                               device=device)
    self.actions = torch.empty((buffer_size, action_size),
                               dtype=torch.float,
                               device=device)
    self.rewards = torch.empty((buffer_size, 1),
                               dtype=torch.float,
                               device=device)
    self.dones = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
    self.next_states = torch.empty((buffer_size, state_size),
                                   dtype=torch.float,
                                   device=device)
    self.next_latents = torch.empty((buffer_size, latent_size),
                                    dtype=torch.float,
                                    device=device)

  def append(self, state: np.array, latent: np.array, action: np.array,
             reward: float, done: bool, next_state: np.array,
             next_latent: np.array):
    """
        Save a transition in the buffer

        Parameters
        ----------
        state: np.array
            current state
        latent: np.array
            current latent state
        action: np.array
            action taken in the state
        reward: float
            reward of the s-a pair
        done: bool
            whether the state is the end of the episode
        next_state: np.array
            next states that the s-x-a pair transferred to
        next_latent: np.array
            next latent states that the s-x-a pair transferred to
        """
    self.states[self._p].copy_(torch.from_numpy(state))
    self.latents[self._p].copy_(torch.from_numpy(latent))
    self.actions[self._p].copy_(torch.from_numpy(action))
    self.rewards[self._p] = float(reward)
    self.dones[self._p] = float(done)
    self.next_states[self._p].copy_(torch.from_numpy(next_state))
    self.next_latents[self._p].copy_(torch.from_numpy(next_latent))

    self._p = (self._p + 1) % self.buffer_size
    self._n = min(self._n + 1, self.buffer_size)

  def save(self, path: str):
    """
        Save the buffer

        Parameters
        ----------
        path: str
            path to save
        """
    if not os.path.exists(os.path.dirname(path)):
      os.makedirs(os.path.dirname(path))

    torch.save(
        {
            'state': self.states.clone().cpu(),
            'latent': self.latents.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'next_state': self.next_states.clone().cpu(),
            'next_latent': self.next_latents.clone().cpu(),
        }, path)


class RolloutBuffer:
  """
    Rollout buffer that often used in training RL agents

    Parameters
    ----------
    buffer_size: int
        size of the buffer
    state_size: int
        size of the state space
    action_size: int
        size of the action space
    device: torch.device
        cpu or cuda
    mix: int
        the buffer will be mixed using these time of data
    """
  def __init__(self,
               buffer_size: int,
               state_size: int,
               latent_size: int,
               action_size: int,
               discrete_state: bool,
               discrete_latent: bool,
               discrete_action: bool,
               device: torch.device,
               mix: int = 1):
    self._n = 0
    self._p = 0
    self.mix = mix
    self.device = device
    self.buffer_size = buffer_size
    self.total_size = mix * buffer_size

    state_size = 1 if discrete_state else state_size
    latent_size = 1 if discrete_latent else latent_size
    action_size = 1 if discrete_action else action_size

    self.states = torch.empty((self.total_size, state_size),
                              dtype=torch.float,
                              device=device)
    self.latents = torch.empty((self.total_size, latent_size),
                               dtype=torch.float,
                               device=device)
    self.actions = torch.empty((self.total_size, action_size),
                               dtype=torch.float,
                               device=device)
    self.rewards = torch.empty((self.total_size, 1),
                               dtype=torch.float,
                               device=device)
    self.dones = torch.empty((self.total_size, 1),
                             dtype=torch.float,
                             device=device)
    self.log_pis = torch.empty((self.total_size, 1),
                               dtype=torch.float,
                               device=device)
    self.next_states = torch.empty((self.total_size, state_size),
                                   dtype=torch.float,
                                   device=device)
    self.next_latents = torch.empty((self.total_size, latent_size),
                                    dtype=torch.float,
                                    device=device)

  def append(self, state: np.array, latent: np.array, action: np.array,
             reward: float, done: bool, log_pi: float, next_state: np.array,
             next_latent: np.array):
    """
        Save a transition in the buffer

        Parameters
        ----------
        state: np.array
            current state
        latent: np.array
            current latent state
        action: np.array
            action taken in the state
        reward: float
            reward of the s-a pair
        done: bool
            whether the state is the end of the episode
        log_pi: float
            log(\pi(a|s, x))
        next_state: np.array
            next states that the s-a pair transferred to
        next_latent: np.array
            next latent states that the s-a pair transferred to
        """
    self.states[self._p].copy_(torch.from_numpy(state))
    self.latents[self._p].copy_(torch.from_numpy(latent))
    self.actions[self._p].copy_(torch.from_numpy(action))
    self.rewards[self._p] = float(reward)
    self.dones[self._p] = float(done)
    self.log_pis[self._p] = float(log_pi)
    self.next_states[self._p].copy_(torch.from_numpy(next_state))
    self.next_latents[self._p].copy_(torch.from_numpy(next_latent))

    self._p = (self._p + 1) % self.total_size
    self._n = min(self._n + 1, self.total_size)

  def get(
      self
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
             torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
        Get all data in the buffer

        Returns
        -------
        states: torch.Tensor
        latents: torch.Tensor
        actions: torch.Tensor
        rewards: torch.Tensor
        dones: torch.Tensor
        log_pis: torch.Tensor
        next_states: torch.Tensor
        next_latents: torch.Tensor
        """
    assert self._p % self.buffer_size == 0
    start = (self._p - self.buffer_size) % self.total_size
    idxes = slice(start, start + self.buffer_size)
    return (self.states[idxes], self.latents[idxes], self.actions[idxes],
            self.rewards[idxes], self.dones[idxes], self.log_pis[idxes],
            self.next_states[idxes], self.next_latents[idxes])

  def sample(
      self, batch_size: int
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
             torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
        Sample data from the buffer

        Parameters
        ----------
        batch_size: int
            batch size

        Returns
        -------
        states: torch.Tensor
        latents: torch.Tensor
        actions: torch.Tensor
        rewards: torch.Tensor
        dones: torch.Tensor
        log_pis: torch.Tensor
        next_states: torch.Tensor
        next_latents: torch.Tensor
        """
    assert self._p % self.buffer_size == 0
    idxes = np.random.randint(low=0, high=self._n, size=batch_size)
    return (self.states[idxes], self.latents[idxes], self.actions[idxes],
            self.rewards[idxes], self.dones[idxes], self.log_pis[idxes],
            self.next_states[idxes], self.next_latents[idxes])

  def sample_traj(
      self, batch_size: int
  ) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
    """
        Sample trajectories from the buffer

        Parameters
        ----------
        batch_size: int
            number of trajectories in a batch

        Returns
        -------
        sample_states: an array of torch.Tensor
            each tensor is the states in one trajectory
        sample_latents: an array of torch.Tensor
            each tensor is the latent states in one trajectory
        sample_actions: an array of torch.Tensor
            each tensor is the actions in one trajectory
        sample_rewards: an array of torch.Tensor
            each tensor is the rewards in one trajectory
        sample_next_states: an array of torch.Tensor
            each tensor is the next_states in one trajectory
        sample_next_latents: an array of torch.Tensor
            each tensor is the next_latents in one trajectory
        """
    assert self._p % self.buffer_size == 0

    n_traj = 0
    all_traj_states = []
    all_traj_latents = []
    all_traj_actions = []
    all_traj_next_states = []
    all_traj_next_latents = []
    all_traj_rewards = []
    traj_states = torch.Tensor([]).to(self.device)
    traj_latents = torch.Tensor([]).to(self.device)
    traj_actions = torch.Tensor([]).to(self.device)
    traj_next_states = torch.Tensor([]).to(self.device)
    traj_next_latents = torch.Tensor([]).to(self.device)
    traj_rewards = 0
    for i, done in enumerate(self.dones):
      traj_states = torch.cat((traj_states, self.states[i].unsqueeze(0)), dim=0)
      traj_latents = torch.cat((traj_latents, self.latents[i].unsqueeze(0)),
                               dim=0)
      traj_actions = torch.cat((traj_actions, self.actions[i].unsqueeze(0)),
                               dim=0)
      traj_next_states = torch.cat(
          (traj_next_states, self.next_states[i].unsqueeze(0)), dim=0)
      traj_next_latents = torch.cat(
          (traj_next_latents, self.next_latents[i].unsqueeze(0)), dim=0)
      traj_rewards += self.rewards[i]
      if done == 1:
        all_traj_states.append(traj_states)
        all_traj_latents.append(traj_latents)
        all_traj_actions.append(traj_actions)
        all_traj_next_states.append(traj_next_states)
        all_traj_next_latents.append(traj_next_latents)
        all_traj_rewards.append(traj_rewards)
        traj_states = torch.Tensor([]).to(self.device)
        traj_latents = torch.Tensor([]).to(self.device)
        traj_actions = torch.Tensor([]).to(self.device)
        traj_next_states = torch.Tensor([]).to(self.device)
        traj_next_latents = torch.Tensor([]).to(self.device)
        traj_rewards = 0
        n_traj += 1

    idxes = np.random.randint(low=0, high=n_traj, size=batch_size)
    return (np.array(all_traj_states)[idxes], np.array(all_traj_latents)[idxes],
            np.array(all_traj_actions)[idxes],
            np.array(all_traj_rewards)[idxes],
            np.array(all_traj_next_states)[idxes],
            np.array(all_traj_next_latents)[idxes])
