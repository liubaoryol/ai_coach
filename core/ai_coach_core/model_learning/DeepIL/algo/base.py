import os
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from abc import abstractmethod
from typing import Tuple, Optional, Callable
from .utils import one_hot
from ..network import AbstractPolicy, AbstractTransition


class Algorithm:
  """
    Base class for all algorithms

    Parameters
    ----------
    device: torch.device
        cpu or cuda
    seed: int
        random seed
    gamma: float
        discount factor
    """
  def __init__(
      self,
      state_size: torch.Tensor,
      latent_size: torch.Tensor,
      action_size: torch.Tensor,
      discrete_state: bool,
      discrete_latent: bool,
      discrete_action: bool,
      actor: AbstractPolicy,
      transition: AbstractTransition,
      #  cb_init_latent: Callable[[torch.Tensor], torch.Tensor],
      device: torch.device,
      seed: int,
      gamma: float):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    self.actor = actor
    self.trans = transition
    # self.cb_init_latent = cb_init_latent

    self.pretrain_mode = False
    self.learning_steps = 0
    self.pretraining_steps = 0
    self.device = device
    self.gamma = gamma

    self.state_size = state_size
    self.latent_size = latent_size
    self.action_size = action_size

    self.discrete_state = discrete_state
    self.discrete_latent = discrete_latent
    self.discrete_action = discrete_action

  def np_to_input(self, input: np.array, size: int, discrete: bool):
    if discrete:
      return one_hot(torch.from_numpy(input), size, device=self.device)
    else:
      return torch.tensor(input, dtype=torch.float, device=self.device)

  def explore(self, state: np.array,
              latent: np.array) -> Tuple[np.array, float]:
    """
        Act with policy with randomness

        Parameters
        ----------
        state: np.array
            current state
        latent: np.array
            current latent state

        Returns
        -------
        action: np.array
            mean action
        log_pi: float
            log(\pi(a|s, x)) of the action
        """
    state = self.np_to_input(state, self.state_size, self.discrete_state)
    latent = self.np_to_input(latent, self.latent_size, self.discrete_latent)

    with torch.no_grad():
      action, log_pi = self.actor.sample(state.unsqueeze_(0),
                                         latent.unsqueeze_(0))
    return action.cpu().numpy()[0], log_pi.item()

  def exploit(self, state: np.array, latent: np.array) -> np.array:
    """
        Act with deterministic policy

        Parameters
        ----------
        state: np.array
            current state
        latent: np.array
            current latent state

        Returns
        -------
        action: np.array
            action to take
        """

    state = self.np_to_input(state, self.state_size, self.discrete_state)
    latent = self.np_to_input(latent, self.latent_size, self.discrete_latent)

    with torch.no_grad():
      action = self.actor.exploit(state.unsqueeze_(0), latent.unsqueeze_(0))
    return action.cpu().numpy()[0]

  def set_pretrain(self, pretrain_mode: bool):
    self.pretrain_mode = pretrain_mode

  def explore_latent(
      self,
      t: int,
      state: Optional[np.array] = None,
      prev_latent: Optional[np.array] = None,
      prev_action: Optional[np.array] = None,
      prev_state: Optional[np.array] = None) -> Tuple[np.array, float]:

    state = self.np_to_input(state, self.state_size, self.discrete_state)
    prev_state = self.np_to_input(prev_state, self.state_size,
                                  self.discrete_state)
    prev_latent = self.np_to_input(prev_latent, self.latent_size,
                                   self.discrete_latent)
    prev_action = self.np_to_input(prev_action, self.action_size,
                                   self.discrete_action)

    with torch.no_grad():
      latent, log_Tx = self.trans.sample(state, prev_latent, prev_action)
    return latent.cpu().numpy()[0], log_Tx.item()

  def exploit_latent(self,
                     t: int,
                     state: Optional[np.array] = None,
                     prev_latent: Optional[np.array] = None,
                     prev_action: Optional[np.array] = None,
                     prev_state: Optional[np.array] = None) -> np.array:

    state = self.np_to_input(state, self.state_size, self.discrete_state)
    prev_state = self.np_to_input(prev_state, self.state_size,
                                  self.discrete_state)
    prev_latent = self.np_to_input(prev_latent, self.latent_size,
                                   self.discrete_latent)
    prev_action = self.np_to_input(prev_action, self.action_size,
                                   self.discrete_action)

    with torch.no_grad():
      latent = self.trans.exploit(state, prev_latent, prev_action)
    return latent.cpu().numpy()[0]

  def add_to_buffer(self, state: np.array, latent: np.array, action: np.array,
                    reward: float, done: bool, log_pi: float,
                    next_state: np.array, next_latent: np.array):
    pass

  @abstractmethod
  def is_update(self, step: int):
    """
        Whether the time is for update

        Parameters
        ----------
        step: int
            current training step
        """
    pass

  @abstractmethod
  def update(self, writer: SummaryWriter):
    """
        Update the algorithm

        Parameters
        ----------
        writer: SummaryWriter
            writer for logs
        """
    pass

  @abstractmethod
  def save_models(self, save_dir: str):
    """
        Save the model

        Parameters
        ----------
        save_dir: str
            path to save
        """
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)


class Expert:
  """
    Base class for all well-trained algorithms

    Parameters
    ----------
    device: torch.device
        cpu or cuda
    """
  def __init__(self, state_size: torch.Tensor, latent_size: torch.Tensor,
               action_size: torch.Tensor, discrete_state: bool,
               discrete_latent: bool, discrete_action: bool,
               actor: AbstractPolicy, transition: AbstractTransition,
               device: torch.device):
    self.device = device
    self.actor = actor
    self.trans = transition

    self.state_size = state_size
    self.latent_size = latent_size
    self.action_size = action_size

    self.discrete_state = discrete_state
    self.discrete_latent = discrete_latent
    self.discrete_action = discrete_action

  def exploit(self, state: np.array, latent: np.array) -> np.array:
    """
        Act with deterministic policy

        Parameters
        ----------
        state: np.array
            current state

        Returns
        -------
        action: np.array
            action to take
        """
    if self.discrete_state:
      state = one_hot(torch.from_numpy(state),
                      self.state_size,
                      device=self.device)
    else:
      state = torch.tensor(state, dtype=torch.float, device=self.device)

    if self.discrete_latent:
      latent = one_hot(torch.from_numpy(latent),
                       self.latent_size,
                       device=self.device)
    else:
      latent = torch.tensor(latent, dtype=torch.float, device=self.device)

    with torch.no_grad():
      action = self.actor.exploit(state.unsqueeze_(0), latent.unsqueeze_(0))
    return action.cpu().numpy()[0]

  def get_latent(
      self,
      t: int,
      state: Optional[np.array] = None,
      prev_latent: Optional[np.array] = None,
      prev_action: Optional[np.array] = None,
      prev_state: Optional[np.array] = None) -> Tuple[np.array, bool]:

    if self.discrete_state:
      state = one_hot(torch.from_numpy(state),
                      self.state_size,
                      device=self.device)
      prev_state = one_hot(torch.from_numpy(prev_state),
                           self.state_size,
                           device=self.device)
    else:
      state = torch.tensor(state, dtype=torch.float, device=self.device)
      prev_state = torch.tensor(prev_state,
                                dtype=torch.float,
                                device=self.device)

    if self.discrete_latent:
      prev_latent = one_hot(torch.from_numpy(prev_latent),
                            self.latent_size,
                            device=self.device)
    else:
      prev_latent = torch.tensor(prev_latent,
                                 dtype=torch.float,
                                 device=self.device)

    if self.discrete_action:
      prev_action = one_hot(torch.from_numpy(prev_action),
                            self.action_size,
                            device=self.device)
    else:
      prev_action = torch.tensor(prev_action,
                                 dtype=torch.float,
                                 device=self.device)
    with torch.no_grad():
      latent, latent_log_probs = self.trans.sample(state, prev_latent,
                                                   prev_action)
    return latent, latent_log_probs
