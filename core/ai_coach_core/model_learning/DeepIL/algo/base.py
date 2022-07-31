import os
import numpy as np
import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter
from abc import abstractmethod
from typing import Tuple, Optional, Callable
from .utils import one_hot
from ..network import (AbstractPolicy, AbstractTransition, DiscretePolicy,
                       ContinousPolicy, DiscreteTransition, ContinousTransition)
from .utils import disable_gradient

T_InitLatent = Callable[[torch.Tensor], torch.Tensor]
T_GetLatent = Callable[[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                       np.ndarray]
T_Exploit = Callable[[np.ndarray, np.ndarray], np.ndarray]
T_GetReward = Callable[[np.ndarray, np.ndarray, np.ndarray, float], float]


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
  def __init__(self, state_size: torch.Tensor, latent_size: torch.Tensor,
               action_size: torch.Tensor, discrete_state: bool,
               discrete_latent: bool, discrete_action: bool,
               actor: AbstractPolicy, transition: AbstractTransition,
               cb_init_latent: T_InitLatent, device: torch.device, seed: int,
               gamma: float):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    self.actor = actor
    self.trans = transition
    self.cb_init_latent = cb_init_latent

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

    self.cb_reward: Optional[T_GetReward] = None

  def np_to_input(self, input: np.ndarray, size: int,
                  discrete: bool) -> torch.Tensor:
    if discrete:
      return one_hot(torch.from_numpy(input).unsqueeze_(0),
                     size,
                     device=self.device)
    else:
      return torch.tensor(input, dtype=torch.float,
                          device=self.device).unsqueeze_(0)

  def explore(self, state: np.ndarray,
              latent: np.ndarray) -> Tuple[np.ndarray, float]:
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
      action, log_pi = self.actor.sample(state, latent)
    return action.cpu().numpy()[0], log_pi.item()

  def is_max_time(self, t: int):
    return False

  def exploit(self, state: np.ndarray, latent: np.ndarray) -> np.ndarray:
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
      action = self.actor.exploit(state, latent)
    return action.cpu().numpy()[0]

  def set_pretrain(self, pretrain_mode: bool):
    self.pretrain_mode = pretrain_mode

  def explore_latent(
      self,
      t: int,
      state: Optional[np.ndarray] = None,
      prev_latent: Optional[np.ndarray] = None,
      prev_action: Optional[np.ndarray] = None,
      prev_state: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:

    if t == 0:
      state = torch.tensor(state, dtype=torch.float,
                           device=self.device).unsqueeze_(0)
      return self.cb_init_latent(state).cpu().numpy()[0], float("-inf")
    else:
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

  def get_latent(self,
                 t: int,
                 state: Optional[np.ndarray] = None,
                 prev_latent: Optional[np.ndarray] = None,
                 prev_action: Optional[np.ndarray] = None,
                 prev_state: Optional[np.ndarray] = None) -> np.ndarray:

    if t == 0:
      state = torch.tensor(state, dtype=torch.float,
                           device=self.device).unsqueeze_(0)
      return self.cb_init_latent(state).cpu().numpy()[0]
    else:
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

  def add_to_buffer(self, state: np.ndarray, latent: np.ndarray,
                    action: np.ndarray, reward: float, done: bool,
                    log_pi: float, next_state: np.ndarray,
                    next_latent: np.ndarray):
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
    Base class for the expert
    """
  def __init__(self, cb_exploit: T_Exploit, cb_get_latent: T_GetLatent):
    self.cb_exploit = cb_exploit
    self.cb_get_latent = cb_get_latent
    self.cb_reward: Optional[T_GetReward] = None

  def set_reward(self, cb_reward: T_GetReward):
    self.cb_reward = cb_reward

  def exploit(self, state: np.ndarray, latent: np.ndarray) -> np.ndarray:
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
    return self.cb_exploit(state, latent)

  def get_latent(self,
                 t: int,
                 state: Optional[np.ndarray] = None,
                 prev_latent: Optional[np.ndarray] = None,
                 prev_action: Optional[np.ndarray] = None,
                 prev_state: Optional[np.ndarray] = None) -> np.ndarray:

    return self.cb_get_latent(t, state, prev_latent, prev_action, prev_state)


class NNExpert(Expert):
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
               cb_init_latent: T_InitLatent, device: torch.device,
               path_actor: str, path_trans: str, units_actor: Tuple,
               units_trans: Tuple):
    super().__init__(self._exploit, self._get_latent)
    self.device = device
    if discrete_action:
      self.actor = DiscretePolicy(state_size,
                                  latent_size,
                                  action_size,
                                  units_actor,
                                  hidden_activation=nn.Tanh())
    else:
      self.actor = ContinousPolicy(state_size,
                                   latent_size,
                                   action_size,
                                   units_actor,
                                   hidden_activation=nn.Tanh())
    if discrete_latent:
      self.trans = DiscreteTransition(state_size,
                                      latent_size,
                                      action_size,
                                      units_trans,
                                      hidden_activation=nn.Tanh())
    else:
      self.trans = ContinousTransition(state_size,
                                       latent_size,
                                       action_size,
                                       units_trans,
                                       hidden_activation=nn.Tanh())
    self.cb_init_latent = cb_init_latent

    self.state_size = state_size
    self.latent_size = latent_size
    self.action_size = action_size

    self.discrete_state = discrete_state
    self.discrete_latent = discrete_latent
    self.discrete_action = discrete_action

    self.actor.load_state_dict(torch.load(path_actor, map_location=device))
    self.trans.load_state_dict(torch.load(path_trans, map_location=device))
    disable_gradient(self.actor)
    disable_gradient(self.trans)

  def np_to_input(self, input: np.ndarray, size: int,
                  discrete: bool) -> torch.Tensor:
    if discrete:
      return one_hot(torch.from_numpy(input).unsqueeze_(0),
                     size,
                     device=self.device)
    else:
      return torch.tensor(input, dtype=torch.float,
                          device=self.device).unsqueeze_(0)

  def _exploit(self, state: np.ndarray, latent: np.ndarray) -> np.ndarray:
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
      action = self.actor.exploit(state, latent)
    return action.cpu().numpy()[0]

  def _get_latent(self,
                  t: int,
                  state: Optional[np.ndarray] = None,
                  prev_latent: Optional[np.ndarray] = None,
                  prev_action: Optional[np.ndarray] = None,
                  prev_state: Optional[np.ndarray] = None) -> np.ndarray:

    if t == 0:
      state = torch.tensor(state, dtype=torch.float,
                           device=self.device).unsqueeze_(0)
      return self.cb_init_latent(state).cpu().numpy()[0]
    else:
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
