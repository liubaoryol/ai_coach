import abc
from typing import Optional
import numpy as np
from utils.mdp_utils import StateSpace
from models.mdp import MDP


class MindInterface:
  __metaclass__ = abc.ABCMeta

  def __init__(self, mdp: MDP) -> None:
    self.mdp = mdp
    self.latent_space = None  # type: StateSpace
    self.num_latent = None  # int
    self.init_latentspace()
    self.init_latentspace_helper_vars()

  @abc.abstractmethod
  def transition_mental_state(self, obstate_idx: int, action_idx: int,
                              latstate_idx: int) -> np.ndarray:
    'return: 1-D array'
    raise NotImplementedError

  @abc.abstractmethod
  def init_latentspace(self) -> None:
    self.latent_space = StateSpace()
    raise NotImplementedError

  def init_latentspace_helper_vars(self) -> None:
    self.num_latent = self.latent_space.num_states

  @abc.abstractmethod
  def initial_distribution(self) -> np.ndarray:
    'return: 1-D array'
    raise NotImplementedError


class PolicyInterface:
  __metaclass__ = abc.ABCMeta

  def __init__(self, mdp: MDP, num_action_space: int = 1) -> None:
    self.mdp = mdp
    self._num_action_space = num_action_space

  @abc.abstractmethod
  def policy(self, obstate_idx: int, latstate_idx: int) -> np.ndarray:
    '''
        returns the distribution of an agent's actions as the numpy array where
        the 1st column is the probability and the rest columns are joint actions
    '''
    raise NotImplementedError

  @property
  def num_action_space(self):
    return self._num_action_space


class Agent:
  __metaclass__ = abc.ABCMeta

  def __init__(self, mind_model: MindInterface,
               policy_model: PolicyInterface) -> None:
    self.mind_model = mind_model
    self.policy_model = policy_model
    self.current_latent = None  # type: int

  def set_current_latent(self, init_latent: Optional[int] = None):
    if init_latent is None:
      np_p_init = self.mind_model.initial_distribution()
      self.current_latent = np.random.choice(range(self.mind_model.num_latent),
                                             p=np_p_init)
    else:
      self.current_latent = init_latent

  def get_action(self, obstate_idx: int, latstate_idx: int):
    np_p_next_action = self.policy_model.policy(obstate_idx, latstate_idx)
    idx = np.random.choice(range(self.policy_model.num_action_space),
                           p=np_p_next_action[:, 0])
    return np_p_next_action[idx, 1:]

  def update_mental_state(self, obstate_idx: int, action_idx: int):
    np_p_next_latent = self.mind_model.transition_mental_state(
        obstate_idx, action_idx, self.current_latent)

    self.current_latent = np.random.choice(np_p_next_latent[:, 1],
                                           p=np_p_next_latent[:, 0])
