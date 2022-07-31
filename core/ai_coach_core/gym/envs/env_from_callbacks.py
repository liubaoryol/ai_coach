from typing import Union, Sequence
import collections.abc
import gym
from gym import spaces


class EnvFromCallbacks(gym.Env):
  # uncomment below line if you need to render the environment
  # metadata = {'render.modes': ['console']}

  def __init__(self, num_states: int, num_actions: Union[int, Sequence[int]],
               cb_transition, cb_is_terminal, cb_is_legal_action,
               cb_sample_init_state):
    '''
    num_actions: can be either int or tuple
    '''
    super().__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    if isinstance(num_actions, collections.abc.Sequence):
      if len(num_actions) == 1:
        self.action_space = spaces.Discrete(num_actions[0])
      else:
        self.action_space = spaces.MultiDiscrete(num_actions)
    else:
      self.action_space = spaces.Discrete(num_actions)

    # Example for using image as input (channel-first; channel-last also works):
    self.observation_space = spaces.Discrete(num_states)

    # new members for custom env
    self.cb_transition = cb_transition
    self.cb_is_terminal = cb_is_terminal
    self.cb_is_legal_action = cb_is_legal_action
    self.cb_sample = cb_sample_init_state
    self.cur_state = self.cb_sample()

  def step(self, action):
    info = {}
    if not self.cb_is_legal_action(self.cur_state, action):
      info["invalid_transition"] = True
      return self.cur_state, 0, False, info

    self.cur_state = self.cb_transition(self.cur_state, action)
    # reward = self.mdp.reward(self.cur_state, action)
    reward = -1
    done = self.cb_is_terminal(self.cur_state)

    return self.cur_state, reward, done, info

  def reset(self):
    self.cur_state = self.cb_sample()

    return self.cur_state  # reward, done, info can't be included

  # implement render function if need to be
  # def render(self, mode='human'):
  #   ...

  # implement close function if need to be
  # def close(self):
  #   ...
