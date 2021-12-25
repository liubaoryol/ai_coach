import gym
from gym import spaces


class GymEnvFromMDP(gym.Env):
  # uncomment below line if you need to render the environment
  # metadata = {'render.modes': ['console']}

  def __init__(self, num_states, num_actions, cb_transition, cb_is_terminal,
               cb_legal_actions, init_state):
    super().__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(num_actions)
    # Example for using image as input (channel-first; channel-last also works):
    self.observation_space = spaces.Discrete(num_states)

    # new members for custom env
    self.num_states = num_states
    self.num_actions = num_actions
    self.cb_transition = cb_transition
    self.cb_is_terminal = cb_is_terminal
    self.cb_legal_actions = cb_legal_actions
    self.init_state = init_state
    self.cur_state = init_state

  def step(self, action):
    info = {}
    if action not in self.cb_legal_actions(self.cur_state):
      return self.cur_state, 0, False, info

    self.cur_state = self.cb_transition(self.cur_state, action)
    # reward = self.mdp.reward(self.cur_state, action)
    reward = -1
    done = self.cb_is_terminal(self.cur_state)

    return self.cur_state, reward, done, info

  def reset(self):
    self.cur_state = self.init_state

    return self.cur_state  # reward, done, info can't be included

  # implement render function if need to be
  # def render(self, mode='human'):
  #   ...

  # implement close function if need to be
  # def close(self):
  #   ...
