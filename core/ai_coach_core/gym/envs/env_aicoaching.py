from typing import Sequence, Optional
import gym
from gym import spaces
from ai_coach_core.models.mdp import MDP
import numpy as np


class EnvFromLearnedModels(gym.Env):
  # uncomment below line if you need to render the environment
  # metadata = {'render.modes': ['console']}

  def __init__(self,
               mdp: MDP,
               np_abs: np.ndarray,
               list_np_pi: np.ndarray,
               list_np_tx: np.ndarray,
               list_np_bx: np.ndarray,
               possible_init_states: Optional[Sequence[int]] = None,
               init_state_dist: Optional[np.ndarray] = None,
               use_central_action=True):
    '''
    either possible_init_state or init_state_dist should not be None
    '''
    assert possible_init_states or init_state_dist

    super().__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:

    self.num_abs = np_abs.shape[-1]
    self.num_agents = len(list_np_tx)
    self.tup_num_mental = tuple(
        [list_np_tx[idx_a].shape[-1] for idx_a in range(self.num_agents)])
    self.np_abs = np_abs
    self.list_np_pi = list_np_pi
    self.list_np_tx = list_np_tx
    self.list_np_bx = list_np_bx
    self.use_central_action = use_central_action

    self.observation_space = spaces.Discrete(self.num_abs)
    if use_central_action:
      self.action_space = spaces.Discrete(np.prod(self.tup_num_mental))
    else:
      self.action_space = spaces.MultiDiscrete(self.tup_num_mental)

    self.each_2_joint = np.arange(np.prod(self.tup_num_mental)).reshape(
        self.tup_num_mental)
    self.joint_2_each = {
        ind: coord
        for coord, ind in np.ndenumerate(self.each_2_joint)
    }

    self.mdp = mdp
    self.possible_init_states = possible_init_states
    self.init_state_dist = init_state_dist

    if possible_init_states:
      self.sample_init_s = lambda: int(
          np.random.choice(self.possible_init_states))
    else:
      self.sample_init_s = lambda: int(
          np.random.choice(mdp.num_states, p=self.init_state_dist))

    self.reset()

  def step(self, joint_mental):
    info = {}
    if self.use_central_action:
      each_x = self.joint_2_each[joint_mental]
    else:
      each_x = joint_mental

    real_action = []
    for idx_a in range(self.num_agents):
      act_dist = self.list_np_pi[idx_a][each_x[idx_a], self.cur_abs]
      num_act = len(act_dist)
      act = np.random.choice(num_act, p=act_dist)
      real_action.append(act)

    real_action_idx = self.mdp.conv_action_to_idx(tuple(real_action))

    if real_action_idx not in self.mdp.legal_actions(self.cur_state):
      info["invalid_transition"] = True
      return self.cur_abs, -10000, False, info

    self.cur_state = self.mdp.transition(self.cur_state, real_action_idx)
    self.cur_abs = self.conv_obstate_to_abstate(self.cur_state)

    reward = -1  # we don't need reward for imitation learning
    done = self.mdp.is_terminal(self.cur_state)

    return self.cur_abs, reward, done, info

  def reset(self):
    self.cur_state = self.sample_init_s()
    self.cur_abs = self.conv_obstate_to_abstate(self.cur_state)

    return self.cur_abs  # reward, done, info can't be included

  def conv_obstate_to_abstate(self, obstate_idx):
    TOP_1 = True
    TOP_3 = False
    if TOP_1:
      abstate = np.argmax(self.np_abs[obstate_idx])
    elif TOP_3:
      ind = np.argpartition(self.np_abs[obstate_idx], -3)[-3:]
      np_new_dist = self.np_abs[obstate_idx][ind]
      print(np_new_dist)
      np_new_dist = np_new_dist / np.sum(np_new_dist)[..., None]
      abstate = np.random.choice(ind, p=np_new_dist)
    else:
      abstate = np.random.choice(self.np_abs.shape[-1],
                                 p=self.np_abs[obstate_idx])
    return abstate

  # implement render function if need to be
  # def render(self, mode='human'):
  #   ...

  # implement close function if need to be
  # def close(self):
  #   ...