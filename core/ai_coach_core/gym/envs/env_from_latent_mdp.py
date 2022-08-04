from typing import Sequence, Optional
import gym
from gym import spaces
from ai_coach_core.models.mdp import LatentMDP
from ai_coach_core.models.agent_model import AgentModel
import numpy as np


class EnvFromLatentMDP(gym.Env):
  # uncomment below line if you need to render the environment
  # metadata = {'render.modes': ['console']}

  def __init__(self,
               mdp: LatentMDP,
               agent_models: Sequence[AgentModel],
               possible_init_states: Optional[Sequence[int]] = None,
               init_state_dist: Optional[np.ndarray] = None):
    '''
    either possible_init_state or init_state_dist should not be None
    '''
    assert possible_init_states or init_state_dist

    super().__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:

    if mdp.num_action_factors == 1:
      self.action_space = spaces.Discrete(mdp.num_actions)
    else:
      self.action_space = spaces.MultiDiscrete(mdp.list_num_actions)

    tuple_num_latent = tuple([
        agent_model.get_reference_mdp().num_latents
        for agent_model in agent_models
    ])

    self.observation_space = spaces.MultiDiscrete(
        (mdp.num_states, *tuple_num_latent))

    self.mdp = mdp
    self.agent_models = agent_models
    self.possible_init_states = possible_init_states
    self.init_state_dist = init_state_dist

    if possible_init_states:
      self.sample_obs = lambda: int(np.random.choice(self.possible_init_states))
    else:
      self.sample_obs = lambda: int(
          np.random.choice(mdp.num_states, p=self.init_state_dist))

    self.reset()

  def step(self, action):
    info = {}

    action_idx = (action if len(self.mdp.list_num_actions) == 1 else
                  self.mdp.conv_action_to_idx(tuple(action)))

    if action_idx not in self.mdp.legal_actions(self.cur_obstate):
      info["invalid_transition"] = True
      return (self.cur_obstate, *self.cur_lstates), 0, False, info

    next_obstate = self.mdp.transition(self.cur_obstate, action_idx)
    done = self.mdp.is_terminal(next_obstate)
    if not done:
      list_lstates = []
      for agent_model in self.agent_models:
        action = (action, ) if len(self.mdp.list_num_actions) == 1 else action
        agent_model.update_mental_state_idx(self.cur_obstate, action,
                                            next_obstate)
        list_lstates.append(agent_model.current_latent)

      self.cur_lstates = tuple(list_lstates)

    self.cur_obstate = next_obstate
    reward = -1  # we don't need reward for imitation learning

    return (self.cur_obstate, *self.cur_lstates), reward, done, info

  def reset(self):
    self.cur_obstate = self.sample_obs()

    list_lstates = []
    for agent_model in self.agent_models:
      agent_model.set_init_mental_state_idx(self.cur_obstate)
      list_lstates.append(agent_model.current_latent)

    self.cur_lstates = tuple(list_lstates)

    return (self.cur_obstate, *self.cur_lstates)
