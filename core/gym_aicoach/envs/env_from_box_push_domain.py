from typing import Sequence, Optional
import gym
from gym import spaces
from ai_coach_domain.box_push.mdp import BoxPushMDP
from ai_coach_domain.box_push.agent import BoxPushAIAgent_Abstract
import numpy as np


class EnvFromBoxPushDomain(gym.Env):
  # uncomment below line if you need to render the environment
  # metadata = {'render.modes': ['console']}

  def __init__(self,
               mdp: BoxPushMDP,
               agents: Sequence[BoxPushAIAgent_Abstract],
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

    tuple_num_latent = tuple(
        [agent.agent_model.get_reference_mdp().num_latents for agent in agents])

    self.observation_space = spaces.MultiDiscrete(
        (mdp.num_states, *tuple_num_latent))

    self.mdp = mdp
    self.agents = agents
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

    cur_sim_state = self.mdp.conv_mdp_sidx_to_sim_states(self.cur_obstate)

    self.cur_obstate = self.mdp.transition(self.cur_obstate, action_idx)
    done = self.mdp.is_terminal(self.cur_obstate)
    if not done:
      nxt_sim_state = self.mdp.conv_mdp_sidx_to_sim_states(self.cur_obstate)

      sim_action = self.mdp.conv_mdp_aidx_to_sim_actions(action_idx)

      list_lstates = []
      for agent in self.agents:
        agent.update_mental_state(cur_sim_state, sim_action, nxt_sim_state)
        list_lstates.append(agent.agent_model.current_latent)

      self.cur_lstates = tuple(list_lstates)

    reward = -1  # we don't need reward for imitation learning

    return (self.cur_obstate, *self.cur_lstates), reward, done, info

  def reset(self):
    self.cur_obstate = self.sample_obs()
    tup_states = self.mdp.conv_mdp_sidx_to_sim_states(self.cur_obstate)

    list_lstates = []
    for agent in self.agents:
      agent.init_latent(tup_states)
      list_lstates.append(agent.agent_model.current_latent)

    self.cur_lstates = tuple(list_lstates)

    return (self.cur_obstate, *self.cur_lstates)

  # implement render function if need to be
  # def render(self, mode='human'):
  #   ...

  # implement close function if need to be
  # def close(self):
  #   ...
