import abc
import os
import numpy as np
from ai_coach_core.utils.exceptions import NoPolicyError


class MMDPPolicy:
  # Python 2 style but Python3 is also compatible with this.
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    pass

  @abc.abstractmethod
  def pi(self, obstate_idx, latstate_idx):
    '''
        returns the distribution of actions as the numpy array where
        the 1st column is the probability and the 2nd column is the action
        '''
    pass

  @abc.abstractmethod
  def get_possible_latstate_indices(self):
    pass


class Environment:
  def __init__(self, mmdp, policy, num_brains):
    self.mmdp = mmdp
    self.policy = policy
    self.num_brains = num_brains
    self.map_agent_2_brain = {i: i for i in range(num_brains)}

  @abc.abstractmethod
  def get_initial_state_dist(self):
    pass

  @abc.abstractmethod
  def is_terminal_state(self, obstate_idx):
    pass

  @abc.abstractmethod
  def get_latentstate(self, obstate_idx):
    '''
        return None or the tuple of each agent's latent states
        None means latent stats should not be set yet.
        '''
    pass

  @abc.abstractmethod
  def get_latentstate_prior(self, obstate_idx):
    pass

  def generate_sequence(self,
                        start_state_idx,
                        timeout,
                        save=False,
                        file_name='sequence.txt'):
    '''
        start_state_idx: observable state
        '''
    if self.policy is None:
      raise NoPolicyError

    # print(self.mmdp.num_agents)
    # print(len(tuple_latstates))
    list_sequence = []
    obstate_idx = start_state_idx
    list_lat_states = [None for dummy_i in range(self.num_brains)]
    for dummy_i in range(timeout):
      if self.is_terminal_state(obstate_idx):
        break

      # check if latent state need to be set
      if list_lat_states[0] is None:
        lat_states = self.get_latentstate(obstate_idx)
        if lat_states is not None:
          list_lat_states = list(lat_states)

      each_agent_action = []
      for i_a in range(self.mmdp.num_agents):
        brain_idx = self.map_agent_2_brain[i_a]
        np_p_action = self.policy.pi(obstate_idx, list_lat_states[brain_idx])
        action_choice = np.random.choice(np_p_action[:, 1],
                                         1,
                                         p=np_p_action[:, 0])
        action_vector = self.mmdp.np_idx_to_action[action_choice[0].astype(
            np.int32)]
        each_agent_action.append(action_vector[i_a])

      action_idx = self.mmdp.np_action_to_idx[tuple(each_agent_action)]
      list_sequence.append((obstate_idx, action_idx))
      # self.print_obstate(obstate_idx)
      # self.print_action(action_idx)

      np_next_p_state_idx = self.mmdp.transition_model(obstate_idx, action_idx)
      state_choice = np.random.choice(np_next_p_state_idx[:, 1],
                                      1,
                                      p=np_next_p_state_idx[:, 0])
      obstate_idx = state_choice[0].astype(np.int32)

    list_sequence.append((obstate_idx, -1))  # -1 indicates the end of task

    if save:
      dir_path = os.path.dirname(file_name)
      if dir_path != '' and not os.path.exists(dir_path):
        os.makedirs(dir_path)

      with open(file_name, 'w', newline='') as txtfile:
        # header
        txtfile.write('# latent states\n')
        txtfile.write(', '.join(str(i) for i in list_lat_states))
        txtfile.write('\n')
        # sequence
        txtfile.write('# state action sequence\n')
        for state_idx, action_idx in list_sequence:
          txtfile.write('%d, %d' % (state_idx, action_idx))
          txtfile.write('\n')

    return list_sequence, list_lat_states

  def print_obstate(self, obstate_idx):
    state_vector = self.mmdp.np_idx_to_state[obstate_idx]
    num_factors = len(state_vector)
    list_state = []
    for i_f in range(num_factors):
      statespace = self.mmdp.dict_factored_statespace[i_f]
      list_state.append(statespace.idx_to_state[state_vector[i_f]])

    print(*list_state, sep='; ')

  def print_action(self, action_idx):
    if action_idx < 0:
      print("None")
      return
    action_vector = self.mmdp.np_idx_to_action[action_idx]
    num_factors = len(action_vector)
    list_action = []
    for i_f in range(num_factors):
      actionspace = self.mmdp.dict_factored_actionspace[i_f]
      list_action.append(actionspace.idx_to_action[action_vector[i_f]])

    print(*list_action, sep='; ')


class RequestEnvironment(Environment):
  def __init__(self, mmdp, policy, num_brains):
    super().__init__(mmdp, policy, num_brains)

  @abc.abstractmethod
  def is_initiated_state(self, state_idx):
    pass
