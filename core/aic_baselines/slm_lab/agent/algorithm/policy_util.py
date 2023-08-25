# Action policy module
# Constructs action probability distribution used by agent to sample action and calculate log_prob, entropy, etc.
from gym import spaces
from ...lib import distribution
from torch import distributions
import numpy as np
import torch

# register custom distributions
setattr(distributions, 'Argmax', distribution.Argmax)
setattr(distributions, 'GumbelSoftmax', distribution.GumbelSoftmax)
setattr(distributions, 'MultiCategorical', distribution.MultiCategorical)
# probability distributions constraints for different action types; the first in the list is the default
ACTION_PDS = {
    'continuous': ['Normal', 'Beta', 'Gumbel', 'LogNormal'],
    'multi_continuous': ['MultivariateNormal'],
    'discrete': ['Categorical', 'Argmax', 'GumbelSoftmax'],
    'multi_discrete': ['MultiCategorical'],
    'multi_binary': ['Bernoulli'],
}


def get_action_type(action_space):
  '''Method to get the action type to choose prob. dist. to sample actions from NN logits output'''
  if isinstance(action_space, spaces.Box):
    shape = action_space.shape
    assert len(shape) == 1
    if shape[0] == 1:
      return 'continuous'
    else:
      return 'multi_continuous'
  elif isinstance(action_space, spaces.Discrete):
    return 'discrete'
  elif isinstance(action_space, spaces.MultiDiscrete):
    return 'multi_discrete'
  elif isinstance(action_space, spaces.MultiBinary):
    return 'multi_binary'
  else:
    raise NotImplementedError


# action_policy base methods


def get_action_pd_cls(action_pdtype, action_type):
  '''
    Verify and get the action prob. distribution class for construction
    Called by body at init to set its own ActionPD
    '''
  pdtypes = ACTION_PDS[action_type]
  assert action_pdtype in pdtypes, f'Pdtype {action_pdtype} is not compatible/supported with action_type {action_type}. Options are: {pdtypes}'
  ActionPD = getattr(distributions, action_pdtype)
  return ActionPD


def guard_tensor(state):
  '''Guard-cast tensor before being input to network'''
  state = torch.from_numpy(state.astype(np.float32))
  return state


def calc_pdparam(state, algorithm):
  '''
    Prepare the state and run algorithm.calc_pdparam to get pdparam for action_pd
    @param tensor:state For pdparam = net(state)
    @param algorithm The algorithm containing self.net
    @param body Body which links algorithm to the env which the action is for
    @returns tensor:pdparam
    @example

    pdparam = calc_pdparam(state, algorithm, body)
    action_pd = ActionPD(logits=pdparam)  # e.g. ActionPD is Categorical
    action = action_pd.sample()
    '''
  if not torch.is_tensor(state):  # dont need to cast from numpy
    state = guard_tensor(state)
    state = state.to(algorithm.net.device)
  pdparam = algorithm.calc_pdparam(state)
  return pdparam


def init_action_pd(ActionPD, pdparam):
  '''
    Initialize the action_pd for discrete or continuous actions:
    - discrete: action_pd = ActionPD(logits)
    - continuous: action_pd = ActionPD(loc, scale)
    '''
  args = ActionPD.arg_constraints
  if 'logits' in args:  # discrete
    # for relaxed discrete dist. with reparametrizable discrete actions
    pd_kwargs = {
        'temperature': torch.tensor(1.0)
    } if hasattr(ActionPD, 'temperature') else {}
    action_pd = ActionPD(logits=pdparam, **pd_kwargs)
  else:  # continuous, args = loc and scale
    if isinstance(pdparam, list):  # split output
      loc, scale = pdparam
    else:
      loc, scale = pdparam.transpose(0, 1)
    # scale (stdev) must be > 0, log-clamp-exp
    scale = torch.clamp(scale, min=-20, max=2).exp()
    if 'covariance_matrix' in args:  # split output
      # construct covars from a batched scale tensor
      covars = torch.diag_embed(scale)
      action_pd = ActionPD(loc=loc, covariance_matrix=covars)
    else:
      action_pd = ActionPD(loc=loc, scale=scale)
  return action_pd


def sample_action(ActionPD, pdparam):
  '''
    Convenience method to sample action(s) from action_pd = ActionPD(pdparam)
    Works with batched pdparam too
    @returns tensor:action Sampled action(s)
    @example

    # policy contains:
    pdparam = calc_pdparam(state, algorithm, body)
    action = sample_action(body.ActionPD, pdparam)
    '''
  action_pd = init_action_pd(ActionPD, pdparam)
  action = action_pd.sample()
  return action


# action_policy used by agent


def default(state, algorithm, ActionPD):
  '''Plain policy by direct sampling from a default action probability defined by body.ActionPD'''
  pdparam = calc_pdparam(state, algorithm)
  action = sample_action(ActionPD, pdparam)
  return action


def random(state, env):
  '''Random action using gym.action_space.sample(), with the same format as default()'''
  _action = [env.action_space.sample()]
  action = torch.tensor(_action)
  return action


# multi-body/multi-env action_policy used by agent
# TODO rework


def multi_default(states, algorithm, body_list, pdparam):
  '''
    Apply default policy body-wise
    Note, for efficiency, do a single forward pass to calculate pdparam, then call this policy like:
    @example

    pdparam = self.calc_pdparam(state)
    action_a = self.action_policy(pdparam, self, body_list)
    '''
  # assert pdparam has been chunked
  assert pdparam.dim() > 1 and len(pdparam) == len(
      body_list), f'pdparam shape: {pdparam.shape}, bodies: {len(body_list)}'
  action_list = []
  for idx, sub_pdparam in enumerate(pdparam):
    body = body_list[idx]
    guard_tensor(states[idx],
                 body)  # for consistency with singleton inner logic
    action = sample_action(body.ActionPD, sub_pdparam)
    action_list.append(action)
  action_a = torch.tensor(action_list,
                          device=algorithm.net.device).unsqueeze(dim=1)
  return action_a


def multi_random(states, algorithm, body_list, pdparam):
  '''Apply random policy body-wise.'''
  action_list = []
  for idx, body in body_list:
    action = random(states[idx], algorithm, body)
    action_list.append(action)
  action_a = torch.tensor(action_list,
                          device=algorithm.net.device).unsqueeze(dim=1)
  return action_a


def multi_epsilon_greedy(states, algorithm, body_list, pdparam):
  '''Apply epsilon-greedy policy body-wise'''
  assert len(pdparam) > 1 and len(pdparam) == len(
      body_list), f'pdparam shape: {pdparam.shape}, bodies: {len(body_list)}'
  action_list = []
  for idx, sub_pdparam in enumerate(pdparam):
    body = body_list[idx]
    epsilon = body.explore_var
    if epsilon > np.random.rand():
      action = random(states[idx], algorithm, body)
    else:
      guard_tensor(states[idx],
                   body)  # for consistency with singleton inner logic
      action = sample_action(body.ActionPD, sub_pdparam)
    action_list.append(action)
  action_a = torch.tensor(action_list,
                          device=algorithm.net.device).unsqueeze(dim=1)
  return action_a


def multi_boltzmann(states, algorithm, body_list, pdparam):
  '''Apply Boltzmann policy body-wise'''
  assert len(pdparam) > 1 and len(pdparam) == len(
      body_list), f'pdparam shape: {pdparam.shape}, bodies: {len(body_list)}'
  action_list = []
  for idx, sub_pdparam in enumerate(pdparam):
    body = body_list[idx]
    guard_tensor(states[idx],
                 body)  # for consistency with singleton inner logic
    tau = body.explore_var
    sub_pdparam /= tau
    action = sample_action(body.ActionPD, sub_pdparam)
    action_list.append(action)
  action_a = torch.tensor(action_list,
                          device=algorithm.net.device).unsqueeze(dim=1)
  return action_a
