from typing import Optional, Tuple, Sequence, Callable
import numpy as np
import random

T_StateJointActionSeqence = Sequence[Tuple[int, Tuple[int, ...]]]


def bayesian_mind_inference_for_individual(
    trajectory: T_StateJointActionSeqence,
    mind_idx: int,
    num_latent: int,
    cb_n_sxa_policy: Callable[[int, int, int, Tuple[int, ...]], float],
    # sxa_policy: np.ndarray,
    np_prior: Optional[np.ndarray] = None):
  '''
  trajectory - consists of s, (a1, a2, ..., an), where n is the number of actors
  list_sxa_policy - list of n policies
  list_actor_idx - list of actor indices that belong to one mind
  '''

  # _, num_latent, _ = sxa_policy.shape

  np_px = np.ones((num_latent, ))
  if np_prior is None:
    np_prior = np.full((num_latent, ), 1.0 / num_latent)

  np_log_prior = np.log(np_prior)
  np_log_px = np.zeros((num_latent, ))
  if len(trajectory) < 1:
    print("Empty trajectory")
    return None

  for xidx in range(num_latent):
    for state_idx, joint_action in trajectory:
      p_a_sx = cb_n_sxa_policy(mind_idx, state_idx, xidx, joint_action)
      # p_a_sx = sxa_policy[state_idx][xidx][aidx]

      np_px[xidx] *= p_a_sx
      np_log_px[xidx] += np.log(p_a_sx)

  np_px = np_px * np_prior
  np_log_px = np_log_px + np_log_prior

  list_same_idx = np.argwhere(np_log_px == np.max(np_log_px))
  return random.choice(list_same_idx)[0]


def bayesian_mind_inference(
    trajectory: T_StateJointActionSeqence,
    tuple_num_latent: Tuple[int, ...],
    cb_n_sxa_policy: Callable[[int, int, int, Tuple[int, ...]], float],
    # list_sxa_policy: Sequence[np.ndarray],
    num_minds: int,
    list_np_prior: Optional[Sequence[np.ndarray]] = None):
  '''
  num_minds - The number of agents' minds we need to infer.
  '''

  list_inferred_x = []
  for mind_idx in range(num_minds):
    if list_np_prior is None:
      np_prior = None
    else:
      np_prior = list_np_prior[mind_idx]

    # inferred_x = bayesian_latent_inference_for_each_agent(
    #     trajectory, agent_idx, list_sxa_policy[agent_idx], np_prior)
    inferred_x = bayesian_mind_inference_for_individual(
        trajectory, mind_idx, tuple_num_latent[mind_idx], cb_n_sxa_policy,
        np_prior)
    list_inferred_x.append(inferred_x)

  return tuple(list_inferred_x)


class BayesianLatentInference:
  def __init__(self, true_environment):
    self.env = true_environment

    self.map_brain_2_agents = {i: [] for i in range(self.env.num_brains)}
    for agent_idx in self.env.map_agent_2_brain:
      brain_idx = self.env.map_agent_2_brain[agent_idx]
      self.map_brain_2_agents[brain_idx].append(agent_idx)

  def infer_latentstates(self, trajectory):
    pass


class BayesianLatentInference1(BayesianLatentInference):
  def __init__(self, true_environment):
    super().__init__(true_environment)

  def infer_latentstates(self, trajectory):
    list_latstates = []
    for i_b in range(self.env.num_brains):
      lat = self.infer_inidivial_latentstate(trajectory, i_b)
      list_latstates.append(lat)

    return tuple(list_latstates)

  def infer_inidivial_latentstate(self, trajectory, i_brain):
    xstate_indices = self.env.policy.get_possible_latstate_indices()
    dict_xidx_idx = {}
    for idx, xidx in enumerate(xstate_indices):
      dict_xidx_idx[xidx] = idx

    num_xstate = len(xstate_indices)
    np_px = np.ones((num_xstate, ))
    np_prior = None
    for state_idx, action_idx in trajectory:
      vector_priors = self.env.get_latentstate_prior(state_idx)
      if vector_priors[i_brain] is not None:
        np_p_latent = vector_priors[i_brain]
        np_prior = np.zeros((num_xstate, ))
        for p_x, x_idx in np_p_latent:
          if p_x == 1.0:
            return x_idx.astype(np.int32)
          np_prior[dict_xidx_idx[x_idx]] = p_x
        break

    if np_prior is None:
      # return np.random.choice(xstate_indices)
      # if prior is not set, just return None
      print("no latent state prior")
      return None

    np_log_prior = np.log(np_prior)
    np_log_px = np.zeros((num_xstate, ))
    if len(trajectory) < 1:
      print("Empty trajectory")
      return None

    for idx in range(num_xstate):
      for state_idx, action_idx in trajectory:
        if action_idx < 0:
          break

        p_a_sx = self.compute_p_a_sx(action_idx, state_idx, xstate_indices[idx],
                                     i_brain)
        np_px[idx] = np_px[idx] * p_a_sx
        np_log_px[idx] += np.log([p_a_sx])[0]

    # print("np_prior_indv")
    # print(np_prior)
    # print("np_px_indv")
    # print(np_px)
    # print("np_log_prior")
    # print(np_log_prior)
    # print("np_log_px")
    # print(np_log_px)
    np_px = np_px * np_prior
    np_log_px = np_log_px + np_log_prior
    # sum_p = np.sum(np_px)
    # if sum_p == 0.0:
    #     print("sum of probability is 0")
    #     return None

    # np_px = np_px / sum_p
    # index = np.argmax(np_px)
    index = np.argmax(np_log_px)
    return xstate_indices[index]

  def compute_p_a_sx(self, action_idx, state_idx, x_idx, i_brain):
    'compute p(a|s,x) of agent i'
    if action_idx < 0:
      return 1.

    np_idx_to_action = self.env.mmdp.np_idx_to_action  # alias
    act_cur = np_idx_to_action[action_idx]

    p_a_sx = 0.0
    np_p_action = self.env.policy.pi(state_idx, x_idx)
    for p_act, act_i in np_p_action:
      act_tmp = np_idx_to_action[act_i.astype(np.int32)]
      is_same = True
      for i_agent in self.map_brain_2_agents[i_brain]:
        if act_cur[i_agent] != act_tmp[i_agent]:
          is_same = False
          break
      if is_same:
        p_a_sx += p_act

    return p_a_sx


class BayesianLatentInference2(BayesianLatentInference):
  def __init__(self, true_environment):
    super().__init__(true_environment)

  def infer_latentstates(self, trajectory):
    xstate_indices = self.env.policy.get_possible_latstate_indices()
    dict_xidx_idx = {}
    for idx, xidx in enumerate(xstate_indices):
      dict_xidx_idx[xidx] = idx
    num_xstate = len(xstate_indices)

    def get_possible_combinations(num_brains, list_lats=[]):
      if num_brains == 1:
        list_combinations = []
        for xidx in xstate_indices:
          list_combinations.append(tuple(list_lats + [xidx]))
        return list_combinations
      else:
        list_combinations = []
        for xidx in xstate_indices:
          inter_combi = get_possible_combinations(num_brains - 1,
                                                  list_lats + [xidx])
          list_combinations = list_combinations + inter_combi
        return list_combinations

    list_combinations = get_possible_combinations(self.env.num_brains)
    num_combinations = len(list_combinations)
    # print(num_combinations)
    # print(list_combinations)

    np_prior_team = None
    for state_idx, action_idx in trajectory:
      vector_priors = self.env.get_latentstate_prior(state_idx)
      if vector_priors[0] is not None:
        list_np_priors = []
        for i_brain in range(len(vector_priors)):
          np_p_latent = vector_priors[i_brain]
          np_prior_indv = np.zeros((num_xstate, ))
          for p_x, xidx in np_p_latent:
            np_prior_indv[dict_xidx_idx[xidx]] = p_x
          list_np_priors.append(np_prior_indv)

        np_prior_team = np.ones((num_combinations, ))
        for idx, combo in enumerate(list_combinations):
          for bidx, xidx in enumerate(combo):
            np_prior_team[idx] = (np_prior_team[idx] *
                                  list_np_priors[bidx][dict_xidx_idx[xidx]])
        break

    if np_prior_team is None:
      # if prior is not set, just return None
      print("no latent state prior")
      return tuple([None for dummy_i in range(self.env.num_brains)])

    np_log_prior_team = np.log(np_prior_team)
    np_log_px_team = np.zeros((num_combinations, ))
    np_px = np.ones((num_combinations, ))
    for idx in range(num_combinations):
      for state_idx, action_idx in trajectory:
        if action_idx < 0:
          break

        p_a_sx = self.compute_p_a_sx_team(action_idx, state_idx,
                                          list_combinations[idx])
        np_px[idx] = np_px[idx] * p_a_sx
        np_log_px_team[idx] += np.log([p_a_sx])[0]

    # print("np_prior_team")
    # print(np_prior_team)
    # print("np_px_team")
    # print(np_px)
    # print("np_log_prior_team")
    # print(np_log_prior_team)
    # print("np_log_px_team")
    # print(np_log_px_team)
    np_px = np_px * np_prior_team
    np_log_px_team = np_log_px_team + np_log_prior_team
    # sum_p = np.sum(np_px)
    # if sum_p == 0.0:
    #     print("sum of probability is 0")
    #     return tuple([None for dummy_i in range(self.env.num_brains)])

    # np_px = np_px / sum_p
    # index = np.argmax(np_px)
    index = np.argmax(np_log_px_team)
    return list_combinations[index]

  def compute_p_a_sx_team(self, action_idx, state_idx, latent_combo):
    'compute p(a|s,x)'
    if action_idx < 0:
      return 1.

    np_idx_to_action = self.env.mmdp.np_idx_to_action  # alias
    act_cur = np_idx_to_action[action_idx]

    p_a_sx_team = 1.0
    for bidx, xidx in enumerate(latent_combo):
      np_p_action = self.env.policy.pi(state_idx, xidx)
      p_a_sx = 0.0
      for p_act, act_i in np_p_action:
        act_tmp = np_idx_to_action[act_i.astype(np.int32)]
        is_same = True
        for i_agent in self.map_brain_2_agents[bidx]:
          if act_cur[i_agent] != act_tmp[i_agent]:
            is_same = False
            break
        if is_same:
          p_a_sx += p_act
      p_a_sx_team *= p_a_sx

    return p_a_sx_team
