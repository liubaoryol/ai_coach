from typing import Optional, Tuple, Sequence, Callable
import numpy as np
import random

T_StateJointActionSeqence = Sequence[Tuple[int, Tuple[int, ...]]]

# type for a policy that takes the input: (agent_idx, latent_idx, state_idx,
#                                          tuple of a team's action_idx)
T_CallbackPolicy = Callable[[int, int, int, Tuple[int, ...]], float]


def bayesian_mental_state_inference_for_individual(
    trajectory: T_StateJointActionSeqence,
    agent_idx: int,
    num_latent: int,
    cb_n_xsa_policy: T_CallbackPolicy,
    np_prior: Optional[np.ndarray] = None):
  '''
  trajectory - consists of s, (a1, a2, ..., an), where n is the number of actors
  cb_n_xsa_policy - callback taking agent_idx, latent_idx, state_idx, and
                                    tuple of action_idx as its input
  '''

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
      p_a_sx = cb_n_xsa_policy(agent_idx, xidx, state_idx, joint_action)

      np_px[xidx] *= p_a_sx
      np_log_px[xidx] += np.log(p_a_sx)

  np_px = np_px * np_prior
  np_log_px = np_log_px + np_log_prior

  list_same_idx = np.argwhere(np_log_px == np.max(np_log_px))
  return random.choice(list_same_idx)[0]


def bayesian_mental_state_inference(
    trajectory: T_StateJointActionSeqence,
    tuple_num_latent: Tuple[int, ...],
    cb_n_xsa_policy: T_CallbackPolicy,
    num_agents: int,
    list_np_prior: Optional[Sequence[np.ndarray]] = None):
  '''
  trajectory: trajectory of (state_idx, the tuple of action_idx of a team)
  tuple_num_latent: tuple of the number of possible mental states for each agent
  cb_n_xsa_policy: callback taking agent_idx, latent_idx, state_idx, and
                                  tuple of action_idx as its input
  num_agents: the number of agents that can individually have a mental state
  list_np_prior: list of each agent's prior distribution stored by numpy array
  '''

  list_inferred_x = []
  for agent_idx in range(num_agents):
    if list_np_prior is None:
      np_prior = None
    else:
      np_prior = list_np_prior[agent_idx]

    inferred_x = bayesian_mental_state_inference_for_individual(
        trajectory, agent_idx, tuple_num_latent[agent_idx], cb_n_xsa_policy,
        np_prior)
    list_inferred_x.append(inferred_x)

  return tuple(list_inferred_x)
