from typing import Optional, Tuple, Sequence, Callable
import numpy as np
import random

T_StateJointActionSeqence = Sequence[Tuple[int, Tuple[int, ...]]]


def bayesian_mind_inference_for_individual(
    trajectory: T_StateJointActionSeqence,
    mind_idx: int,
    num_latent: int,
    cb_n_xsa_policy: Callable[[int, int, int, Tuple[int, ...]], float],
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
      p_a_sx = cb_n_xsa_policy(mind_idx, xidx, state_idx, joint_action)
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
    cb_n_xsa_policy: Callable[[int, int, int, Tuple[int, ...]], float],
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
        trajectory, mind_idx, tuple_num_latent[mind_idx], cb_n_xsa_policy,
        np_prior)
    list_inferred_x.append(inferred_x)

  return tuple(list_inferred_x)
