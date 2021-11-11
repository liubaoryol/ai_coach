"""Implements planning techniques for solving MDPs.

Provides a domain independent implementation of policy and value iteration
using numpy.
"""

from typing import Optional, Union

import numpy as np
from tqdm import tqdm
import scipy.special as sc

import ai_coach_core.models.mdp as mdp_lib


def softmax_by_row(q_val, temperature=1.):
  """
  Note: softmax problem
    - it continuously adds non-necessary term log(blah) to v-values, even if 
      initial v-value was 0.
      If reward range is small, e.g. [0, 1], this is not neglectible
    - q_val should not include any invalid number (-inf, nan, inf) 
  """
  USE_SCIPY_SOFTMAX = False
  if USE_SCIPY_SOFTMAX:
    # scipy version is slower.
    # but maybe this is numerically more stable as it's widely-used library
    return temperature * sc.logsumexp(q_val, axis=-1, b=(1 / temperature))
  else:
    max_q = np.max(q_val, axis=1)
    sub_q_val = q_val - max_q[:, np.newaxis]

    return max_q + temperature * np.log(
        np.sum(np.exp(sub_q_val / temperature), axis=1))


def mellowmax_by_row(q_val, temperature=1.):
  """
  An alternative for softmax
  Note: not working well
  """
  return (softmax_by_row(q_val, temperature) -
          temperature * np.log(q_val.shape[1]))


def sparsemax_by_row(q_val):
  """
  An alternative for softmax
  TODO: implementation needs to be verified. not test yet.
  """
  sorted_qval = np.flip(np.sort(q_val, axis=-1), axis=-1)

  cumsum = np.cumsum(sorted_qval, axis=-1)
  k = np.arange(1, sorted_qval.shape[1] + 1)
  one_plus_kz = 1 + k[None, :] * sorted_qval
  is_support = one_plus_kz > cumsum
  k_max = np.sum(is_support, axis=-1)

  tau_z = (cumsum[np.arange(q_val.shape[0])[..., None], k_max - 1] - 1) / k_max
  return (q_val - tau_z).clip(0)


def soft_value_iteration(transition_model: Union[np.ndarray,
                                                 mdp_lib.sparse.COO],
                         reward_model: np.ndarray,
                         discount_factor: float = 0.95,
                         max_iteration: int = 20,
                         epsilon: float = 1e-6,
                         temperature: float = 1.,
                         show_progress_bar: bool = True) -> np.ndarray:
  """Soft value iteration"""
  # slightly differnet from the original paper
  # i.e. utilize a discount factor to ensure convergence
  #      initilize with 0s as convergence is guaranteed

  # Vsoft = np.full((mdp.num_states, ), SMALL_NUMBER)
  num_states, num_actions, _ = transition_model.shape
  v_value = np.zeros((num_states, ))
  # v_value = np.full((num_states, ), -10000)
  # v_value[terminal_states] = 0

  iteration_idx = 0
  delta_v = epsilon + 1
  progress_bar = tqdm(total=max_iteration, disable=not show_progress_bar)
  while (iteration_idx < max_iteration) and (delta_v > epsilon):
    q_value = mdp_lib.q_value_from_v_value(v_value, transition_model,
                                           reward_model, discount_factor)
    new_v_value = softmax_by_row(np.nan_to_num(q_value), temperature)
    delta_v = np.linalg.norm(new_v_value[:] - v_value[:])
    iteration_idx += 1
    v_value = new_v_value
    progress_bar.update()
  progress_bar.close()

  return v_value, q_value


def value_iteration(
    transition_model: Union[np.ndarray, mdp_lib.sparse.COO],
    reward_model: np.ndarray,
    discount_factor: float = 0.95,
    max_iteration: int = 20,
    epsilon: float = 1e-6,
    v_value_initial: Optional[np.ndarray] = None,
) -> np.ndarray:
  """Implements the value iteration algorithm.

  Args:
    transition_model: A transition model as a numpy 3-d array.
    reward_model: A reward model as a numpy 2-d array.
    discount_factor: MDP discount factor to be used for policy evaluation.
    max_iteration: Maximum number of iterations for policy evaluation.
    epsilon: Desired v-value threshold. Used for termination condition.
    v_value_initial: Optional. Initial guess for V value.

  Returns:
    A tuple of policy, v_value, and q_value.
  """
  num_states, num_actions, _ = transition_model.shape

  if v_value_initial is not None:
    assert v_value_initial.shape == (num_states, ), (
        "Initial V value has incorrect shape.")
    v_value = v_value_initial
  else:
    v_value = np.zeros((num_states))

  iteration_idx = 0
  delta_v = epsilon + 1.
  progress_bar = tqdm(total=max_iteration)
  while (iteration_idx < max_iteration) and (delta_v > epsilon):
    q_value = mdp_lib.q_value_from_v_value(v_value, transition_model,
                                           reward_model, discount_factor)
    new_v_value = q_value.max(axis=-1)
    delta_v = np.linalg.norm(new_v_value[:] - v_value[:])
    iteration_idx += 1
    v_value = new_v_value
    progress_bar.set_postfix({'delta': delta_v})
    progress_bar.update()
  progress_bar.close()

  policy = mdp_lib.deterministic_policy_from_q_value(q_value)

  return (policy, v_value, q_value)


def policy_iteration(
    transition_model: Union[np.ndarray, mdp_lib.sparse.COO],
    reward_model: np.ndarray,
    discount_factor: float = 0.95,
    max_iteration: int = 20,
    epsilon: float = 1e-6,
    policy_initial: Optional[np.ndarray] = None,
    v_value_initial: Optional[np.ndarray] = None,
):
  """Implements the policy iteration algorithm.

  Args:
    transition_model: A transition model as a numpy 3-d array.
    reward_model: A reward model as a numpy 2-d array.
    discount_factor: MDP discount factor to be used for policy evaluation.
    max_iteration: Maximum number of iterations for policy evaluation.
    epsilon: Desired v-value threshold. Used for termination condition.
    policy_initial: Optional. A deterministic policy.
    v_value_initial: Optional. Initial guess for V value.

  Returns:
    A tuple of policy, v_value, and q_value.
  """
  num_states, num_actions, _ = transition_model.shape

  if policy_initial is not None:
    policy = policy_initial
  else:
    policy = np.zeros(num_states, dtype=int)

  v_value = v_value_initial

  iteration_idx = 0
  delta_policy = 1
  progress_bar = tqdm(total=max_iteration)
  while (iteration_idx < max_iteration) and (delta_policy > 0):
    v_value = mdp_lib.v_value_from_policy(
        policy=policy,
        transition_model=transition_model,
        reward_model=reward_model,
        discount_factor=discount_factor,
        epsilon=epsilon,
        v_value_initial=v_value,
    )
    q_value = mdp_lib.q_value_from_v_value(
        v_value=v_value,
        transition_model=transition_model,
        reward_model=reward_model,
        discount_factor=discount_factor,
    )

    new_policy = mdp_lib.deterministic_policy_from_q_value(q_value)
    delta_policy = (policy != new_policy).sum()
    policy = new_policy
    iteration_idx += 1
    progress_bar.update()
  progress_bar.close()

  return (policy, v_value, q_value)
