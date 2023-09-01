import numpy as np
from scipy.optimize import linear_sum_assignment
import itertools
from copy import deepcopy
from aic_core.utils.result_utils import compute_kl


def find_best_order(true_latent: list, est_latent: list, num_latent):
  """Find the order of est_latent that best approximates true_latent.

    Uses the Jonker-Volgenant algorithm.
    Params:
        - true_latent (list of numpy.ndarray):
            list of true latent states during demonstrations

        - est_latent (list of numpy.ndarray):
            list of estimated latent states for observed demonstrations

        - num_states (list or np.ndarray):
            number of states accross each feature.
            One dimensional vector of shape D
    
    Output:
        - new_est_latent (list of numpy.ndarray):
            a list of the same dimensions as est_latent, with reordered
            elements to better approximate true_latent
        - order (list of numpy.ndarray):
            the order used to reaccomodate elements in new_est_latent
            for each feature d
    """
  if not isinstance(est_latent, list):
    est_latent = [est_latent]
  if not isinstance(true_latent, list):
    true_latent = [true_latent]
  # Concatenate all demonstrations into one
  est_latent_concat = np.concatenate(est_latent)
  true_latent_concat = np.concatenate(true_latent)
  # n_feats = true_latent_concat.shape[1]
  cost, count = get_cost(true_latent_concat, est_latent_concat, num_latent)

  # orders = []
  # for feat in range(n_feats):
  _, order = linear_sum_assignment(cost)
  # orders.append(order)

  return order, count


def find_best_order_kl(xs_policy_true: list, xs_policy_est: list, weights_xs,
                       num_latent):

  def kl_divergence(policy_true, policy_est, weight):
    # compute_kl
    pass

  count = 0
  cost = np.zeros((num_latent, num_latent))

  for row in range(num_latent):
    for col in range(num_latent):
      count += 1
      cost[row, col] = kl_divergence(xs_policy_true, xs_policy_est, weights_xs)

  _, order = linear_sum_assignment(cost)

  return order, count


def get_cost(list_true_latent, list_est_latent, num_latent):
  """Gets the cost for each possible reordering of latent states.

    Params:
        - true_latent (np.ndarray):
            true latent state sequence of shape N X D
        - est_latent (np.ndarray):
            estimated latent state seq. of shape N X D
        - num_states (np.ndarray):
            number of states accross each feature.
            One dimensional vector of shape D
    
    Output:
        - Matrices of costs (list of np.ndarray):
            One matrix for each feature d.
            Its shape is num_states[d] X num_states[d]
    """
  count = 0
  cost = np.zeros((num_latent, num_latent))

  for row in range(num_latent):
    for col in range(num_latent):
      count += 1
      # print(count)
      reordered_est_states = [
          col if list_est_latent[j] == row else
          row if list_est_latent[j] == col else list_est_latent[j]
          for j in range(len(list_est_latent))
      ]

      cost[row, col] = np.sum(
          np.array(reordered_est_states) != np.array(list_true_latent))

  return cost, count


def order_latent_seq(est_latent, orders):
  est_latent_concat = np.concatenate(est_latent)
  new_est_concat = np.zeros_like(est_latent_concat)
  for feat in range(len(orders)):
    order = orders[feat]
    new_est_concat[:, feat] = [order[i] for i in est_latent_concat[:, feat]]

  # Get a list of same shape and dimensions as est_latent
  new_est = []
  init = 0
  for lat_seq in est_latent:
    end = len(lat_seq) + init
    new_est.append(new_est_concat[init:end])
    init = end
  return new_est


def order_policy(policy, order):
  new_matrix = deepcopy(policy)
  # for j in range(len(order)):
  s = (tuple(order), )
  new_matrix = new_matrix[s]
  return new_matrix
