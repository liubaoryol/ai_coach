import numpy as np


def sparsemax_by_row(q_val: np.ndarray):
  """
  TODO: implementation needs to be verified. not test yet.
  note: just by the name itself, initially i thought this can be a alternative
        for softmax in MaxEntIrl. however, this computes probabilities, not max.
        It also seems not possible to be used to compute a stochastic policy
        because this only works with values > -1, while all q-values can be < -1
  """
  sorted_qval = np.flip(np.sort(q_val, axis=-1), axis=-1)

  cumsum = np.cumsum(sorted_qval, axis=-1)
  k = np.arange(1, sorted_qval.shape[1] + 1)
  one_plus_kz = 1 + k[None, :] * sorted_qval
  is_support = one_plus_kz > cumsum
  k_max = np.sum(is_support, axis=-1)

  tau_z = (cumsum[np.arange(q_val.shape[0])[..., None], k_max - 1] - 1) / k_max
  return (q_val - tau_z).clip(0)
