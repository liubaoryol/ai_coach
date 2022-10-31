from abc import abstractmethod, ABC
from enum import Enum
from typing import Sequence, Callable
import itertools
import numpy as np
from ai_coach_core.utils.result_utils import compute_js


class E_ComboSelection(Enum):
  Value = 0  # need to specify whether the full change is allowed or not
  Acceptibility = 1  # pick one that is likely to be accepted by the team
  LeastChange = 2  # pick one that changes the least number of members
  Confidence = 3


class E_CertaintyHandling(Enum):
  Threshold = 0
  Average = 1


def get_combos_sorted_by_simulated_values(np_v_values, obstate_idx):
  np_obstate_values = np_v_values[obstate_idx]
  list_combos = []
  for combos, value in np.ndenumerate(np_obstate_values):
    list_combos.append((value, combos))

  list_combos.sort(reverse=True)

  return list_combos


class InterventionAbstract(ABC):

  def __init__(
      self,
      e_combo_selection: E_ComboSelection,
      e_inference_certainty: E_CertaintyHandling,
      inference_threshold: float,  # theta
      intervention_threshold: float,  # delta
      intervention_cost: float = 0.0):
    self.e_combo_selection = e_combo_selection
    self.e_inference_certainty = e_inference_certainty

    self.intervention_threshold = intervention_threshold
    self.intervention_cost = intervention_cost
    self.inference_threshold = inference_threshold

  @abstractmethod
  def get_intervention(self, list_np_latent_distribution: Sequence[np.ndarray],
                       obstate_idx: int):
    '''
    return: a map from an agent index to the tuple of
          (inference_probability, inferred_mental_models, desired_mental_model)
    '''
    pass


class InterventionRuleBased(InterventionAbstract):

  def __init__(self,
               cb_get_valid_latent: Callable,
               num_agent: int,
               e_inference_certainty: E_CertaintyHandling,
               inference_threshold: float,
               intervention_threshold: float,
               intervention_cost: float = 0):
    super().__init__(e_combo_selection=E_ComboSelection.Confidence,
                     e_inference_certainty=e_inference_certainty,
                     inference_threshold=inference_threshold,
                     intervention_threshold=intervention_threshold,
                     intervention_cost=intervention_cost)
    self.cb_get_valid_latent = cb_get_valid_latent
    self.num_agent = num_agent

  def get_compatible_latent_combos(self, obstate_idx):
    'no valid'
    valid_latent = self.cb_get_valid_latent(obstate_idx)
    list_valid_combo_pairs = []
    for lat in valid_latent:
      list_valid_combo_pairs.append(tuple([lat] * self.num_agent))

    return list_valid_combo_pairs

  def _confidence_based_combo(self, list_valid_combos,
                              list_np_latent_distribution):
    max_prob = -1
    max_idx = -1
    if len(list_valid_combos) == 0:
      return None

    for idx, combo in enumerate(list_valid_combos):
      for nidx, lat in enumerate(combo):
        prob = list_np_latent_distribution[nidx][lat]
        if max_prob < prob:
          max_prob = prob
          max_idx = idx

    return list_valid_combos[max_idx]

  def deterministic_decision(self, list_valid_combos, est_combo):
    if len(list_valid_combos) == 0:
      return False

    return not (est_combo in list_valid_combos)

  def get_argmax(self, list_np_latent_distribution):
    # get argmax
    list_p = []
    list_argmax = []
    for np_dist in list_np_latent_distribution:
      max_p = np_dist.max()
      max_indices = np.where(np_dist == max_p)[0]
      list_p.append(max_p)
      list_argmax.append(max_indices)

    return list_argmax, list_p

  def get_intervention(self, list_np_latent_distribution: Sequence[np.ndarray],
                       obstate_idx: int):
    '''
    return: a map from an agent index to the tuple of
          (inference_probability, inferred_mental_models, desired_mental_model)
    '''
    list_valid_combos = self.get_compatible_latent_combos(obstate_idx)

    picked_combo = None
    # find the optimal combination that satisfies the method and criteria
    if self.e_combo_selection == E_ComboSelection.Confidence:
      picked_combo = self._confidence_based_combo(list_valid_combos,
                                                  list_np_latent_distribution)

    else:
      raise NotImplementedError

    if picked_combo is None:
      return None

    # whether to intervene or not, based on value threshold
    # deterministic approach assumes the inferred combination is argmax of
    # each distribution
    do_intervention = False
    if self.e_inference_certainty == E_CertaintyHandling.Average:
      decision_average = 0
      # all possible combinations
      list_range_latent = [
          range(len(np_dist)) for np_dist in list_np_latent_distribution
      ]
      for combo in itertools.product(*list_range_latent):
        prob = 1.0
        for idx, lat in enumerate(combo):
          prob *= list_np_latent_distribution[idx][lat]

        decision_average += prob * int(
            self.deterministic_decision(list_valid_combos, combo))

      if decision_average > 0.5:
        do_intervention = True
    elif self.e_inference_certainty == E_CertaintyHandling.Threshold:
      list_argmax, list_p = self.get_argmax(list_np_latent_distribution)

      p_all = 1.0
      for prob in list_p:
        p_all *= prob

      if p_all >= self.inference_threshold:
        decision_average = 0
        count = 0
        for combo_cur in itertools.product(*list_argmax):
          decision_average += int(
              self.deterministic_decision(list_valid_combos, combo_cur))
          count += 1

        decision_average = decision_average / count
        if decision_average > 0.5:
          do_intervention = True
    else:
      raise NotImplementedError

    # check if inferred combination is in the set of compatible combinations
    # if not in, get the agents whose mental model needs to be updated
    if do_intervention:
      dict_intervention_p_x_inf_x_int = {}
      for idx, lat in enumerate(picked_combo):
        # if lat not in list_argmax[idx]:
        dict_intervention_p_x_inf_x_int[idx] = lat

      return dict_intervention_p_x_inf_x_int

    return None


class InterventionValueBased(InterventionAbstract):

  def __init__(
      self,
      np_v_values: np.ndarray,
      e_inference_certainty: E_CertaintyHandling,
      inference_threshold: float,  # theta
      intervention_threshold: float,  # delta
      intervention_cost: float = 0.0,
      compatibility_tolerance: float = 0.0,
      argmax_cur_val_by_average=False) -> None:
    super().__init__(e_combo_selection=E_ComboSelection.Value,
                     e_inference_certainty=e_inference_certainty,
                     inference_threshold=inference_threshold,
                     intervention_threshold=intervention_threshold,
                     intervention_cost=intervention_cost)
    self.np_v_values = np_v_values
    self.argmax_cur_val_by_average = argmax_cur_val_by_average

    # NOTE: there could be multiple ways to define compatible combinations
    #       with v-values
    # e.g. top K ranks, percentile of ranks,
    #      tolerance from the optimal, thresholding after normalizing the range
    self.compatibility_tolerance = compatibility_tolerance

  def get_compatible_latent_combos(self, obstate_idx):
    list_vval_combo_pairs = get_combos_sorted_by_simulated_values(
        self.np_v_values, obstate_idx)

    optimal_val = list_vval_combo_pairs[0][0]
    criteria_val = optimal_val - optimal_val * self.compatibility_tolerance
    num_combos = 0
    for idx in range(len(list_vval_combo_pairs)):
      if list_vval_combo_pairs[idx][0] > criteria_val:
        num_combos = idx
        break
    return list_vval_combo_pairs[:num_combos]

  def get_vval_of_combo(self, input_combo, list_compatible_vval_combos):
    vval = None
    for val, combo in list_compatible_vval_combos:
      if input_combo == combo:
        vval = val
        break

    return vval

  def get_argmax(self, list_np_latent_distribution):
    # get argmax
    list_p = []
    list_argmax = []
    for np_dist in list_np_latent_distribution:
      max_p = np_dist.max()
      max_indices = np.where(np_dist == max_p)[0]
      list_p.append(max_p)
      list_argmax.append(max_indices)

    return list_argmax, list_p

  def get_vval_current(self,
                       list_compatible_vval_combos,
                       list_argmax,
                       average=False):
    if not average:
      max_vval = float("-inf")
      best_argmax = None
      for combo_cur in itertools.product(*list_argmax):
        vval = self.get_vval_of_combo(combo_cur, list_compatible_vval_combos)
        if vval > max_vval:
          max_vval = vval
          best_argmax = combo_cur

      if best_argmax is None:
        return None, None
      else:
        return max_vval, best_argmax
    else:
      avg_val = 0
      count = 0
      for combo_cur in itertools.product(*list_argmax):
        count += 1
        avg_val += self.get_vval_of_combo(combo_cur,
                                          list_compatible_vval_combos)

      avg_val = avg_val / count
      return avg_val, None

  def get_intervention(self, list_np_latent_distribution: Sequence[np.ndarray],
                       obstate_idx: int):
    '''
    return: a map from an agent index to the tuple of
          (inference_probability, inferred_mental_models, desired_mental_model)
    '''
    list_vval_combo_pairs = None

    # find the optimal combination that satisfies the method and criteria
    vval_combo = None
    if self.e_combo_selection == E_ComboSelection.LeastChange:
      list_vval_combo_pairs = self.get_compatible_latent_combos(obstate_idx)
      vval_combo = self._hamming_dist_based_combo(list_vval_combo_pairs,
                                                  list_np_latent_distribution)

    elif self.e_combo_selection == E_ComboSelection.Value:
      list_vval_combo_pairs = get_combos_sorted_by_simulated_values(
          self.np_v_values, obstate_idx)
      vval_combo = self._value_based_combo(list_vval_combo_pairs,
                                           list_np_latent_distribution)
    else:
      raise NotImplementedError

    if vval_combo is None:
      return None

    # whether to intervene or not, based on value threshold
    # deterministic approach assumes the inferred combination is argmax of
    # each distribution
    do_intervention = False
    opt_vval = vval_combo[0]
    if self.e_inference_certainty == E_CertaintyHandling.Average:
      decision_average = 0
      # all possible combinations
      list_range_latent = [
          range(len(np_dist)) for np_dist in list_np_latent_distribution
      ]
      for combo in itertools.product(*list_range_latent):
        prob = 1.0
        for idx, lat in enumerate(combo):
          prob *= list_np_latent_distribution[idx][lat]

        vval = self.get_vval_of_combo(combo, list_vval_combo_pairs)
        decision_average += prob * int(
            self.deterministic_decision(opt_vval, vval))

      if decision_average > 0.5:
        do_intervention = True
    elif self.e_inference_certainty == E_CertaintyHandling.Threshold:
      list_argmax, list_p = self.get_argmax(list_np_latent_distribution)
      cur_vval, best_one = self.get_vval_current(list_vval_combo_pairs,
                                                 list_argmax,
                                                 self.argmax_cur_val_by_average)

      p_all = 1.0
      for prob in list_p:
        p_all *= prob

      if p_all >= self.inference_threshold:
        if self.deterministic_decision(opt_vval, cur_vval):
          do_intervention = True
    else:
      raise NotImplementedError

    # check if inferred combination is in the set of compatible combinations
    # if not in, get the agents whose mental model needs to be updated
    if do_intervention:
      dict_intervention_p_x_inf_x_int = {}
      for idx, lat in enumerate(vval_combo[1]):
        # if lat not in list_argmax[idx]:
        dict_intervention_p_x_inf_x_int[idx] = lat

      return dict_intervention_p_x_inf_x_int

    return None

  def deterministic_decision(self, opt_vval, cur_vval) -> bool:
    if cur_vval is None:
      return True
    else:
      return (opt_vval - cur_vval - self.intervention_cost >=
              self.intervention_threshold)

  def _value_based_combo(self, list_vval_combo_pairs,
                         list_np_latent_distribution):
    return list_vval_combo_pairs[0]

  def _hamming_dist_based_combo(self, list_vval_combo_pairs,
                                list_np_latent_distribution):
    # get argmax
    list_argmax, _ = self.get_argmax(list_np_latent_distribution)

    count_matched = []
    for _, combo in list_vval_combo_pairs:
      cnt = 0
      for idx, lat in enumerate(combo):
        if lat in list_argmax[idx]:
          cnt += 1

      count_matched.append(cnt)

    max_cnt = max(count_matched)
    if max_cnt == len(list_argmax):
      return None

    idx_combo = count_matched.index(max_cnt)
    return list_vval_combo_pairs[idx_combo]


class InterventionHighKLDiv(InterventionAbstract):
  '''
  this intervention strategy will only work for the case where mental models
  only consist of shared goals of the task.
  '''

  def __init__(self, e_combo_selection: E_ComboSelection,
               threshold: float) -> None:
    self.e_combo_selection = e_combo_selection
    self.threshold = threshold

  def intervention(self, list_np_latent_distribution: Sequence[np.ndarray],
                   obstate_idx: int):

    num_agents = len(list_np_latent_distribution)
    js_mat = np.zeros((num_agents, num_agents))
    for a_i in range(num_agents):
      for a_j in range(num_agents):
        if a_i == a_j:
          continue

        np_latent_dist_i = list_np_latent_distribution[a_i]
        np_latent_dist_j = list_np_latent_distribution[a_j]
        js_mat[a_i, a_j] = compute_js(np_latent_dist_i, np_latent_dist_j)

    mean_js = js_mat.mean(axis=1)

    ref_agent_idx = np.argmin(mean_js)
    ref_np_dist = list_np_latent_distribution[ref_agent_idx]
    ref_max_p = ref_np_dist.max()
    ref_max_latent_idx = np.where(ref_np_dist == ref_max_p)[0]

    # desired mental model is uncertain
    if len(ref_max_latent_idx) > 1:
      return None

    agent_idx = np.argmax(mean_js)
    np_dist = list_np_latent_distribution[agent_idx]
    max_p = np_dist.max()
    max_latent_idxs = np.where(np_dist == max_p)[0]

    return {agent_idx: (max_p, max_latent_idxs, ref_max_latent_idx[0])}
