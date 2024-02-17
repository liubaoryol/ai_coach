from abc import abstractmethod, ABC
from enum import Enum
from typing import Sequence, Callable
import itertools
import numpy as np
from aic_core.utils.result_utils import compute_js


class E_ComboSelection(Enum):
  '''
    methods to select one latent state combination 
      if there are multiple compatible combinations that satisfy the criteria
  '''
  # Value: pick one that has the highest v-value
  Value = 0
  # Acceptibility: pick one that is likely to be accepted
  #               by the team (not implemented)
  Acceptibility = 1
  # LeastChange: pick one that changes the least number of members
  #             (not implemented)
  LeastChange = 2
  # Confidence: pick one containing a latent about which the intervention system
  #            is most confident in its inference.
  Confidence = 3


class E_CertaintyHandling(Enum):
  Threshold = 0
  Average = 1


def get_sorted_x_combos(np_v_values, obstate_idx):
  'given a state(obstate_idx), get mental state combinations sorted by v-value'
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

  def get_point_estimate_of_latent_state(self, list_np_latent_distribution):
    '''
    return point estimate of latent state and its probability.
    if more than 1 latent state has the same probability, return all of them.
    '''
    list_p = []
    list_argmax = []
    for np_dist in list_np_latent_distribution:
      max_p = np_dist.max()
      max_indices = np.where(np_dist == max_p)[0]
      list_p.append(max_p)
      list_argmax.append(max_indices)

    return list_argmax, list_p

  @abstractmethod
  def get_intervention(self, list_np_latent_distribution: Sequence[np.ndarray],
                       obstate_idx: int):
    '''
    return: a map from an agent index to desired_mental_model
            if None, no intervention is needed.
    '''
    pass


class InterventionRuleBased(InterventionAbstract):
  def __init__(self,
               cb_get_compatible_latent_combos: Callable,
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
    self.cb_get_compatible_latent_combos = cb_get_compatible_latent_combos
    self.num_agent = num_agent

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

  def deterministic_rule_based_decision(self, list_valid_combos, est_combo):
    if len(list_valid_combos) == 0:
      return False

    return not (est_combo in list_valid_combos)

  def get_intervention(self, list_np_latent_distribution: Sequence[np.ndarray],
                       obstate_idx: int):
    '''
    return: a map from an agent index to desired_mental_model
            if None, no intervention is needed.
    '''
    list_valid_combos = self.cb_get_compatible_latent_combos(obstate_idx)

    picked_combo = None
    # find the optimal combination that satisfies the method and criteria
    if self.e_combo_selection == E_ComboSelection.Confidence:
      # = Example =
      # Latent spaces:
      #   - Agent 1: {A, B}    /   - Agent 2: {W, Y, Z}
      # Compatible combinations: (A, W), (B, Y), (B, Z)
      # Latent state inferences:
      #   - Agent 1: (A: 0.8, B: 0.2)   /  - Agent 2: (W: 0.4, Y: 0.1, Z: 0.5)
      #
      # This strategy will choose (a, w) because the intervention system can be
      # almost certain about its inference at least for one agent.
      # Hope to avoid unwanted interventions due to wrong estimate of latent
      picked_combo = self._confidence_based_combo(list_valid_combos,
                                                  list_np_latent_distribution)

    else:
      raise NotImplementedError

    if picked_combo is None:
      return None

    # whether to intervene or not, based on threshold
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
            self.deterministic_rule_based_decision(list_valid_combos, combo))

      if decision_average > 0.5:
        do_intervention = True
    elif self.e_inference_certainty == E_CertaintyHandling.Threshold:
      list_argmax, list_p = self.get_point_estimate_of_latent_state(
          list_np_latent_distribution)

      p_all = 1.0
      for prob in list_p:
        p_all *= prob

      if p_all >= self.inference_threshold:
        decision_average = 0
        count = 0
        for combo_cur in itertools.product(*list_argmax):
          decision_average += int(
              self.deterministic_rule_based_decision(list_valid_combos,
                                                     combo_cur))
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
      intervention_cost: float = 0.0) -> None:
    '''
    fixed_agents: if not None, the agents in the list will not be intervened.
                  if None, all agents will receive intervention.
    '''
    super().__init__(e_combo_selection=E_ComboSelection.Value,
                     e_inference_certainty=e_inference_certainty,
                     inference_threshold=inference_threshold,
                     intervention_threshold=intervention_threshold,
                     intervention_cost=intervention_cost)
    self.np_v_values = np_v_values

  def get_vval_of_combo(self, input_combo, list_compatible_vval_combos):
    vval = None
    for val, combo in list_compatible_vval_combos:
      if input_combo == combo:
        vval = val
        break

    return vval

  def get_vval_current(self, list_compatible_vval_combos, list_argmax):
    # NOTE: USE_AVERGE is for the case where there are multiple combinations
    #           with the same confidence
    # if True, use the average of their v-values as the representative v-value.
    # if False, use the highest v-value among them.
    USE_AVERAGE = False
    if not USE_AVERAGE:
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

  def _get_sorted_x_combos(self, np_v_values, obstate_idx, **kwargs):
    'given a state(obstate_idx), get mental state combinations sorted by v-value'
    return get_sorted_x_combos(np_v_values, obstate_idx)

  def get_intervention(self, list_np_latent_distribution: Sequence[np.ndarray],
                       obstate_idx: int):
    '''
    return: a map from an agent index to desired_mental_model
            if None, no intervention is needed.
    '''
    list_vval_combo_pairs = None

    # find the optimal combination that satisfies the method and criteria
    vval_combo = None
    if self.e_combo_selection == E_ComboSelection.Value:
      list_vval_combo_pairs = self._get_sorted_x_combos(self.np_v_values,
                                                        obstate_idx)
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
            self.deterministic_value_based_decision(opt_vval, vval))

      if decision_average > 0.5:
        do_intervention = True
    elif self.e_inference_certainty == E_CertaintyHandling.Threshold:
      list_argmax, list_p = self.get_point_estimate_of_latent_state(
          list_np_latent_distribution)
      cur_vval, best_one = self.get_vval_current(list_vval_combo_pairs,
                                                 list_argmax)

      p_all = 1.0
      for prob in list_p:
        p_all *= prob

      if p_all >= self.inference_threshold:
        if self.deterministic_value_based_decision(opt_vval, cur_vval):
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

  def deterministic_value_based_decision(self, opt_vval, cur_vval) -> bool:
    if cur_vval is None:
      return True
    else:
      return (opt_vval - cur_vval - self.intervention_cost >
              self.intervention_threshold)

  def _value_based_combo(self, list_vval_combo_pairs,
                         list_np_latent_distribution):
    return list_vval_combo_pairs[0]


class PartialInterventionValueBased(InterventionValueBased):
  def __init__(self,
               np_v_values: np.ndarray,
               inference_threshold: float,
               intervention_threshold: float,
               intervention_cost: float = 0,
               fixed_agents: Sequence[int] = None) -> None:
    super().__init__(np_v_values, E_CertaintyHandling.Threshold,
                     inference_threshold, intervention_threshold,
                     intervention_cost)

    self.fixed_agents = fixed_agents

  def _get_sorted_x_combos(self, np_v_values, obstate_idx, dict_fixed_latents):
    'given a state(obstate_idx), get mental state combinations sorted by v-value'

    indices = []
    num_agents = len(np_v_values.shape) - 1
    for idx in range(num_agents):
      if idx in dict_fixed_latents:
        indices.append(dict_fixed_latents[idx])
      else:
        indices.append(np.arange(np_v_values.shape[idx + 1]))

    list_combos = []
    for combo in itertools.product(*indices):
      list_combos.append((np_v_values[obstate_idx][combo], combo))

    list_combos.sort(reverse=True)

    return list_combos

  def get_intervention(self, list_np_latent_distribution: Sequence[np.ndarray],
                       obstate_idx: int):
    '''
    return: a map from an agent index to desired_mental_model
            if None, no intervention is needed.
    '''

    # point estimate of fixed agents
    dict_fixed_latents = {}
    prop_fixed = 1.0
    for idx in self.fixed_agents:
      np_dist = list_np_latent_distribution[idx]
      max_p = np_dist.max()
      max_indices = np.where(np_dist == max_p)[0]
      dict_fixed_latents[idx] = max_indices
      prop_fixed *= max_p

    list_vval_combo_pairs = self._get_sorted_x_combos(self.np_v_values,
                                                      obstate_idx,
                                                      dict_fixed_latents)
    vval_combo = self._value_based_combo(list_vval_combo_pairs,
                                         list_np_latent_distribution)

    if vval_combo is None:
      return None

    do_intervention = False
    opt_vval = vval_combo[0]

    # the same as confidence-based (threshold-based) intervention
    list_argmax, list_p = self.get_point_estimate_of_latent_state(
        list_np_latent_distribution)
    cur_vval, best_one = self.get_vval_current(list_vval_combo_pairs,
                                               list_argmax)

    p_all = 1.0
    for prob in list_p:
      p_all *= prob

    if p_all >= self.inference_threshold:
      if self.deterministic_value_based_decision(opt_vval, cur_vval):
        do_intervention = True

    if do_intervention:
      # NOTE: for fixed agents, set intervention as None
      dict_intervention = {idx: None for idx in self.fixed_agents}
      for idx, lat in enumerate(vval_combo[1]):
        if idx not in dict_intervention:
          dict_intervention[idx] = lat

      return dict_intervention

    return None
