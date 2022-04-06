from typing import Callable, Sequence, Optional
import numpy as np
import scipy.optimize
from ai_coach_core.model_inference.IRL.maxent_irl import (MaxEntIRL,
                                                          T_StateActionSeqence)
from ai_coach_core.models.mdp import MDP
from ai_coach_core.model_inference.IRL.constraints import (RewardConstraints,
                                                           NumericalRelation)


class CGMaxEntIRL(MaxEntIRL):
  def __init__(self,
               trajectories: Sequence[T_StateActionSeqence],
               mdp: MDP,
               feature_extractor: Callable[[MDP, int, int], np.ndarray],
               reward_constraints: RewardConstraints,
               gamma: float = 0.9,
               initial_prop: Optional[np.ndarray] = None,
               learning_rate: float = 0.01,
               decay: float = 0.001,
               max_value_iter: int = 100,
               epsilon: float = 0.001):
    super().__init__(trajectories, mdp, feature_extractor, gamma, initial_prop,
                     learning_rate, decay, max_value_iter, epsilon)
    possible_states = list(range(self.mdp.num_states))
    self.raw_constraints = reward_constraints.get_numerical_constraints(
        possible_states)

  def update_weights(self):
    super().update_weights()  # original weights

    # projected gradient descent
    if self.iteration % 5 == 0:

      equalitiy_constraints_A = []
      equalitiy_constraints_b = []
      inequalitiy_constraints_A = []
      inequalitiy_constraints_b = []
      for cons in self.raw_constraints:
        equ, input_states = cons
        assert equ.num_var == len(input_states)
        feat_sum = np.zeros(self.weights.shape)
        for idx, state in enumerate(input_states):
          feat = self.feature_extractor(self.mdp, state, None)
          feat_sum = feat_sum + equ.a[idx] * feat

        if equ.op == NumericalRelation.EQUAL:
          equalitiy_constraints_A.append(feat_sum)
          equalitiy_constraints_b.append(equ.b)
        elif equ.op == NumericalRelation.GREATER_EQUAL:
          inequalitiy_constraints_A.append(feat_sum)
          inequalitiy_constraints_b.append(equ.b)
        else:
          raise ValueError

      def min_euclidean_dist(projected_weights):
        return np.linalg.norm(self.weights - projected_weights)

      constraints = []
      if equalitiy_constraints_A:
        A_eq = np.array(equalitiy_constraints_A)
        b_eq = np.array(equalitiy_constraints_b)
        constraints.append({"type": "eq", "fun": lambda x: A_eq @ x + b_eq})

      if inequalitiy_constraints_A:
        A_in = np.array(inequalitiy_constraints_A)
        b_in = np.array(inequalitiy_constraints_b)
        constraints.append({"type": "ineq", "fun": lambda x: A_in @ x + b_in})

      res = scipy.optimize.minimize(min_euclidean_dist,
                                    self.weights,
                                    constraints=constraints,
                                    method="COBYLA")
      self.weights = res.x
