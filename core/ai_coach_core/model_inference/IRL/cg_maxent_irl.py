from typing import Callable, Sequence, Optional
import numpy as np
import scipy.optimize
from ai_coach_core.model_inference.IRL.maxent_irl import (MaxEntIRL,
                                                          T_StateActionSeqence)
from ai_coach_core.models.mdp import MDP


class CGMaxEntIRL(MaxEntIRL):
  def __init__(self,
               trajectories: Sequence[T_StateActionSeqence],
               mdp: MDP,
               feature_extractor: Callable[[MDP, int, int], np.ndarray],
               cb_get_unary_constraints,
               cb_get_binary_constraints,
               gamma: float = 0.9,
               initial_prop: Optional[np.ndarray] = None,
               learning_rate: float = 0.01,
               decay: float = 0.001,
               max_value_iter: int = 100,
               epsilon: float = 0.001):
    super().__init__(trajectories, mdp, feature_extractor, gamma, initial_prop,
                     learning_rate, decay, max_value_iter, epsilon)

  def update_weights(self):
    super().update_weights()

    def min_euclidean_dist(new_weights):
      return np.linalg.norm(self.weights - new_weights)

    # constraints = [{"type": "ineq", "fun": lambda x:   }]
    # scipy.optimize.minimize(min_euclidean_dist, self.weights, args)
