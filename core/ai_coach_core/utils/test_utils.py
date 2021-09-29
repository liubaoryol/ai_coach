import numpy as np
from tqdm import tqdm
from ai_coach_core.utils.exceptions import InvalidTransitionError
from ai_coach_core.models.mdp import MDP


def check_transition_validity(mdp: MDP):
  for sidx in tqdm(range(mdp.num_states)):
    for aidx in range(mdp.num_actions):
      try:
        np_next_p_state = mdp.transition_model(sidx, aidx)
        if not np.isclose(np.sum(np_next_p_state[:, 0]), 1, rtol=0, atol=1e-10):
          return False
      except InvalidTransitionError:
        pass

  return True
