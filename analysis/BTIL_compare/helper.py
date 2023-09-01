from typing import Tuple, Optional, Union
import os
import warnings
import logging
import pickle
from aic_domain.box_push.mdp import BoxPushMDP
from aic_domain.rescue.mdp import MDP_Rescue

g_loaded_transition_model = None


def cached_transition(data_dir: str,
                      save_prefix: str,
                      mdp_task: Union[BoxPushMDP, MDP_Rescue],
                      sidx: int,
                      tup_aidx: Tuple[int, ...],
                      sidx_n: Optional[int] = None):
  global g_loaded_transition_model
  if g_loaded_transition_model is None:
    file_name = save_prefix + "_transition_" + mdp_task.map_to_str()
    pickle_trans_s = os.path.join(data_dir, file_name + ".pickle")
    if os.path.exists(pickle_trans_s):
      with open(pickle_trans_s, 'rb') as handle:
        g_loaded_transition_model = pickle.load(handle)
      logging.info("transition_s loaded by pickle")
      warnings.warn(
          "The transition has been loaded from a file ({}). "
          "If any related implementation is changed, "
          "be sure to delete the saved file and regenerate it.".format(
              os.path.basename(pickle_trans_s)),
          stacklevel=2)
    else:
      g_loaded_transition_model = mdp_task.np_transition_model
      dir_name = os.path.dirname(pickle_trans_s)
      if not os.path.exists(dir_name):
        os.makedirs(dir_name)
      logging.info("save transition_s by pickle")
      with open(pickle_trans_s, 'wb') as handle:
        pickle.dump(g_loaded_transition_model,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

  aidx_team = mdp_task.np_action_to_idx[tup_aidx]
  if sidx_n is None:
    return g_loaded_transition_model[sidx, aidx_team].todense()
  else:
    p = g_loaded_transition_model[sidx, aidx_team, sidx_n]
    return p
