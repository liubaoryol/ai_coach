from typing import Callable, Tuple
import numpy as np
from ai_coach_core.utils.data_utils import Trajectories
from ai_coach_domain.rescue_v2.define import AGENT_ACTIONSPACE
from ai_coach_domain.rescue_v2.simulator import RescueSimulatorV2
from ai_coach_domain.rescue_v2.mdp import MDP_Rescue_Task


class RescueV2Trajectories(Trajectories):

  def __init__(self, task_mdp: MDP_Rescue_Task, tup_num_latents: Tuple[int,
                                                                       ...],
               cb_conv_latent_to_idx: Callable[[int, int], int]) -> None:
    super().__init__(num_state_factors=1,
                     num_action_factors=3,
                     num_latent_factors=3,
                     tup_num_latents=tup_num_latents)
    self.task_mdp = task_mdp
    self.cb_conv_latent_to_idx = cb_conv_latent_to_idx

  def load_from_files(self, file_names):
    for file_nm in file_names:
      trj = RescueSimulatorV2.read_file(file_nm)
      if len(trj) == 0:
        continue

      np_trj = np.zeros((len(trj), self.get_width()), dtype=np.int32)
      for tidx, vec_state_action in enumerate(trj):
        (scr, wstt, a1pos, a2pos, a3pos, a1act, a2act, a3act, a1lat, a2lat,
         a3lat) = vec_state_action

        sidx = self.task_mdp.conv_sim_states_to_mdp_sidx(
            [wstt, a1pos, a2pos, a3pos])
        aidx1 = (AGENT_ACTIONSPACE.action_to_idx[a1act]
                 if a1act is not None else Trajectories.EPISODE_END)
        aidx2 = (AGENT_ACTIONSPACE.action_to_idx[a2act]
                 if a2act is not None else Trajectories.EPISODE_END)
        aidx3 = (AGENT_ACTIONSPACE.action_to_idx[a3act]
                 if a3act is not None else Trajectories.EPISODE_END)

        xidx1 = (self.cb_conv_latent_to_idx(0, a1lat)
                 if a1lat is not None else Trajectories.EPISODE_END)
        xidx2 = (self.cb_conv_latent_to_idx(1, a2lat)
                 if a2lat is not None else Trajectories.EPISODE_END)
        xidx3 = (self.cb_conv_latent_to_idx(2, a3lat)
                 if a3lat is not None else Trajectories.EPISODE_END)
        np_trj[tidx, :] = [sidx, aidx1, aidx2, aidx3, xidx1, xidx2, xidx3]

      self.list_np_trajectory.append(np_trj)
