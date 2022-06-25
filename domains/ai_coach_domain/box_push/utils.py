import numpy as np
import ai_coach_domain.box_push.agent as bp_agent
from ai_coach_domain.box_push.simulator import BoxPushSimulator
from ai_coach_core.utils.data_utils import Trajectories
from ai_coach_domain.box_push.mdp import BoxPushTeamMDP, BoxPushMDP
# learned policy
# learned tx


class TrueModelConverter:
  def __init__(self, agent1: bp_agent.BoxPushAIAgent_Abstract,
               agent2: bp_agent.BoxPushAIAgent_Abstract, num_latents) -> None:
    self.agent1 = agent1
    self.agent2 = agent2
    self.num_latents = num_latents

  def get_true_policy(self, agent_idx, latent_idx, state_idx):
    if agent_idx == 0:
      return self.agent1.policy_from_task_mdp_POV(state_idx, latent_idx)
    else:
      return self.agent2.policy_from_task_mdp_POV(state_idx, latent_idx)

  def get_true_Tx_nxsas(self, agent_idx, latent_idx, state_idx,
                        tuple_action_idx, next_state_idx):
    if agent_idx == 0:
      return self.agent1.transition_model_from_task_mdp_POV(
          latent_idx, state_idx, tuple_action_idx, next_state_idx)
    else:
      return self.agent2.transition_model_from_task_mdp_POV(
          latent_idx, state_idx, tuple_action_idx, next_state_idx)

  def get_init_latent_dist(self, agent_idx, state_idx):
    if agent_idx == 0:
      return self.agent1.init_latent_dist_from_task_mdp_POV(state_idx)
    else:
      return self.agent2.init_latent_dist_from_task_mdp_POV(state_idx)

  def true_Tx_for_var_infer(self, agent_idx, state_idx, action1_idx,
                            action2_idx, next_state_idx):
    joint_action = (action1_idx, action2_idx)
    np_Txx = np.zeros((self.num_latents, self.num_latents))
    for xidx in range(self.num_latents):
      np_Txx[xidx, :] = self.get_true_Tx_nxsas(agent_idx, xidx, state_idx,
                                               joint_action, next_state_idx)

    return np_Txx


class BoxPushTrajectories(Trajectories):
  def __init__(self, simulator: BoxPushSimulator, task_mdp: BoxPushTeamMDP,
               agent_mdp: BoxPushMDP) -> None:
    super().__init__(num_state_factors=1,
                     num_action_factors=2,
                     num_latent_factors=2,
                     num_latents=agent_mdp.num_latents)
    self.simulator = simulator
    self.task_mdp = task_mdp
    self.agent_mdp = agent_mdp

  def load_from_files(self, file_names):
    for file_nm in file_names:
      trj = self.simulator.read_file(file_nm)
      if len(trj) == 0:
        continue

      np_trj = np.zeros((len(trj), self.get_width()), dtype=np.int32)
      for tidx, vec_state_action in enumerate(trj):
        bstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat = vec_state_action

        sidx = self.task_mdp.conv_sim_states_to_mdp_sidx([bstt, a1pos, a2pos])
        aidx1 = (self.task_mdp.a1_a_space.action_to_idx[a1act]
                 if a1act is not None else Trajectories.EPISODE_END)
        aidx2 = (self.task_mdp.a2_a_space.action_to_idx[a2act]
                 if a2act is not None else Trajectories.EPISODE_END)

        xidx1 = (self.agent_mdp.latent_space.state_to_idx[a1lat]
                 if a1lat is not None else Trajectories.EPISODE_END)
        xidx2 = (self.agent_mdp.latent_space.state_to_idx[a2lat]
                 if a2lat is not None else Trajectories.EPISODE_END)

        np_trj[tidx, :] = [sidx, aidx1, aidx2, xidx1, xidx2]

      self.list_np_trajectory.append(np_trj)
