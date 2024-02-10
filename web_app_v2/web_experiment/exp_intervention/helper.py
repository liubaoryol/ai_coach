from aic_domain.box_push.simulator import (BoxPushSimulator)
from web_experiment.define import EDomainType
import numpy as np
from aic_core.utils.decoding import forward_inference
from aic_core.intervention.feedback_strategy import InterventionAbstract
from aic_core.models.policy import PolicyInterface
from aic_domain.rescue.mdp import MDP_Rescue_Task


def task_intervention(latest_history_tuple, game: BoxPushSimulator,
                      policy_model: PolicyInterface, domain_type: EDomainType,
                      intervention: InterventionAbstract, prev_inference,
                      cb_policy, cb_Tx):

  task_mdp = policy_model.mdp  # type: MDP_Rescue_Task
  if domain_type == EDomainType.Movers:

    def get_state_action(latest_tuple):
      step, bstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat = latest_tuple
      return (bstt, a1pos, a2pos), (a1act, a2act)

  elif domain_type == EDomainType.Rescue:

    def get_state_action(latest_tuple):
      step, score, wstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat = latest_tuple
      return (wstt, a1pos, a2pos), (a1act, a2act)

  def conv_human_xidx_to_latent(idx):
    # NOTE: For movers and rescue domains,
    #       human latent space is the same as the robot's
    latent = policy_model.conv_idx_to_latent(idx)
    return latent

  tup_state_prev, tup_action_prev = get_state_action(latest_history_tuple)

  sidx = task_mdp.conv_sim_states_to_mdp_sidx(tup_state_prev)
  joint_action = []
  for agent_idx in range(game.get_num_agents()):
    aidx_i = task_mdp.dict_factored_actionspace[agent_idx].action_to_idx[
        tup_action_prev[agent_idx]]
    joint_action.append(aidx_i)

  sidx_n = task_mdp.conv_sim_states_to_mdp_sidx(
      tuple(game.get_state_for_each_agent(0)))
  list_state = [sidx, sidx_n]
  list_action = [tuple(joint_action)]

  num_lat = policy_model.get_num_latent_states()

  def init_latent_nxs(nidx, xidx, sidx):
    return 1 / num_lat  # uniform

  _, list_np_x_dist = forward_inference(list_state, list_action,
                                        game.get_num_agents(), num_lat,
                                        cb_policy, cb_Tx, init_latent_nxs,
                                        prev_inference)

  feedback = intervention.get_intervention(list_np_x_dist, sidx_n)

  if feedback is None:
    return list_np_x_dist, None
  else:
    # update robot latent
    I_ROBOT = 1
    lat_robot = feedback[I_ROBOT]
    game.agent_2.set_latent(policy_model.conv_idx_to_latent(lat_robot))

    # update latent distribution according to intervention (robot)
    np_ntv_x_dist = np.zeros(len(list_np_x_dist[I_ROBOT]))
    np_ntv_x_dist[lat_robot] = 1.0
    list_np_x_dist[I_ROBOT] = np_ntv_x_dist

    # update latent distribution according to intervention (human)
    I_HUMAN = 0
    lat_human = feedback[I_HUMAN]
    np_inf_x_dist = list_np_x_dist[I_HUMAN]
    np_ntv_x_dist = np.zeros(len(np_inf_x_dist))
    np_ntv_x_dist[lat_human] = 1.0
    p_a = 1.0
    list_np_x_dist[I_HUMAN] = (np_ntv_x_dist * p_a + np_inf_x_dist * (1 - p_a))

    return list_np_x_dist, conv_human_xidx_to_latent(feedback[0])
