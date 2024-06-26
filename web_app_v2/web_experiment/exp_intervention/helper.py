import os
import time
from typing import Union
from aic_domain.box_push.simulator import (BoxPushSimulator)
from aic_domain.rescue.simulator import RescueSimulator
from web_experiment.define import EDomainType
import numpy as np
from aic_core.utils.decoding import forward_inference
from aic_core.intervention.feedback_strategy import InterventionAbstract
from aic_core.models.policy import PolicyInterface
from aic_domain.rescue.mdp import MDP_Rescue_Task


def task_intervention(latest_history_tuple, game: Union[BoxPushSimulator,
                                                        RescueSimulator],
                      policy_model: PolicyInterface, domain_type: EDomainType,
                      intervention: InterventionAbstract, prev_inference,
                      cb_policy, cb_Tx):

  task_mdp = policy_model.mdp  # type: MDP_Rescue_Task

  def conv_xidx_to_latent(idx, agent_id):
    # NOTE: For movers and rescue domains,
    #       human latent space is the same as the robot's
    latent = policy_model.conv_idx_to_latent(idx)
    return latent

  tup_state_prev, tup_action_prev = game.get_state_action_from_history_item(
      latest_history_tuple)

  sidx = task_mdp.conv_sim_states_to_mdp_sidx(tup_state_prev)
  joint_action = []
  for agent_idx in range(game.get_num_agents()):
    aidx_i = task_mdp.dict_factored_actionspace[agent_idx].action_to_idx[
        tup_action_prev[agent_idx]]
    joint_action.append(aidx_i)

  sidx_n = task_mdp.conv_sim_states_to_mdp_sidx(tuple(game.get_current_state()))
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

  human_intervention_latent = None
  robot_intervention_latent = None
  if feedback is not None:
    # update robot latent
    I_ROBOT = 1
    if I_ROBOT in feedback and feedback[I_ROBOT] is not None:
      lat_robot = feedback[I_ROBOT]
      game.agent_2.set_latent(policy_model.conv_idx_to_latent(lat_robot))

      # update latent distribution according to intervention (robot)
      np_ntv_x_dist = np.zeros(len(list_np_x_dist[I_ROBOT]))
      np_ntv_x_dist[lat_robot] = 1.0
      list_np_x_dist[I_ROBOT] = np_ntv_x_dist
      robot_intervention_latent = conv_xidx_to_latent(lat_robot, I_ROBOT)

    # update latent distribution according to intervention (human)
    I_HUMAN = 0
    if I_HUMAN in feedback and feedback[I_HUMAN] is not None:
      lat_human = feedback[I_HUMAN]
      np_inf_x_dist = list_np_x_dist[I_HUMAN]
      np_ntv_x_dist = np.zeros(len(np_inf_x_dist))
      np_ntv_x_dist[lat_human] = 1.0
      p_a = 0.9
      list_np_x_dist[I_HUMAN] = (np_ntv_x_dist * p_a + np_inf_x_dist *
                                 (1 - p_a))

      human_intervention_latent = conv_xidx_to_latent(lat_human, I_HUMAN)

  return list_np_x_dist, human_intervention_latent, robot_intervention_latent


def store_intervention_history(path, intervention_history, user_id,
                               session_name):
  file_name = get_intervention_history_file_name(path, user_id, session_name)
  dir_path = os.path.dirname(file_name)
  if dir_path != '' and not os.path.exists(dir_path):
    os.makedirs(dir_path)

  with open(file_name, 'w', newline='') as txtfile:
    # sequence
    txtfile.write('# cur_step, human_intervention, robot_intervention\n')

    for tup_label in intervention_history:
      txtfile.write('%d; %s; %s' % tup_label)
      txtfile.write('\n')


def get_intervention_history_file_name(path, user_id, session_name):
  traj_dir = os.path.join(path, user_id)

  # save somewhere
  if not os.path.exists(traj_dir):
    os.makedirs(traj_dir)

  sec, msec = divmod(time.time() * 1000, 1000)
  time_stamp = '%s.%03d' % (time.strftime('%Y-%m-%d_%H_%M_%S',
                                          time.gmtime(sec)), msec)
  file_name = ('interventions_' + session_name + '_' + str(user_id) + '_' +
               time_stamp + '.txt')
  return os.path.join(traj_dir, file_name)
