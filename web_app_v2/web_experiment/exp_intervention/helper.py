import json
from flask_socketio import emit
from web_experiment.review.util import predict_human_latent
from ai_coach_domain.box_push.simulator import (BoxPushSimulator)
from ai_coach_domain.box_push import AGENT_ACTIONSPACE
from web_experiment.define import EDomainType
import random
import numpy as np
from ai_coach_core.latent_inference.decoding import forward_inference
from ai_coach_core.intervention.feedback_strategy import InterventionAbstract


def task_intervention(game_history, game: BoxPushSimulator,
                      domain_type: EDomainType,
                      intervention: InterventionAbstract, prev_inference,
                      cb_policy, cb_Tx):
  traj = []
  _, _, _, _, _, _, _, latent_robot = game_history[-1]
  for step, bstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat in game_history:
    traj.append({
        "box_states": bstt,
        "a1_pos": a1pos,
        "a2_pos": a2pos,
        "a1_latent": a1lat,
        "a2_latent": a2lat,
        "a1_action": a1act,
        "a2_action": a2act
    })

  if domain_type == EDomainType.Movers:
    task_mdp = game.agent_2.agent_model.get_reference_mdp()

    def get_state_action(history):
      step, bstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat = history
      return (bstt, a1pos, a2pos), (a1act, a2act)

    tup_state_prev, tup_action_prev = get_state_action(game_history[-1])

    sidx = task_mdp.conv_sim_states_to_mdp_sidx(tup_state_prev)
    joint_action = []
    for agent_idx in range(game.get_num_agents()):
      aidx_i, = game.agent_2.agent_model.policy_model.conv_action_to_idx(
          (tup_action_prev[agent_idx], ))
      joint_action.append(aidx_i)

    sidx_n = task_mdp.conv_sim_states_to_mdp_sidx(
        tuple(game.get_state_for_each_agent(0)))
    list_state = [sidx, sidx_n]
    list_action = [tuple(joint_action)]

    num_lat = game.agent_2.agent_model.policy_model.get_num_latent_states()

    def init_latent_nxs(nidx, xidx, sidx):
      return 1 / num_lat  # uniform

    # list_latent = get_possible_latent_states(len(game.boxes), len(game.drops),
    #                                          len(game.goals))
    # num_latent = len(list_latent)

  _, list_np_x_dist = forward_inference(list_state, list_action,
                                        game.get_num_agents(), num_lat,
                                        cb_policy, cb_Tx, init_latent_nxs,
                                        prev_inference)
  prev_inference = list_np_x_dist

  feedback = intervention.get_intervention(list_np_x_dist, sidx_n)

  require_intervention = feedback is not None
  cur_inference = feedback[0] if feedback is not None else None\

  objs = {}
  if require_intervention:
    # hardcode intervention to happen every time

    objs["latent_human_predicted"] = feedback[0]
    objs["latent_robot"] = "Non"
    objs["prob"] = random.random()
    objs_json = json.dumps(objs)
    emit("intervention", objs_json)
  else:
    objs_json = json.dumps(objs)
    emit("no_intervention", objs_json)

  return prev_inference, cur_inference, require_intervention

  # if check_misalignment(current_box_states, latent, latent_robot, domain_type):
  #   objs = {}
  #   objs["latent_human_predicted"] = latent_human_predicted_state
  #   objs["latent_robot"] = latent_robot_state
  #   objs["prob"] = prob
  #   objs_json = json.dumps(objs)
  #   emit("intervention", objs_json)


def check_misalignment(box_states, a1_latent, a2_latent,
                       domain_type: EDomainType):
  if domain_type == EDomainType.Movers:
    return a1_latent != a2_latent
  elif domain_type == EDomainType.Cleanup:
    available_box_count = 0
    for state in box_states:
      if state == 0:
        available_box_count += 1

    if available_box_count == 0:
      return False
    # there are more than one bag on the floor but both agents try to pick up
    # the same bag
    if a1_latent == a2_latent and available_box_count > 1:
      if a1_latent[0] == 'pickup':
        return True

    # should I keep this?
    # there is at least one bag on the floor but an agent is targeting the bag
    # that the other agent is currently holding
    # if available_box_count > 0:
    #   # agent 1 targeting what agent 2 is holding
    #   if a1_latent[0] == 'pickup':
    #     if a2_latent[0] == 'goal':
    #       a1_target = a1_latent[1]
    #       # a2 is holding a1's target
    #       if box_states[a1_target] == 2:
    #         return True

    #   # agent 2 targeting what agent 1 is holding
    #   elif a2_latent[0] == 'pickup':
    #     if a1_latent[0] == 'goal':
    #       a2_target = a2_latent[1]
    #       # a2 is holding a1's target
    #       if box_states[a2_target] == 1:
    #         return True

    return False
  else:
    raise NotImplementedError
