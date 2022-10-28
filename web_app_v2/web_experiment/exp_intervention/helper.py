import json
from flask_socketio import emit
from web_experiment.review.util import predict_human_latent
from ai_coach_domain.box_push.simulator import (BoxPushSimulator)
from ai_coach_domain.box_push import AGENT_ACTIONSPACE
from web_experiment.define import EDomainType
import random


def task_intervention(game_history, game: BoxPushSimulator,
                      domain_type: EDomainType):
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

  # latent, prob = predict_human_latent(traj, len(traj) - 1, domain_type)
  current_box_states = traj[-1]["box_states"]

  # latent_human_predicted_state = str(latent)
  latent_robot_state = str(latent_robot)

  # print(f"latent human predicted: {latent_human_predicted_state}" +
  #       f"with probability {prob}")
  print(f"latent robot: {latent_robot}")

  # hardcode intervention to happen every time
  objs = {}
  objs["latent_human_predicted"] = ""
  objs["latent_robot"] = latent_robot_state
  objs["prob"] = random.random(1)
  objs_json = json.dumps(objs)
  emit("intervention", objs_json)

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
