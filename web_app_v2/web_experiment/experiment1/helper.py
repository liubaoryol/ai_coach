import json
from unicodedata import name
import web_experiment.auth.util as util
from ai_coach_domain.box_push.simulator import (BoxPushSimulator)
from web_experiment import socketio

def task_intervention(game_history, game: BoxPushSimulator, room_id, name_space):
  is_mover_domain = False
  if "both_user_random" in name_space:
    is_mover_domain = True

  traj = []
  _, _, _, _, _, _, _, latent_robot = game_history[-1]
  for step, bstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat in game_history:
    traj.append({
      "box_states": bstt,
      "a1_pos": a1pos,
      "a2_pos": a2pos,
      "a1_latent": a1lat,
      "a2_latent": a2lat,
      "a1_action": game.cb_action_to_idx(game.AGENT1, a1act),
      "a2_action": game.cb_action_to_idx(game.AGENT2, a2act)
    })
  latent, prob = util.predict_human_latent(traj, len(traj) - 1, is_mover_domain)
  current_box_states = traj[-1]["box_states"]

  latent_human_predicted_state = f"{latent[0]}, {latent[1]}"
  latent_robot_state = f"{latent_robot[0]}, {latent_robot[1]}"
  
  
  print(f"latent human predicted: {latent_human_predicted_state} with probability {prob}")
  print(f"latent robot: {latent_robot}")
  
  if check_misalignment(current_box_states, latent, latent_robot, is_mover_domain):
    objs = {}
    objs["latent_human_predicted"] = latent_human_predicted_state
    objs["latent_robot"] = latent_robot_state
    objs["prob"] = prob
    objs_json = json.dumps(objs)
    socketio.emit("intervention", objs_json, room=room_id, namespace=name_space)
  

def check_misalignment(box_states, a1_latent, a2_latent, is_mover_domain):
  if is_mover_domain:
    return a1_latent != a2_latent
  else:
    available_box_count = 0
    for state in box_states:
      if state == 0:
        available_box_count += 1
    if available_box_count == 0:
      return 0
    # there are more than one bag on the floor but both agents try to pick up the same bag
    if a1_latent == a2_latent and available_box_count > 1:
      if a1_latent[0] == 'pickup':
        return True

    # should I keep this?
    # there is at least one bag on the floor but an agent is targeting the bag that the other agent is currently holding
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

# # test cases for cleanup domain

# # pick up two different bags
# assert(check_misalignment([0, 0, 0], ["pickup", 0], ["pickup", 1], False) == False)

# # pick up the same box, while there are more than 1 bag remaining, misalignment
# assert(check_misalignment([4, 0, 0], ["pickup", 1], ["pickup", 1], False) == True)

# # pick up same box, with only one bag remaining, should not be misalignment 
# assert(check_misalignment([4, 4, 0], ["pickup", 1], ["pickup", 2], False) == False)

# # agent 1 tries to pick up what agent 2 is holding, with 1 bag on the floor, should be misalignment 
# assert(check_misalignment([4, 2, 0], ["pickup", 1], ["goal", 0], False) == True)

# # agent 1 tries to pick up what agent 2 is holding, with 2 bag on the floor, should be misalignment 
# assert(check_misalignment([2, 0, 0], ["pickup", 0], ["goal", 0], False) == True)

