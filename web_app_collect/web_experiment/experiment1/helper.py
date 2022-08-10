import json
import web_experiment.auth.util as util
from ai_coach_domain.box_push.simulator import (BoxPushSimulator)
from web_experiment import socketio


def task_intervention(game_history, game: BoxPushSimulator, room_id,
                      name_space):
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
  latent, prob = util.predict_human_latent(traj, len(traj) - 1, True)

  latent_human_predicted = f"{latent[0]}, {latent[1]}"
  latent_robot = f"{latent_robot[0]}, {latent_robot[1]}"

  print(
      f"latent human predicted: {latent_human_predicted} with probability {prob}"
  )
  print(f"latent robot: {latent_robot}")
  if (latent_robot != latent_human_predicted):
    objs = {}
    objs["latent_human_predicted"] = latent_human_predicted
    objs["latent_robot"] = latent_robot
    objs["prob"] = prob
    objs_json = json.dumps(objs)
    socketio.emit("intervention", objs_json, room=room_id, namespace=name_space)
