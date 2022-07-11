import web_experiment.auth.util as util
from ai_coach_domain.box_push.simulator import (BoxPushSimulator)

def task_intervention(game_history, game: BoxPushSimulator):
  traj = []
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
  print(util.predict_human_latent(traj, len(traj) - 1, True))