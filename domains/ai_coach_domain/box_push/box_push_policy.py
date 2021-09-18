import pickle
import numpy as np
import models.mdp as mdp_lib
from .box_push_mdp import BoxPushMDP
from .box_push_simulator import BoxPushSimulator

policy_6by6 = None  # type: BoxPushMDP
mdp_6by6 = None


def get_6by6_action(agent_id, box_states, a1_pos, a2_pos, a1_latent, a2_latent,
                    x_grid, y_grid, boxes, goals, walls, drops, **kwargs):
  global policy_6by6, mdp_6by6
  if policy_6by6 is None:
    with open("box_push_np_q_value.pickle", 'rb') as f:
      q_value = pickle.load(f)
      policy_6by6 = mdp_lib.softmax_policy_from_q_value(q_value, 3)

  if mdp_6by6 is None:
    GRID_X = 6
    GRID_Y = 6
    game_map = {
        "boxes": [(0, 1), (3, 1)],
        "goals": [(GRID_X - 1, GRID_Y - 1)],
        "walls": [(GRID_X - 2, GRID_Y - i - 1) for i in range(3)],
        "drops": []
    }
    mdp_6by6 = BoxPushMDP(GRID_X, GRID_Y, **game_map)

  if ((mdp_6by6.x_grid != x_grid) or (mdp_6by6.y_grid != y_grid)
      or (mdp_6by6.boxes != boxes) or (mdp_6by6.goals != goals)
      or (mdp_6by6.walls != walls) or (mdp_6by6.drops != drops)):
    return None

  sidx = mdp_6by6.conv_sim_states_to_mdp_sidx(a1_pos, a2_pos, box_states)

  next_joint_action = np.random.choice(range(mdp_6by6.num_actions),
                                       size=1,
                                       replace=False,
                                       p=policy_6by6[sidx, :])[0]
  act1, act2 = mdp_6by6.conv_mdp_aidx_to_sim_action(next_joint_action)

  if agent_id == BoxPushSimulator.AGENT1:
    return act1
  elif agent_id == BoxPushSimulator.AGENT2:
    return act2

  return None
