import os
import pickle
import numpy as np
import ai_coach_core.models.mdp as mdp_lib
import ai_coach_core.RL.planning as plan_lib
from ai_coach_domain.box_push.simulator import BoxPushSimulator
from ai_coach_domain.box_push_static.mdp import StaticBoxPushMDP
from ai_coach_domain.box_push.maps import EXP1_MAP

GAME_MAP = EXP1_MAP

policy_static_list = []


def get_static_policy(mdp: StaticBoxPushMDP, temperature):
  global policy_static_list

  if len(policy_static_list) == 0:
    cur_dir = os.path.dirname(__file__)
    for idx in range(mdp.num_latents):
      str_q_val = os.path.join(
          cur_dir,
          "data/" + "box_push_np_q_value_static_" + "%d.pickle" % (idx, ))
      with open(str_q_val, "rb") as f:
        q_value = pickle.load(f)
        policy_static_list.append(
            mdp_lib.softmax_policy_from_q_value(q_value, temperature))

  return policy_static_list


def get_static_action(mdp: StaticBoxPushMDP, agent_id, temperature, box_states,
                      a1_pos, a2_pos, x_grid, y_grid, boxes, goals, walls,
                      drops, a1_latent, a2_latent, **kwargs):
  policy_list = get_static_policy(mdp, temperature)

  if ((mdp.x_grid != x_grid) or (mdp.y_grid != y_grid) or (mdp.boxes != boxes)
      or (mdp.goals != goals) or (mdp.walls != walls) or (mdp.drops != drops)):
    return None

  sidx = mdp.conv_sim_states_to_mdp_sidx(a1_pos, a2_pos, box_states)

  latent = a1_latent
  if agent_id == BoxPushSimulator.AGENT2:
    latent = a2_latent

  if latent not in mdp.latent_space.state_to_idx:
    return None

  lat_idx = mdp.latent_space.state_to_idx[latent]

  next_joint_action = np.random.choice(range(mdp.num_actions),
                                       size=1,
                                       replace=False,
                                       p=policy_list[lat_idx][sidx, :])[0]
  act1, act2 = mdp.conv_mdp_aidx_to_sim_action(next_joint_action)

  if agent_id == BoxPushSimulator.AGENT1:
    return act1
  elif agent_id == BoxPushSimulator.AGENT2:
    return act2

  return None


if __name__ == "__main__":
  cur_dir = os.path.dirname(__file__)

  box_push_mdp = StaticBoxPushMDP(**GAME_MAP)
  print("Num latents: " + str(box_push_mdp.num_latents))
  print("Num states: " + str(box_push_mdp.num_states))
  print("Num actions: " + str(box_push_mdp.num_actions))

  GAMMA = 0.95
  for idx in range(box_push_mdp.num_latents):
    pi, np_v_value, np_q_value = plan_lib.value_iteration(
        box_push_mdp.np_transition_model,
        box_push_mdp.np_reward_model[idx],
        discount_factor=GAMMA,
        max_iteration=500,
        epsilon=0.01)

    str_v_val = os.path.join(
        cur_dir, "data/box_push_np_v_value_static_%d.pickle" % (idx, ))
    with open(str_v_val, "wb") as f:
      pickle.dump(np_v_value, f, pickle.HIGHEST_PROTOCOL)

    str_q_val = os.path.join(
        cur_dir, "data/box_push_np_q_value_static_%d.pickle" % (idx, ))
    with open(str_q_val, "wb") as f:
      pickle.dump(np_q_value, f, pickle.HIGHEST_PROTOCOL)
