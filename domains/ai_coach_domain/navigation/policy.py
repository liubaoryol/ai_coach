import os
import pickle
import numpy as np
import ai_coach_core.models.mdp as mdp_lib
import ai_coach_core.RL.planning as plan_lib
from ai_coach_domain.navigation.simulator import NavigationSimulator
from ai_coach_domain.navigation.mdp import NavigationMDP
from ai_coach_domain.navigation.maps import NAVI_MAP

GAME_MAP = NAVI_MAP

policy_static_list = []


def get_static_policy(mdp: NavigationMDP, temperature):
  global policy_static_list

  if len(policy_static_list) == 0:
    cur_dir = os.path.dirname(__file__)
    for idx in range(mdp.num_latents):
      str_q_val = os.path.join(
          cur_dir, "data/" + "navi_np_q_value_" + "%d.pickle" % (idx, ))
      with open(str_q_val, "rb") as f:
        q_value = pickle.load(f)
        policy_static_list.append(
            mdp_lib.softmax_policy_from_q_value(q_value, temperature))

  return policy_static_list


def get_static_action(mdp: NavigationMDP, agent_id, temperature, box_states,
                      a1_pos, a2_pos, x_grid, y_grid, boxes, goals, walls,
                      drops, a1_latent, a2_latent, **kwargs):
  policy_list = get_static_policy(mdp, temperature)

  if ((mdp.x_grid != x_grid) or (mdp.y_grid != y_grid) or (mdp.goals != goals)
      or (mdp.walls != walls)):
    return None

  # sidx = mdp.conv_sim_states_to_mdp_sidx(a1_pos, a2_pos, box_states)
  s1 = mdp.a1_pos_space.state_to_idx[a1_pos]
  s2 = mdp.a2_pos_space.state_to_idx[a2_pos]
  sidx = mdp.np_state_to_idx[(s1, s2)]

  latent = a1_latent
  if agent_id == NavigationSimulator.AGENT2:
    latent = a2_latent

  if latent not in mdp.latent_space.state_to_idx:
    return None

  lat_idx = mdp.latent_space.state_to_idx[latent]

  next_joint_action = np.random.choice(range(mdp.num_actions),
                                       size=1,
                                       replace=False,
                                       p=policy_list[lat_idx][sidx, :])[0]
  a1, a2 = mdp.np_idx_to_action[next_joint_action]
  act1 = mdp.a1_a_space.idx_to_action[a1]
  act2 = mdp.a2_a_space.idx_to_action[a2]

  if agent_id == NavigationSimulator.AGENT1:
    return act1
  elif agent_id == NavigationSimulator.AGENT2:
    return act2

  return None


if __name__ == "__main__":
  cur_dir = os.path.dirname(__file__)

  mdp = NavigationMDP(**GAME_MAP)
  print("Num latents: " + str(mdp.num_latents))
  print("Num states: " + str(mdp.num_states))
  print("Num actions: " + str(mdp.num_actions))

  GAMMA = 0.95
  for idx in range(mdp.num_latents):
    pi, np_v_value, np_q_value = plan_lib.value_iteration(
        mdp.np_transition_model,
        mdp.np_reward_model[idx],
        discount_factor=GAMMA,
        max_iteration=500,
        epsilon=0.01)

    str_v_val = os.path.join(cur_dir,
                             "data/navi_np_v_value_%d.pickle" % (idx, ))
    with open(str_v_val, "wb") as f:
      pickle.dump(np_v_value, f, pickle.HIGHEST_PROTOCOL)

    str_q_val = os.path.join(cur_dir,
                             "data/navi_np_q_value_%d.pickle" % (idx, ))
    with open(str_q_val, "wb") as f:
      pickle.dump(np_q_value, f, pickle.HIGHEST_PROTOCOL)
