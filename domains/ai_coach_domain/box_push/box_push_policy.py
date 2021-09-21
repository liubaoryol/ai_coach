import os
import pickle
import numpy as np
import models.mdp as mdp_lib
from ai_coach_domain.box_push.box_push_team_mdp import BoxPushTeamMDP
from ai_coach_domain.box_push.box_push_agent_mdp import (
    BoxPushAgentMDP, get_agent_switched_boxstates)
from ai_coach_domain.box_push.box_push_simulator import BoxPushSimulator
from ai_coach_domain.box_push.box_push_maps import EXP1_MAP

policy_exp1_list = None
policy_indv_list = None


def get_exp1_action(mdp_exp1: BoxPushTeamMDP, agent_id, temperature, box_states,
                    a1_pos, a2_pos, x_grid, y_grid, boxes, goals, walls, drops,
                    a1_latent, a2_latent, **kwargs):
  global policy_exp1_list

  if policy_exp1_list is None:
    policy_exp1_list = []
    cur_dir = os.path.dirname(__file__)
    for idx in range(mdp_exp1.num_latents):
      str_q_val = os.path.join(
          cur_dir, "data/box_push_np_q_value_exp1_%d.pickle" % (idx, ))
      with open(str_q_val, "rb") as f:
        q_value = pickle.load(f)
        policy_exp1_list.append(
            mdp_lib.softmax_policy_from_q_value(q_value, temperature))

  if ((mdp_exp1.x_grid != x_grid) or (mdp_exp1.y_grid != y_grid)
      or (mdp_exp1.boxes != boxes) or (mdp_exp1.goals != goals)
      or (mdp_exp1.walls != walls) or (mdp_exp1.drops != drops)):
    return None

  sidx = mdp_exp1.conv_sim_states_to_mdp_sidx(a1_pos, a2_pos, box_states)

  latent = a1_latent
  if agent_id == BoxPushSimulator.AGENT2:
    latent = a2_latent

  lat_idx = mdp_exp1.latent_space.state_to_idx[latent]

  next_joint_action = np.random.choice(range(mdp_exp1.num_actions),
                                       size=1,
                                       replace=False,
                                       p=policy_exp1_list[lat_idx][sidx, :])[0]
  act1, act2 = mdp_exp1.conv_mdp_aidx_to_sim_action(next_joint_action)

  if agent_id == BoxPushSimulator.AGENT1:
    return act1
  elif agent_id == BoxPushSimulator.AGENT2:
    return act2

  return None


def get_indv_action(mdp_indv: BoxPushAgentMDP, agent_id, temperature,
                    box_states, a1_pos, a2_pos, x_grid, y_grid, boxes, goals,
                    walls, drops, a1_latent, a2_latent, **kwargs):
  global policy_indv_list

  if policy_indv_list is None:
    policy_indv_list = []
    cur_dir = os.path.dirname(__file__)
    for idx in range(mdp_indv.num_latents):
      str_q_val = os.path.join(
          cur_dir, "data/box_push_np_q_value_indv_%d.pickle" % (idx, ))
      with open(str_q_val, "rb") as f:
        q_value = pickle.load(f)
        policy_indv_list.append(
            mdp_lib.softmax_policy_from_q_value(q_value, temperature))

  if ((mdp_indv.x_grid != x_grid) or (mdp_indv.y_grid != y_grid)
      or (mdp_indv.boxes != boxes) or (mdp_indv.goals != goals)
      or (mdp_indv.walls != walls) or (mdp_indv.drops != drops)):
    return None

  my_pos = a1_pos
  teammate_pos = a2_pos
  relative_box_states = box_states
  latent = a1_latent
  if agent_id == BoxPushSimulator.AGENT2:
    my_pos = a2_pos
    teammate_pos = a1_pos
    latent = a2_latent
    relative_box_states = get_agent_switched_boxstates(box_states, len(drops),
                                                       len(goals))

  sidx = mdp_indv.conv_sim_states_to_mdp_sidx(my_pos, teammate_pos,
                                              relative_box_states)

  lat_idx = mdp_indv.latent_space.state_to_idx[latent]

  next_aidx = np.random.choice(range(mdp_indv.num_actions),
                               size=1,
                               replace=False,
                               p=policy_indv_list[lat_idx][sidx, :])[0]
  act1 = mdp_indv.conv_mdp_aidx_to_sim_action(next_aidx)

  return act1


if __name__ == "__main__":
  import RL.planning as plan_lib
  GEN_EXP1 = False
  GEN_INDV = False

  cur_dir = os.path.dirname(__file__)
  if GEN_EXP1:
    from ai_coach_domain.box_push.box_push_team_mdp import (
        BoxPushTeamMDP_AlwaysTogether)

    box_push_mdp = BoxPushTeamMDP_AlwaysTogether(**EXP1_MAP)
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
          cur_dir, "data/box_push_np_v_value_exp1_%d.pickle" % (idx, ))
      with open(str_v_val, "wb") as f:
        pickle.dump(np_v_value, f, pickle.HIGHEST_PROTOCOL)

      str_q_val = os.path.join(
          cur_dir, "data/box_push_np_q_value_exp1_%d.pickle" % (idx, ))
      with open(str_q_val, "wb") as f:
        pickle.dump(np_q_value, f, pickle.HIGHEST_PROTOCOL)

  if GEN_INDV:
    from ai_coach_domain.box_push.box_push_agent_mdp import (
        BoxPushAgentMDP_AlwaysAlone)

    box_push_mdp = BoxPushAgentMDP_AlwaysAlone(**EXP1_MAP)
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
          cur_dir, "data/box_push_np_v_value_indv_%d.pickle" % (idx, ))
      with open(str_v_val, "wb") as f:
        pickle.dump(np_v_value, f, pickle.HIGHEST_PROTOCOL)

      str_q_val = os.path.join(
          cur_dir, "data/box_push_np_q_value_indv_%d.pickle" % (idx, ))
      with open(str_q_val, "wb") as f:
        pickle.dump(np_q_value, f, pickle.HIGHEST_PROTOCOL)
