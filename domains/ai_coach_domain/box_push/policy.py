import os
import pickle
import numpy as np
import ai_coach_core.models.mdp as mdp_lib
import ai_coach_core.RL.planning as plan_lib
from ai_coach_core.utils.feature_utils import (get_gridworld_astar_distance,
                                               manhattan_distance)
from ai_coach_domain.box_push.mdp import (BoxPushTeamMDP, BoxPushAgentMDP,
                                          get_agent_switched_boxstates)
from ai_coach_domain.box_push.simulator import BoxPushSimulator
from ai_coach_domain.box_push.maps import EXP1_MAP
from ai_coach_domain.box_push.helper import (conv_box_idx_2_state, BoxState,
                                             EventType)

policy_exp1_list = []
policy_indv_list = []
policy_test_agent_list = []
policy_test_team_list = []


def get_action_from_cached_team_mdp(mdp: BoxPushTeamMDP, policy_list,
                                    file_prefix, agent_id, temperature,
                                    box_states, a1_pos, a2_pos, x_grid, y_grid,
                                    boxes, goals, walls, drops, a1_latent,
                                    a2_latent):
  if len(policy_list) == 0:
    cur_dir = os.path.dirname(__file__)
    for idx in range(mdp.num_latents):
      str_q_val = os.path.join(cur_dir,
                               "data/" + file_prefix + "%d.pickle" % (idx, ))
      with open(str_q_val, "rb") as f:
        q_value = pickle.load(f)
        policy_list.append(
            mdp_lib.softmax_policy_from_q_value(q_value, temperature))

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


def get_action_from_cached_agent_mdp(mdp: BoxPushAgentMDP, policy_list,
                                     file_prefix, agent_id, temperature,
                                     box_states, a1_pos, a2_pos, x_grid, y_grid,
                                     boxes, goals, walls, drops, a1_latent,
                                     a2_latent):
  if len(policy_list) == 0:
    cur_dir = os.path.dirname(__file__)
    for idx in range(mdp.num_latents):
      str_q_val = os.path.join(cur_dir,
                               "data/" + file_prefix + "%d.pickle" % (idx, ))
      with open(str_q_val, "rb") as f:
        q_value = pickle.load(f)
        policy_list.append(
            mdp_lib.softmax_policy_from_q_value(q_value, temperature))

  if ((mdp.x_grid != x_grid) or (mdp.y_grid != y_grid) or (mdp.boxes != boxes)
      or (mdp.goals != goals) or (mdp.walls != walls) or (mdp.drops != drops)):
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

  sidx = mdp.conv_sim_states_to_mdp_sidx(my_pos, teammate_pos,
                                         relative_box_states)

  if latent not in mdp.latent_space.state_to_idx:
    return None

  lat_idx = mdp.latent_space.state_to_idx[latent]

  next_aidx = np.random.choice(range(mdp.num_actions),
                               size=1,
                               replace=False,
                               p=policy_list[lat_idx][sidx, :])[0]
  act1 = mdp.conv_mdp_aidx_to_sim_action(next_aidx)

  return act1


def get_exp1_action(mdp_team: BoxPushTeamMDP, agent_id, temperature, box_states,
                    a1_pos, a2_pos, x_grid, y_grid, boxes, goals, walls, drops,
                    a1_latent, a2_latent, **kwargs):
  global policy_exp1_list

  return get_action_from_cached_team_mdp(mdp_team, policy_exp1_list,
                                         "box_push_np_q_value_exp1_", agent_id,
                                         temperature, box_states, a1_pos,
                                         a2_pos, x_grid, y_grid, boxes, goals,
                                         walls, drops, a1_latent, a2_latent)


def get_indv_action(mdp_indv: BoxPushAgentMDP, agent_id, temperature,
                    box_states, a1_pos, a2_pos, x_grid, y_grid, boxes, goals,
                    walls, drops, a1_latent, a2_latent, **kwargs):
  global policy_indv_list

  return get_action_from_cached_agent_mdp(mdp_indv, policy_indv_list,
                                          "box_push_np_q_value_indv_", agent_id,
                                          temperature, box_states, a1_pos,
                                          a2_pos, x_grid, y_grid, boxes, goals,
                                          walls, drops, a1_latent, a2_latent)


def get_exp1_policy(mdp: BoxPushTeamMDP, temperature):
  global policy_exp1_list

  if len(policy_exp1_list) == 0:
    cur_dir = os.path.dirname(__file__)
    for idx in range(mdp.num_latents):
      str_q_val = os.path.join(
          cur_dir,
          "data/" + "box_push_np_q_value_exp1_" + "%d.pickle" % (idx, ))
      with open(str_q_val, "rb") as f:
        q_value = pickle.load(f)
        policy_exp1_list.append(
            mdp_lib.softmax_policy_from_q_value(q_value, temperature))

  return policy_exp1_list


def get_indv_policy(mdp: BoxPushAgentMDP, temperature):
  global policy_indv_list

  if len(policy_indv_list) == 0:
    cur_dir = os.path.dirname(__file__)
    for idx in range(mdp.num_latents):
      str_q_val = os.path.join(
          cur_dir,
          "data/" + "box_push_np_q_value_indv_" + "%d.pickle" % (idx, ))
      with open(str_q_val, "rb") as f:
        q_value = pickle.load(f)
        policy_indv_list.append(
            mdp_lib.softmax_policy_from_q_value(q_value, temperature))

  return policy_indv_list


def get_simple_action(agent_id, box_states, a1_pos, a2_pos, x_grid, y_grid,
                      boxes, goals, walls, drops, a1_latent, a2_latent,
                      **kwargs):
  np_gridworld = np.zeros((x_grid, y_grid))
  for coord in walls:
    np_gridworld[coord] = 1

  a1_hold = False
  a2_hold = False
  for idx, bidx in enumerate(box_states):
    bstate = conv_box_idx_2_state(bidx, len(drops), len(goals))
    if bstate[0] == BoxState.WithAgent1:
      a1_hold = True
    elif bstate[0] == BoxState.WithAgent2:
      a2_hold = True
    elif bstate[0] == BoxState.WithBoth:
      a1_hold = True
      a2_hold = True

  my_pos = a1_pos
  my_hold = a1_hold
  if agent_id == BoxPushSimulator.AGENT2:
    my_pos = a2_pos
    my_hold = a2_hold

  if my_hold:
    if my_pos in goals:
      return EventType.HOLD

    for idx, bidx in enumerate(box_states):
      bstate = conv_box_idx_2_state(bidx, len(drops), len(goals))
      if bstate[0] == BoxState.Original:
        np_gridworld[boxes[idx]] = 1
      elif bstate[0] in [BoxState.WithAgent1, BoxState.WithBoth]:
        np_gridworld[a1_pos] = 1
      elif bstate[0] == BoxState.WithAgent2:
        np_gridworld[a2_pos] = 1
      elif bstate[0] == BoxState.OnDropLoc:
        np_gridworld[drops[bstate[1]]] = 1

    path = get_gridworld_astar_distance(np_gridworld, my_pos, goals,
                                        manhattan_distance)
    if len(path) == 0:
      return EventType.STAY
    else:
      x = path[0][0] - my_pos[0]
      y = path[0][1] - my_pos[1]
      if x > 0:
        return EventType.RIGHT
      elif x < 0:
        return EventType.LEFT
      elif y > 0:
        return EventType.DOWN
      elif y < 0:
        return EventType.UP
      else:
        return EventType.STAY
  else:
    valid_boxes = []
    for idx, bidx in enumerate(box_states):
      bstate = conv_box_idx_2_state(bidx, len(drops), len(goals))
      if bstate[0] == BoxState.Original:
        valid_boxes.append(boxes[idx])
      elif bstate[0] == BoxState.OnDropLoc:
        valid_boxes.append(drops[bstate[1]])

    if len(valid_boxes) == 0:
      return EventType.STAY

    if my_pos in valid_boxes:
      return EventType.HOLD
    else:
      path = get_gridworld_astar_distance(np_gridworld, my_pos, valid_boxes,
                                          manhattan_distance)
      if len(path) == 0:
        return EventType.STAY
      else:
        x = path[0][0] - my_pos[0]
        y = path[0][1] - my_pos[1]
        if x > 0:
          return EventType.RIGHT
        elif x < 0:
          return EventType.LEFT
        elif y > 0:
          return EventType.DOWN
        elif y < 0:
          return EventType.UP
        else:
          return EventType.STAY


def get_test_indv_policy(mdp: BoxPushAgentMDP, temperature):
  global policy_test_agent_list

  if len(policy_test_agent_list) == 0:
    GAMMA = 0.95
    for idx in range(mdp.num_latents):
      pi, np_v_value, np_q_value = plan_lib.value_iteration(
          mdp.np_transition_model,
          mdp.np_reward_model[idx],
          discount_factor=GAMMA,
          max_iteration=500,
          epsilon=0.01)

      policy_test_agent_list.append(
          mdp_lib.softmax_policy_from_q_value(np_q_value, temperature))

  return policy_test_agent_list


def get_test_team_policy(mdp: BoxPushTeamMDP, temperature):
  global policy_test_team_list

  if len(policy_test_team_list) == 0:
    GAMMA = 0.95
    for idx in range(mdp.num_latents):
      pi, np_v_value, np_q_value = plan_lib.value_iteration(
          mdp.np_transition_model,
          mdp.np_reward_model[idx],
          discount_factor=GAMMA,
          max_iteration=500,
          epsilon=0.01)

      policy_test_team_list.append(
          mdp_lib.softmax_policy_from_q_value(np_q_value, temperature))

  return policy_test_team_list


def get_test_indv_action(mdp: BoxPushAgentMDP, agent_id, temperature,
                         box_states, a1_pos, a2_pos, x_grid, y_grid, boxes,
                         goals, walls, drops, a1_latent, a2_latent, **kwargs):
  policy_list = get_test_indv_policy(mdp, temperature)

  if ((mdp.x_grid != x_grid) or (mdp.y_grid != y_grid) or (mdp.boxes != boxes)
      or (mdp.goals != goals) or (mdp.walls != walls) or (mdp.drops != drops)):
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

  sidx = mdp.conv_sim_states_to_mdp_sidx(my_pos, teammate_pos,
                                         relative_box_states)

  if latent not in mdp.latent_space.state_to_idx:
    return None

  lat_idx = mdp.latent_space.state_to_idx[latent]

  next_aidx = np.random.choice(range(mdp.num_actions),
                               size=1,
                               replace=False,
                               p=policy_list[lat_idx][sidx, :])[0]
  act1 = mdp.conv_mdp_aidx_to_sim_action(next_aidx)
  return act1


def get_test_team_action(mdp: BoxPushTeamMDP, agent_id, temperature, box_states,
                         a1_pos, a2_pos, x_grid, y_grid, boxes, goals, walls,
                         drops, a1_latent, a2_latent, **kwargs):
  policy_list = get_test_team_policy(mdp, temperature)

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
  GEN_EXP1 = True
  GEN_INDV = True

  cur_dir = os.path.dirname(__file__)
  if GEN_EXP1:
    from ai_coach_domain.box_push.mdp import BoxPushTeamMDP_AlwaysTogether

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
    from ai_coach_domain.box_push.mdp import BoxPushAgentMDP_AlwaysAlone

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
