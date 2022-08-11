import numpy as np
from web_experiment import socketio
from flask import (session, current_app)
import json
import web_experiment.experiment1.events_impl as event_impl

from ai_coach_core.latent_inference.decoding import (forward_inference,
                                                     most_probable_sequence)
from ai_coach_core.utils.data_utils import Trajectories
from ai_coach_domain.box_push.maps import EXP1_MAP
from ai_coach_domain.box_push.agent_model import (
    assumed_initial_mental_distribution)
import ai_coach_domain.box_push.maps as bp_maps
import ai_coach_domain.box_push.simulator as bp_sim
import ai_coach_domain.box_push.mdp as bp_mdp
from ai_coach_domain.box_push.defines import (idx_to_action_for_simulator,
                                              EventType)
from ai_coach_domain.box_push.defines import get_possible_latent_states
import os
import glob


def load_session_trajectory(session_name, id):
  error = None
  traj_path = current_app.config["TRAJECTORY_PATH"]
  path = f"{id}/session_{session_name}_{id}*.txt"
  fileExpr = os.path.join(traj_path, path)
  # find any matching files
  files = glob.glob(fileExpr)
  if len(files) == 0:
    # does not find a match, error handling
    error = f"No file found that matches {id}, session {session_name}"
  else:
    file = files[0]
    traj = read_file(file)
    session["dict"] = traj
    session['index'] = 0
    session['max_index'] = len(traj) - 1
    session['replay_id'] = id
    session['session_name'] = session_name
    session['possible_latent_states'] = get_possible_latent_states(
        len(traj[0]['boxes']), len(traj[0]['drops']), len(traj[0]['goals']))
    # dummy latent human prediction
    session['latent_human_predicted'] = [None] * len(traj)
    session['latent_human_recorded'] = [None] * len(traj)
     
    print(len(traj))
    return error


def read_file(file_name):
  traj = []
  x_grid = EXP1_MAP['x_grid']
  y_grid = EXP1_MAP['y_grid']
  boxes = EXP1_MAP['boxes']
  goals = EXP1_MAP['goals']
  drops = EXP1_MAP['drops']
  walls = EXP1_MAP['walls']
  wall_dir = EXP1_MAP['wall_dir']

  with open(file_name, newline='') as txtfile:
    lines = txtfile.readlines()
    i_start = 0
    for i_r, row in enumerate(lines):
      if row == ('# cur_step, box_state, a1_pos, a2_pos, ' +
                 'a1_act, a2_act, a1_latent, a2_latent\n'):
        i_start = i_r
        break

    for i_r in range(i_start + 1, len(lines)):
      line = lines[i_r]
      states = line.rstrip()[:-1].split("; ")
      if len(states) < 8:
        for dummy in range(8 - len(states)):
          states.append(None)
      step, bstate, a1pos, a2pos, a1act, a2act, a1lat, a2lat = states
      box_state = tuple([int(elem) for elem in bstate.split(", ")])
      a1_pos = tuple([int(elem) for elem in a1pos.split(", ")])
      a2_pos = tuple([int(elem) for elem in a2pos.split(", ")])
      if a1lat is None:
        a1_lat = None
      else:
        a1lat_tmp = a1lat.split(", ")
        a1_lat = (a1lat_tmp[0], int(a1lat_tmp[1]))
      if a2lat is None:
        a2_lat = None
      else:
        a2lat_tmp = a2lat.split(", ")
        a2_lat = (a2lat_tmp[0], int(a2lat_tmp[1]))
      if a1act is None:
        a1_act = None
      else:
        a1_act = idx_to_action_for_simulator(0, int(a1act)).value
      if a2act is None:
        a2_act = None
      else:
        a2_act = idx_to_action_for_simulator(1, int(a2act)).value
      traj.append({
          "x_grid": x_grid,
          "y_grid": y_grid,
          "box_states": box_state,
          "boxes": boxes,
          "goals": goals,
          "drops": drops,
          "walls": walls,
          "a1_pos": a1_pos,
          "a2_pos": a2_pos,
          "a1_latent": a1_lat,
          "a2_latent": a2_lat,
          "current_step": step,
          "a1_action": a1_act,
          "a2_action": a2_act,
          "wall_dir": wall_dir
      })
  return traj


def update_canvas(env_id, namespace, update_latent, mode,
                  is_movers_domain=True):
  if 'dict' in session and 'index' in session:
    dict = session['dict'][session['index']]
    dict_update = event_impl.get_game_drawing_obj(dict, False, [], is_movers_domain)
    event_impl.update_html_canvas(dict_update, env_id, namespace)

    # update latent states
    if update_latent:
      update_latent_state(env_id, namespace, is_movers_domain, mode)
      

def update_latent_state(env_id, namespace, is_movers_domain, mode):
  latent_human, latent_human_predicted, latent_robot = get_latent_states(
        is_movers_domain, mode = mode)
  objs = {}
  objs['latent_human'] = latent_human
  objs['latent_robot'] = latent_robot
  objs['latent_human_predicted'] = latent_human_predicted
  
  if mode == "replay":
    objs['latent_states'] = "replay"
  elif mode == "predicted":
    objs['latent_states'] = "predicted"
  elif mode == "collected":
    objs['latent_states'] = "collected"
  print(objs)
  objs_json = json.dumps(objs)
  str_emit = 'update_latent'
  socketio.emit(str_emit, objs_json, room=env_id, namespace=namespace)

def get_latent_states(is_movers_domain, mode):
  dict = session['dict'][session['index']]
  latent_human = ""
  latent_robot = ""
  latent_human_predicted = ""
  if mode == "replay":
    if dict['a1_latent']:
      latent_human = f"{dict['a1_latent'][0]}, {dict['a1_latent'][1]}"
    latent, prob =   predict_human_latent(session['dict'],
                                          session['index'],
                                          is_movers_domain)
    latent_human_predicted = f"{latent[0]}, {latent[1]}, P(x) = {prob:.2f}"
  elif mode == "collected":
    latent_human = session['latent_human_recorded'][session['index']]
  elif mode == "predicted":
    latent_human_predicted = session['latent_human_predicted'][session['index']]

  # if latent_from == 'In Game':
  #   if dict['a1_latent']:
  #     latent_human = f"{dict['a1_latent'][0]}, {dict['a1_latent'][1]}"
  # elif latent_from == 'After Game':
  #   latent_human = session['latent_human_recorded'][session['index']]
    
  # if predicted:
  #   latent_human_predicted = session['latent_human_predicted'][session['index']]
  #   latent, prob =   predict_human_latent(session['dict'],
  #                                                   session['index'],
  #                                                   is_movers_domain)
  #   latent_human_predicted = f"{latent[0]}, {latent[1]}, P(x) = {prob:.2f}"
  # else:
    
  if dict['a2_latent']:
    latent_robot = f"{dict['a2_latent'][0]}, {dict['a2_latent'][1]}"
  return latent_human, latent_human_predicted, latent_robot


class BoxPushTrajectoryConverter(Trajectories):
  def __init__(self, task_mdp: bp_mdp.BoxPushTeamMDP,
               agent_mdp: bp_mdp.BoxPushMDP) -> None:
    super().__init__(1, 2, 2, 5)
    self.task_mdp = task_mdp
    self.agent_mdp = agent_mdp

  def single_trajectory_from_list_dict(self, list_dict_state):
    self.list_np_trajectory.clear()
    np_traj = np.zeros((len(list_dict_state), self.get_width()), dtype=np.int32)
    for tidx, dict_state in enumerate(list_dict_state):
      bstt = dict_state["box_states"]
      a1pos = dict_state["a1_pos"]
      a2pos = dict_state["a2_pos"]
      a1act = EventType(
          dict_state["a1_action"]) if dict_state["a1_action"] else None
      a2act = EventType(
          dict_state["a2_action"]) if dict_state["a2_action"] else None
      a1lat = dict_state["a1_latent"]
      a2lat = dict_state["a2_latent"]

      sidx = self.task_mdp.conv_sim_states_to_mdp_sidx([bstt, a1pos, a2pos])
      aidx1 = (self.task_mdp.a1_a_space.action_to_idx[a1act]
               if a1act is not None else Trajectories.EPISODE_END)
      aidx2 = (self.task_mdp.a2_a_space.action_to_idx[a2act]
               if a2act is not None else Trajectories.EPISODE_END)

      xidx1 = (self.agent_mdp.latent_space.state_to_idx[a1lat]
               if a1lat is not None else Trajectories.EPISODE_END)
      xidx2 = (self.agent_mdp.latent_space.state_to_idx[a2lat]
               if a2lat is not None else Trajectories.EPISODE_END)

      np_traj[tidx, :] = [sidx, aidx1, aidx2, xidx1, xidx2]

    self.list_np_trajectory.append(np_traj)


def predict_human_latent(traj, index, is_movers_domain):
  # convert each
  GAME_MAP = bp_maps.EXP1_MAP
  if is_movers_domain:
    BoxPushSimulator = bp_sim.BoxPushSimulator_AlwaysTogether
    MDP_AGENT = bp_mdp.BoxPushTeamMDP_AlwaysTogether(**GAME_MAP)
    MDP_TASK = MDP_AGENT
  else:
    BoxPushSimulator = bp_sim.BoxPushSimulator_AlwaysAlone
    MDP_AGENT = bp_mdp.BoxPushAgentMDP_AlwaysAlone(**GAME_MAP)
    MDP_TASK = bp_mdp.BoxPushTeamMDP_AlwaysAlone(**GAME_MAP)

  # load models
  # TODO: take this codes out so that we need to load models only once
  model_dir = "../misc/BTIL_results/data/learned_models/"

  if is_movers_domain:
    policy_file = "exp1_team_btil_policy_human_woTx_66_1.00_a1.npy"
    tx_file = "exp1_team_btil_tx_human_66_1.00_a1.npy"
  else:
    policy_file = "exp1_indv_btil_policy_human_woTx_99_1.00_a1.npy"
    tx_file = "exp1_indv_btil_tx_human_99_1.00_a1.npy"

  policy = np.load(model_dir + policy_file)
  tx = np.load(model_dir + tx_file)

  # human mental state inference
  def policy_nxsa(nidx, xidx, sidx, tuple_aidx):
    return policy[xidx, sidx, tuple_aidx[0]]

  def Tx_nxsasx(nidx, xidx, sidx, tuple_aidx, sidx_n, xidx_n):
    return tx[xidx, sidx, tuple_aidx[0], tuple_aidx[1], xidx_n]

  def init_latent_nxs(nidx, xidx, sidx):
    return assumed_initial_mental_distribution(0, sidx, MDP_AGENT)[xidx]

  sim = BoxPushSimulator(0)
  sim.init_game(**GAME_MAP)
  trajories = BoxPushTrajectoryConverter(MDP_TASK, MDP_AGENT)
  trajories.single_trajectory_from_list_dict(traj)
  list_state, list_action, _ = trajories.get_as_column_lists(
      include_terminal=True)[0]

  inferred_x, dist_x = forward_inference(list_state[:index + 1],
                                         list_action[:index], 1,
                                         MDP_AGENT.num_latents, policy_nxsa,
                                         Tx_nxsasx, init_latent_nxs)
  latent_idx = inferred_x[0]
  prob = dist_x[0][latent_idx]
  latent = MDP_AGENT.latent_space.idx_to_state[inferred_x[0]]
  return latent, prob
  # return f"{latent[0]}, {latent[1]}, P(x) = {prob:.2f}"


def predict_human_latent_full(traj, is_movers_domain):
    GAME_MAP = bp_maps.EXP1_MAP
    if is_movers_domain:
      BoxPushSimulator = bp_sim.BoxPushSimulator_AlwaysTogether
      MDP_AGENT = bp_mdp.BoxPushTeamMDP_AlwaysTogether(**GAME_MAP)
      MDP_TASK = MDP_AGENT
    else:
      BoxPushSimulator = bp_sim.BoxPushSimulator_AlwaysAlone
      MDP_AGENT = bp_mdp.BoxPushAgentMDP_AlwaysAlone(**GAME_MAP)
      MDP_TASK = bp_mdp.BoxPushTeamMDP_AlwaysAlone(**GAME_MAP)

    # load models
    # TODO: take this codes out so that we need to load models only once
    model_dir = "../misc/BTIL_results/data/learned_models/"

    if is_movers_domain:
      policy_file = "exp1_team_btil_policy_human_woTx_66_1.00_a1.npy"
      tx_file = "exp1_team_btil_tx_human_66_1.00_a1.npy"
    else:
      policy_file = "exp1_indv_btil_policy_human_woTx_99_1.00_a1.npy"
      tx_file = "exp1_indv_btil_tx_human_99_1.00_a1.npy"

    policy = np.load(model_dir + policy_file)
    tx = np.load(model_dir + tx_file)

    # human mental state inference
    def policy_nxsa(nidx, xidx, sidx, tuple_aidx):
      return policy[xidx, sidx, tuple_aidx[0]]

    def Tx_nxsasx(nidx, xidx, sidx, tuple_aidx, sidx_n, xidx_n):
      return tx[xidx, sidx, tuple_aidx[0], tuple_aidx[1], xidx_n]

    def init_latent_nxs(nidx, xidx, sidx):
      return assumed_initial_mental_distribution(0, sidx, MDP_AGENT)[xidx]

    sim = BoxPushSimulator(0)
    sim.init_game(**GAME_MAP)
    trajories = BoxPushTrajectoryConverter(MDP_TASK, MDP_AGENT)
    trajories.single_trajectory_from_list_dict(traj)
    
    list_state, list_action, _ = trajories.get_as_column_lists(
        include_terminal=True)[0]



    list_inferred_x_seq = most_probable_sequence(list_state[:-1],
                                          list_action, 1,
                                          MDP_AGENT.num_latents, policy_nxsa,
                                          Tx_nxsasx, init_latent_nxs)
    inferred_x_seq = list_inferred_x_seq[0]
    latents = [MDP_AGENT.latent_space.idx_to_state[x] for x in inferred_x_seq]
    latents = [f"{latent[0]}, {latent[1]}" for latent in latents]
    return latents
