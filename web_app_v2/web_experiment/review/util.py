from typing import Sequence, Optional
from dataclasses import dataclass
import numpy as np
import json
import os
import glob
from flask import current_app
from flask_socketio import emit
from ai_coach_core.latent_inference.decoding import (forward_inference,
                                                     most_probable_sequence)
from ai_coach_core.utils.data_utils import Trajectories
from ai_coach_domain.box_push.agent_model import (
    assumed_initial_mental_distribution)

from ai_coach_domain.box_push_v2 import get_possible_latent_states
from ai_coach_domain.box_push_v2.maps import MAP_MOVERS, MAP_CLEANUP
from ai_coach_domain.box_push_v2.mdp import (MDP_BoxPushV2, MDP_Cleanup_Agent,
                                             MDP_Cleanup_Task, MDP_Movers_Agent,
                                             MDP_Movers_Task)
from ai_coach_domain.box_push_v2.simulator import BoxPushSimulatorV2
from ai_coach_domain.rescue.simulator import RescueSimulator
from ai_coach_domain.rescue.maps import MAP_RESCUE

from web_experiment.define import EMode, EDomainType, get_domain_type
import web_experiment.exp_common.events_impl as event_impl
from web_experiment.exp_common.page_base import CanvasPageBase, CanvasPageError
from web_experiment.exp_common.page_replay import UserDataReplay


def load_trajectory(session_name, id):
  traj_path = current_app.config["TRAJECTORY_PATH"]
  path = f"{id}/{session_name}_{id}*.txt"
  fileExpr = os.path.join(traj_path, path)
  # find any matching files
  files = glob.glob(fileExpr)
  if len(files) == 0:
    # # does not find a match, error handling
    # error = f"No file found that matches {id}, {session_name}"
    return None
  else:
    files.sort(reverse=True)
    file = files[0]

    return read_file(file, get_domain_type(session_name))


def latent_state_from_traj(traj, domain_type: EDomainType):
  list_latents = []

  for scene in traj:
    a1_latent = scene['a1_latent']
    if a1_latent is not None:
      str_latent = str(a1_latent)
      list_latents.append(str_latent)
    else:
      str_latent = "None"
      list_latents.append(str_latent)

  return list_latents


def possible_latent_states(domain_type: EDomainType):
  list_latents = []
  if domain_type == EDomainType.Movers:
    num_drops = len(MAP_MOVERS["drops"])
    num_goals = len(MAP_MOVERS["goals"])
    num_boxes = len(MAP_MOVERS["boxes"])
    list_latents = get_possible_latent_states(num_boxes, num_drops, num_goals)
  elif domain_type == EDomainType.Cleanup:
    num_drops = len(MAP_CLEANUP["drops"])
    num_goals = len(MAP_CLEANUP["goals"])
    num_boxes = len(MAP_CLEANUP["boxes"])
    list_latents = get_possible_latent_states(num_boxes, num_drops, num_goals)
  elif domain_type == EDomainType.Rescue:
    num_works = len(MAP_RESCUE["work_locations"])
    list_latents = list(range(num_works))

  return [str(item) for item in list_latents]


def read_file(file_name, domain_type: EDomainType):
  traj_of_dict = []

  if domain_type == EDomainType.Movers:
    traj_of_list = BoxPushSimulatorV2.read_file(file_name)
    for step, elem in enumerate(traj_of_list):
      box_state, a1_pos, a2_pos, a1_act, a2_act, a1_lat, a2_lat = elem
      traj_of_dict.append({
          "box_states": box_state,
          "a1_pos": a1_pos,
          "a2_pos": a2_pos,
          "a1_latent": a1_lat,
          "a2_latent": a2_lat,
          "current_step": step,
          "a1_action": a1_act,
          "a2_action": a2_act,
      })

  elif domain_type == EDomainType.Cleanup:
    traj_of_list = BoxPushSimulatorV2.read_file(file_name)
    for step, elem in enumerate(traj_of_list):
      box_state, a1_pos, a2_pos, a1_act, a2_act, a1_lat, a2_lat = elem
      traj_of_dict.append({
          "box_states": box_state,
          "a1_pos": a1_pos,
          "a2_pos": a2_pos,
          "a1_latent": a1_lat,
          "a2_latent": a2_lat,
          "current_step": step,
          "a1_action": a1_act,
          "a2_action": a2_act,
      })
  elif domain_type == EDomainType.Rescue:
    traj_of_list = RescueSimulator.read_file(file_name)
    for step, elem in enumerate(traj_of_list):
      score, work_state, a1_pos, a2_pos, a1_act, a2_act, a1_lat, a2_lat = elem
      traj_of_dict.append({
          "score": score,
          "work_states": work_state,
          "a1_pos": a1_pos,
          "a2_pos": a2_pos,
          "a1_latent": a1_lat,
          "a2_latent": a2_lat,
          "current_step": step,
          "a1_action": a1_act,
          "a2_action": a2_act,
      })
  else:
    raise NotImplementedError

  return traj_of_dict


@dataclass
class SessionData:
  '''
    socketio session data whose life cycle is intended to be
    from "connect" to "disconnect" of socketio
  '''
  user_id: str
  user_data: UserDataReplay
  session_name: str
  trajectory: Sequence
  index: int = 0
  groupid: Optional[str] = None
  latent_collected: Optional[Sequence] = None
  latent_predicted: Optional[Sequence] = None


def no_trajectory_page(sid, name_space, text):
  page = CanvasPageError(text)
  (commands, drawing_objs, drawing_order,
   animations) = page.get_updated_drawing_info(None)
  event_impl.update_gamedata(commands=commands,
                             drawing_objects=drawing_objs,
                             drawing_order=drawing_order,
                             animations=animations)


def update_canvas(sid,
                  name_space,
                  page: CanvasPageBase,
                  session_data: SessionData,
                  init_imgs=False,
                  domain_type: EDomainType = None):
  imgs = None
  if init_imgs:
    imgs = event_impl.get_imgs(domain_type)

  drawing_info = None
  if session_data is not None:
    user_data = session_data.user_data
    user_data.data[UserDataReplay.TRAJECTORY] = session_data.trajectory
    user_data.data[UserDataReplay.TRAJ_IDX] = session_data.index
    page.init_user_data(user_data)
    drawing_info = page.get_updated_drawing_info(user_data)

  if imgs is None and drawing_info is None:
    return

  commands, drawing_objs, drawing_order, animations = None, None, None, None
  if drawing_info is not None:
    commands, drawing_objs, drawing_order, animations = drawing_info
  event_impl.update_gamedata(commands=commands,
                             imgs=imgs,
                             drawing_objects=drawing_objs,
                             drawing_order=drawing_order,
                             animations=animations)


def canvas_button_clicked(sid, name_space, button, page: CanvasPageBase,
                          session_data: SessionData):
  if session_data is None:
    return

  user_data = session_data.user_data
  user_data.data[UserDataReplay.TRAJECTORY] = session_data.trajectory
  user_data.data[UserDataReplay.TRAJ_IDX] = session_data.index

  page.init_user_data(user_data)

  page.button_clicked(user_data, button)

  drawing_info = page.get_updated_drawing_info(user_data, button)
  commands, drawing_objs, drawing_order, animations = None, None, None, None
  if drawing_info is not None:
    commands, drawing_objs, drawing_order, animations = drawing_info
  event_impl.update_gamedata(commands=commands,
                             drawing_objects=drawing_objs,
                             drawing_order=drawing_order,
                             animations=animations)


def update_latent_state(domain_type: EDomainType, mode: EMode,
                        session_data: SessionData):
  latent_human, latent_human_predicted, latent_robot = get_latent_states(
      domain_type, mode=mode, session_data=session_data)
  objs = {}
  objs['latent_human'] = latent_human
  objs['latent_robot'] = latent_robot
  objs['latent_human_predicted'] = latent_human_predicted

  if mode == EMode.Replay:
    objs['latent_states'] = "replay"
  elif mode == EMode.Predicted:
    objs['latent_states'] = "predicted"
  elif mode == EMode.Collected:
    objs['latent_states'] = "collected"
  objs_json = json.dumps(objs)
  str_emit = 'update_latent'
  emit(str_emit, objs_json)


def get_latent_states(domain_type: EDomainType, mode: EMode,
                      session_data: SessionData):
  dict = session_data.trajectory[session_data.index]
  latent_human = "None"
  latent_robot = "None"
  latent_human_predicted = "None"
  if mode == EMode.Replay:
    if dict['a1_latent']:
      latent_human = str(dict["a1_latent"])
    latent, prob = predict_human_latent(session_data.trajectory,
                                        session_data.index, domain_type)
    latent_human_predicted = str(latent) + f", P(x) = {prob:.2f}"
  elif mode == EMode.Collected:
    if session_data.latent_collected is not None:
      latent_human = session_data.latent_collected[session_data.index]
  elif mode == EMode.Predicted:
    if session_data.latent_predicted is not None:
      latent_human_predicted = session_data.latent_predicted[session_data.index]

  if dict['a2_latent']:
    latent_robot = str(dict['a2_latent'])
  return latent_human, latent_human_predicted, latent_robot


class BoxPushTrajectoryConverter(Trajectories):
  def __init__(self, task_mdp: MDP_BoxPushV2, agent_mdp: MDP_BoxPushV2) -> None:
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
      a1act = dict_state["a1_action"]
      a2act = dict_state["a2_action"]
      a1lat = dict_state["a1_latent"]
      a2lat = dict_state["a2_latent"]

      sidx = self.task_mdp.conv_sim_states_to_mdp_sidx([bstt, a1pos, a2pos])
      aidx1 = (a1act if a1act is not None else Trajectories.EPISODE_END)
      aidx2 = (a2act if a2act is not None else Trajectories.EPISODE_END)

      if a1lat is None:
        xidx1 = Trajectories.EPISODE_END
      elif a1lat[0] == "NA":
        xidx1 = Trajectories.DUMMY
      else:
        xidx1 = self.agent_mdp.latent_space.state_to_idx[a1lat]

      if a2lat is None:
        xidx2 = Trajectories.EPISODE_END
      elif a2lat[0] == "NA":
        xidx2 = Trajectories.DUMMY
      else:
        xidx2 = self.agent_mdp.latent_space.state_to_idx[a2lat]

      np_traj[tidx, :] = [sidx, aidx1, aidx2, xidx1, xidx2]

    self.list_np_trajectory.append(np_traj)


def get_mdp_policy_tx(domain_type: EDomainType):
  # load models
  # TODO: take this codes out so that we need to load models only once
  model_dir = "../misc/BTIL_results/data/learned_models/"

  if domain_type == EDomainType.Movers:
    game_map = MAP_MOVERS
    mdp_agent = MDP_Movers_Agent(**game_map)
    mdp_task = MDP_Movers_Task(**game_map)
    policy_file = "movers_v2_btil_policy_synth_woTx_200_1.00_a1.npy"
    tx_file = "movers_v2_btil_tx_synth_200_1.00_a1.npy"
  elif domain_type == EDomainType.Cleanup:
    game_map = MAP_CLEANUP
    mdp_agent = MDP_Cleanup_Agent(**game_map)
    mdp_task = MDP_Cleanup_Task(**game_map)
    policy_file = "cleanup_v2_btil_policy_synth_woTx_200_1.00_a1.npy"
    tx_file = "cleanup_v2_btil_tx_synth_200_1.00_a1.npy"
  else:
    raise NotImplementedError

  policy = np.load(model_dir + policy_file)
  tx = np.load(model_dir + tx_file)

  return mdp_agent, mdp_task, policy, tx


def predict_human_latent(traj, index, domain_type: EDomainType):
  mdp_agent, mdp_task, policy, tx = get_mdp_policy_tx(domain_type)

  # human mental state inference
  def policy_nxsa(nidx, xidx, sidx, tuple_aidx):
    return policy[xidx, sidx, tuple_aidx[0]]

  def Tx_nxsasx(nidx, xidx, sidx, tuple_aidx, sidx_n, xidx_n):
    return tx[xidx, sidx, tuple_aidx[0], tuple_aidx[1], xidx_n]

  def init_latent_nxs(nidx, xidx, sidx):
    return assumed_initial_mental_distribution(0, sidx, mdp_agent)[xidx]

  trajories = BoxPushTrajectoryConverter(mdp_task, mdp_agent)
  trajories.single_trajectory_from_list_dict(traj)
  list_state, list_action, _ = trajories.get_as_column_lists(
      include_terminal=True)[0]

  inferred_x, dist_x = forward_inference(list_state[:index + 1],
                                         list_action[:index], 1,
                                         mdp_agent.num_latents, policy_nxsa,
                                         Tx_nxsasx, init_latent_nxs)
  latent_idx = inferred_x[0]
  prob = dist_x[0][latent_idx]
  latent = mdp_agent.latent_space.idx_to_state[inferred_x[0]]
  return latent, prob


def predict_human_latent_full(traj, domain_type: EDomainType):
  mdp_agent, mdp_task, policy, tx = get_mdp_policy_tx(domain_type)

  # human mental state inference
  def policy_nxsa(nidx, xidx, sidx, tuple_aidx):
    return policy[xidx, sidx, tuple_aidx[0]]

  def Tx_nxsasx(nidx, xidx, sidx, tuple_aidx, sidx_n, xidx_n):
    return tx[xidx, sidx, tuple_aidx[0], tuple_aidx[1], xidx_n]

  def init_latent_nxs(nidx, xidx, sidx):
    return assumed_initial_mental_distribution(0, sidx, mdp_agent)[xidx]

  trajories = BoxPushTrajectoryConverter(mdp_task, mdp_agent)
  trajories.single_trajectory_from_list_dict(traj)

  list_state, list_action, _ = trajories.get_as_column_lists(
      include_terminal=True)[0]

  list_inferred_x_seq = most_probable_sequence(list_state[:-1], list_action, 1,
                                               mdp_agent.num_latents,
                                               policy_nxsa, Tx_nxsasx,
                                               init_latent_nxs)
  inferred_x_seq = list_inferred_x_seq[0]
  latents = [mdp_agent.latent_space.idx_to_state[x] for x in inferred_x_seq]
  latents = [str(latent) for latent in latents]
  return latents
