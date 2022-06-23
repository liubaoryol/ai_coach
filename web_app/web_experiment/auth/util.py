from ai_coach_domain.box_push.maps import EXP1_MAP
from web_experiment import socketio
from flask import (request, session)
import web_experiment.experiment1.events_impl as event_impl
import json

def read_file(file_name):
  traj = []
  x_grid = EXP1_MAP['x_grid']
  y_grid = EXP1_MAP['y_grid']
  boxes = EXP1_MAP['boxes']
  goals = EXP1_MAP['goals']
  drops = EXP1_MAP['drops']
  walls = EXP1_MAP['walls']
  
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
        "current_step": step
      })
  return traj

def update_canvas(env_id, namespace, update_latent):
  if 'dict' in session and 'index' in session:
    dict = session['dict'][session['index']]
    event_impl.update_html_canvas(dict, env_id, False, namespace)
    objs = {}
    latent_human, latent_human_predicted, latent_robot = get_latent_states()
    # update latent states
    if update_latent:
      objs['latent_human'] = latent_human
      objs['latent_robot'] = latent_robot
      objs['latent_human_predicted'] = latent_human_predicted
      objs_json = json.dumps(objs)
      str_emit = 'update_latent'
      socketio.emit(str_emit, objs_json, room=env_id, namespace = namespace)

def get_latent_states():
  dict = session['dict'][session['index']]
  latent_human = "None"
  latent_robot = "None"
  latent_human_predicted = session['latent_human_predicted'][session['index']]
  if dict['a1_latent']:
    latent_human = f"{dict['a1_latent'][0]}, {dict['a1_latent'][1]}"
  if dict['a2_latent']:
    latent_robot = f"{dict['a2_latent'][0]}, {dict['a2_latent'][1]}"
  return latent_human, latent_human_predicted, latent_robot
