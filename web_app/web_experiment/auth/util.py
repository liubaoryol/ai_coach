from ai_coach_domain.box_push.maps import EXP1_MAP

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
