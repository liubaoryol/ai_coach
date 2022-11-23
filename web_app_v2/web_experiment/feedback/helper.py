import os
import time
import glob
from flask import current_app


def store_user_fix_locally(user_id, session_name, user_fixes):
  file_name = get_user_fix_file_name(user_id, session_name)
  dir_path = os.path.dirname(file_name)
  if dir_path != '' and not os.path.exists(dir_path):
    os.makedirs(dir_path)

  with open(file_name, 'w', newline='') as txtfile:
    # sequence
    txtfile.write('# time_step, fixed_user_label\n')

    for key in user_fixes:
      txtfile.write('%d; %s;' % (key, user_fixes[key]))
      txtfile.write('\n')


def get_user_fix_file_name(user_id, session_name):
  traj_dir = os.path.join(current_app.config["USER_LABEL_PATH"], user_id)

  # save somewhere
  if not os.path.exists(traj_dir):
    os.makedirs(traj_dir)

  sec, msec = divmod(time.time() * 1000, 1000)
  time_stamp = '%s.%03d' % (time.strftime('%Y-%m-%d_%H_%M_%S',
                                          time.gmtime(sec)), msec)
  file_name = ('user_fix_' + session_name + '_' + str(user_id) + '_' +
               time_stamp + '.txt')
  return os.path.join(traj_dir, file_name)


def store_latent_locally(user_id, session_name, lstates):
  # # if latent state has not been previously stored
  # if not check_latent_exist(user_id, session_name):
  file_name = get_latent_file_name(user_id, session_name)
  header = session_name + "\n"
  header += "User ID: %s\n" % (str(user_id), )

  dir_path = os.path.dirname(file_name)
  if dir_path != '' and not os.path.exists(dir_path):
    os.makedirs(dir_path)

  with open(file_name, 'w', newline='') as txtfile:
    # sequence
    txtfile.write(header)
    txtfile.write('\n')
    txtfile.write('# cur_step, human_latent\n')

    for idx, lstate in enumerate(lstates):
      txtfile.write('%d; ' % (idx, ))
      txtfile.write('%s; ' % (lstate, ))
      txtfile.write('\n')


def get_latent_file_name(user_id, session_name):
  traj_dir = os.path.join(current_app.config["LATENT_PATH"], user_id)
  # save somewhere
  if not os.path.exists(traj_dir):
    os.makedirs(traj_dir)

  sec, msec = divmod(time.time() * 1000, 1000)
  time_stamp = '%s.%03d' % (time.strftime('%Y-%m-%d_%H_%M_%S',
                                          time.gmtime(sec)), msec)
  file_name = "lstates" + "_" + session_name + '_' + str(
      user_id) + '_' + time_stamp + '.txt'
  return os.path.join(traj_dir, file_name)


def load_latent(user_id, session_name):
  if check_latent_exist(user_id, session_name):
    files = get_latent_files(user_id, session_name)
    files = sorted(files, reverse=True)
    result = read_latent_file(files[0])
    return result


"""
  Input is a regular expression of the latent file to read
"""


def get_latent_files(user_id, session_name):
  traj_path = current_app.config["LATENT_PATH"]
  path = f"{user_id}/lstates_{session_name}_{user_id}*.txt"

  fileExpr = os.path.join(traj_path, path)

  # find any matching files
  files = glob.glob(fileExpr)
  return files


"""
  Check if latent state is already recorded and stored in a file.
"""


def check_latent_exist(user_id, session_name):
  return (len(get_latent_files(user_id, session_name)) > 0)


def read_latent_file(file_name):
  lstates = []

  with open(file_name, newline='') as txtfile:
    lines = txtfile.readlines()
    i_start = 0
    for i_r, row in enumerate(lines):
      if row == ('# cur_step, human_latent\n'):
        i_start = i_r
        break
    print(i_start)

    for i_r in range(i_start + 1, len(lines)):
      line = lines[i_r]
      states = line.rstrip()[:-1].split("; ")
      if len(states) < 2:
        for dummy in range(2 - len(states)):
          states.append(None)

      step, lstate = states
      lstates.append(lstate)
    print(lstates)
  return lstates
