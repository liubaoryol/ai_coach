import os
import glob
import numpy as np


def read_trajectory(file_name):
  traj = []
  with open(file_name, newline='') as txtfile:
    lines = txtfile.readlines()
    for i_r in range(1, len(lines)):
      line = lines[i_r]
      row_elem = [int(elem) for elem in line.rstrip().split(", ")]
      state = row_elem[0]
      action = row_elem[1]
      traj.append((state, action))
  return traj


def behavior_cloning(sa_trajectories, num_states, num_actions):

  pi = np.zeros((num_states, num_actions))

  for traj in sa_trajectories:
    for s, a in traj:
      pi[s, a] += 1

  sum_pi = np.sum(pi, axis=1)
  # find 0-row
  mask_zero_row = sum_pi == 0
  sum_pi[mask_zero_row] = 1
  pi = pi / sum_pi[..., None]
  pi[mask_zero_row] = 1.0 / num_actions

  return pi


if __name__ == "__main__":

  trajectories = []
  len_sum = 0
  file_names = glob.glob(
      os.path.join("tests/data/irl_toy_trajectories/", '*.txt'))
  for file_nm in file_names:
    traj = read_trajectory(file_nm)
    len_sum += len(traj)
    trajectories.append(traj)

  pi = behavior_cloning(trajectories, 45, 10)
  # print(pi)
