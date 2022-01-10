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
  if len(sa_trajectories) == 0:
    return np.ones((num_states, num_actions)) / num_actions

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
  num_states = 5
  num_actions = 3
  trajectories = [
      [(3, 0), (1, 2), (2, 2), (2, 2), (1, 1)],
      [(0, 1), (0, 2), (1, 2), (3, 1), (2, 2), (3, 1)],
      [(3, 1), (0, 1)],
  ]

  pi = behavior_cloning(trajectories, num_states, num_actions)
  print(pi)
