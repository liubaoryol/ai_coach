import os
import json
import h5py
import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import test_robomimic_env as rmenv
import torch

if __name__ == "__main__":

  robomimic_path = '/home/sangwon/Projects/external/robomimic/'
  data_path = os.path.join(robomimic_path, "datasets/can/ph/low_dim_v141.hdf5")
  data_file = h5py.File(data_path, "r")

  # each demonstration is a group under "data"
  demos = list(data_file["data"].keys())
  num_demos = len(demos)

  # each demonstration is named "demo_#" where # is a number.
  # Let's put the demonstration list in increasing episode order
  inds = np.argsort([int(elem[5:]) for elem in demos])
  demos = [demos[i] for i in inds]

  returns = []
  phase_change = []
  phase_freq = []
  for ep in demos:
    demo_grp = data_file["data/{}".format(ep)]

    num_samples = data_file["data/{}/actions".format(ep)].shape[0]
    num_samples_2 = demo_grp.attrs["num_samples"]
    obs_dict = dict()
    next_obs_dict = dict()
    demo_grp_obs = demo_grp['obs']
    for k in demo_grp_obs:
      obs_dict[k] = demo_grp["obs"][k][:]  # numpy array
      next_obs_dict[k] = demo_grp["next_obs"][k][:]  # numpy array
    # TODO: convert obs, next_obs
    obs = rmenv.FlatteningEnvWrapper.flatten_obs_dict(obs_dict)
    next_obs = rmenv.FlatteningEnvWrapper.flatten_obs_dict(next_obs_dict)

    actions = demo_grp["actions"][:]
    dones = demo_grp["dones"][:]
    rewards = demo_grp["rewards"][:]
    returns.append(np.sum(rewards))

    prev_action = actions[0]
    changes = []
    for idx, action in enumerate(actions[1:]):
      if action[6] != prev_action[6]:
        changes.append(idx + 1)
      prev_action = action
    phase_change.append(tuple(changes))
    num_changes = len(changes)
    phase_freq.append(num_changes)
    if num_changes != 2:
      print(num_changes, ep, num_samples)

    # itemidx = np.where(actions[:, 6] == 1.0)
    # itemidx = itemidx[0][0]
    # phase_change.append(itemidx)

  print(phase_freq)
  print(phase_change)
  print(phase_freq.index(4))
  # print(np.mean(returns))
  # print(returns)
