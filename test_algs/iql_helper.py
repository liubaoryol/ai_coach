import os
import pickle
import numpy as np
import torch
import datetime
import time
from collections import defaultdict
from aicoach_baselines.option_gail.utils.state_filter import StateFilter


def get_dirs(base_dir="",
             exp_type="gail",
             env_type="mujoco",
             env_name="HalfCheetah-v2",
             msg="default"):
  assert env_type in ("mini", "mujoco",
                      "rlbench"), f"Error, env_type {env_type} not supported"

  base_log_dir = os.path.join(base_dir, "result/")

  ts_str = datetime.datetime.fromtimestamp(
      time.time()).strftime("%Y-%m-%d_%H-%M-%S")
  log_dir_root = os.path.join(base_log_dir, env_name, exp_type, msg, ts_str)
  save_dir = os.path.join(log_dir_root, "model")
  log_dir = os.path.join(log_dir_root, "log")
  os.makedirs(save_dir)
  os.makedirs(log_dir)

  return log_dir, save_dir, log_dir_root


def conv_torch_trajs_2_iql_format(sar_trajectories, path: str):
  'sa_trajectories: okay to include the terminal state'
  expert_trajs = defaultdict(list)

  for trajectory in sar_trajectories:
    s_arr, a_arr, r_arr = trajectory
    s_arr, a_arr, r_arr = s_arr.numpy(), a_arr.numpy(), r_arr.numpy()

    states = s_arr[:-1]
    next_states = s_arr[1:]
    actions = a_arr[:-1]
    length = len(states)
    dones = np.zeros(length)
    rewards = r_arr[:-1]

    expert_trajs["states"].append(states.reshape(length, -1))
    expert_trajs["next_states"].append(next_states.reshape(length, -1))
    expert_trajs["actions"].append(actions.reshape(length, -1))
    expert_trajs["rewards"].append(rewards)
    expert_trajs["dones"].append(dones)
    expert_trajs["lengths"].append(length)

  with open(path, 'wb') as f:
    pickle.dump(expert_trajs, f)


def conv_iql_trajs_2_optiongail_format(trajectories, path: str):

  use_rs = False
  num_traj = len(trajectories["states"])
  rs = StateFilter(enable=use_rs)

  sample = []
  for epi in range(num_traj):
    n_steps = len(trajectories["rewards"][epi])
    s_array = torch.as_tensor(trajectories["states"][epi],
                              dtype=torch.float32).reshape(n_steps, -1)
    a_array = torch.as_tensor(trajectories["actions"][epi],
                              dtype=torch.float32).reshape(n_steps, -1)
    r_array = torch.as_tensor(trajectories["rewards"][epi],
                              dtype=torch.float32).reshape(n_steps, -1)
    if "latents" in trajectories:
      x_array = torch.as_tensor(trajectories["latents"][epi],
                                dtype=torch.float32).reshape(n_steps, -1)
    else:
      x_array = None

    sample.append((s_array, a_array, r_array, x_array))

  torch.save((sample, rs.state_dict()), path)
