import os
import pickle
import numpy as np
from collections import defaultdict


def get_dirs(seed,
             base_dir="",
             exp_type="gail",
             env_type="mujoco",
             env_name="HalfCheetah-v2"):
  assert env_type in ("mini", "mujoco",
                      "rlbench"), f"Error, env_type {env_type} not supported"

  base_log_dir = os.path.join(base_dir, "result/")
  base_data_dir = os.path.join(base_dir, "data/")
  rand_str = f"{seed}"

  sample_name = os.path.join(base_data_dir, env_type,
                             f"{env_name}_sample.torch")

  log_dir_root = os.path.join(base_log_dir, env_name, f"{exp_type}", rand_str)
  output_dir = os.path.join(log_dir_root, "model")
  log_dir = os.path.join(log_dir_root, "log")
  os.makedirs(output_dir)
  os.makedirs(log_dir)

  return log_dir, output_dir, sample_name


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
