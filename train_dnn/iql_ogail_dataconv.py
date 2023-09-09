import os
import pickle
import numpy as np
import torch
from collections import defaultdict
from aic_ml.baselines.option_gail.utils.state_filter import StateFilter
from aic_ml.baselines.option_gail.utils.mujoco_env import load_demo
from aic_ml.baselines.IQLearn.dataset.expert_dataset import read_file


def conv_torch_trajs_2_iql_format(sar_trajectories,
                                  path: str,
                                  clip_action=False,
                                  is_last_step_done=False):
  'sa_trajectories: okay to include the terminal state'
  expert_trajs = defaultdict(list)

  for trajectory in sar_trajectories:
    s_arr, a_arr, r_arr = trajectory
    s_arr, a_arr, r_arr = s_arr.numpy(), a_arr.numpy(), r_arr.numpy()
    if clip_action:
      a_arr = np.clip(a_arr, -1, 1)

    states = s_arr[:-1]
    next_states = s_arr[1:]
    actions = a_arr[:-1]
    length = len(states)
    dones = np.zeros(length)
    if is_last_step_done:
      dones[-1] = 1
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


def get_pickle_datapath_n_traj(data_path, num_traj, env_name):
  if data_path.endswith("torch"):
    trajs, _ = load_demo(data_path, num_traj)
    data_dir = os.path.dirname(data_path)
    num_traj = len(trajs)

    data_path = os.path.join(data_dir, f"temp_{env_name}_{num_traj}.pkl")
    conv_torch_trajs_2_iql_format(trajs, data_path)

  return data_path, num_traj


def get_torch_datapath(data_path, num_traj, env_name):
  if not data_path.endswith("torch"):
    with open(data_path, 'rb') as f:
      trajs = read_file(data_path, f)
    num_traj = len(trajs["states"])

    data_dir = os.path.dirname(data_path)
    data_path = os.path.join(data_dir, f"temp_{env_name}_{num_traj}.torch")

    conv_iql_trajs_2_optiongail_format(trajs, data_path)

  return data_path


if __name__ == "__main__":
  data_path = "/home/sangwon/Projects/ai_coach/train_dnn/data/mujoco/AntPush-v0_sample.torch"
  trajs, _ = load_demo(data_path, 100000)
  data_dir = os.path.dirname(data_path)
  num_traj = len(trajs)

  data_path = os.path.join(data_dir, f"AntPush-v0_{num_traj}_clip_w_done.pkl")
  conv_torch_trajs_2_iql_format(trajs, data_path, True, True)
