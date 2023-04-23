import os
import pickle
import torch
import numpy as np
from collections import defaultdict
from aicoach_baselines.option_gail.utils.config import ARGConfig
from ai_coach_core.model_learning.LatentIQL.train_mental_iql import (
    train_mental_iql)
from aicoach_baselines.option_gail.utils.mujoco_env import load_demo


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


def learn(config: ARGConfig):

  log_dir, output_dir, sample_name = get_dirs(config.seed, config.base_dir,
                                              "miql", config.env_type,
                                              config.env_name)
  trajs, _ = load_demo(sample_name, config.n_demo)
  data_dir = os.path.dirname(sample_name)
  num_traj = len(trajs)

  path_iq_data = os.path.join(data_dir, f"{config.env_name}_{num_traj}.pkl")
  conv_torch_trajs_2_iql_format(trajs, path_iq_data)

  n_sample = config.n_sample
  n_step = 10
  batch_size = config.mini_batch_size
  clip_grad_val = 0.5
  learn_alpha = True

  update_per_epoch = int(n_sample / batch_size) * n_step
  num_iter = config.n_epoch * update_per_epoch
  log_interval = update_per_epoch
  eval_interval = 20 * update_per_epoch

  train_mental_iql(config.env_name, {},
                   config.seed,
                   batch_size,
                   config.dim_c,
                   path_iq_data,
                   num_traj,
                   log_dir,
                   output_dir,
                   replay_mem=n_sample,
                   initial_mem=n_sample,
                   eps_window=10,
                   num_learn_steps=num_iter,
                   log_interval=log_interval,
                   eval_interval=eval_interval,
                   list_hidden_dims=config.hidden_policy,
                   clip_grad_val=clip_grad_val,
                   learn_alpha=learn_alpha,
                   learning_rate=config.optimizer_lr_policy,
                   gumbel_temperature=1.0,
                   bounded_actor=config.bounded_actor)
