import random
import numpy as np
import torch
from aic_ml.MentalIQL.train_miql import load_expert_data_w_labels
import pickle

if __name__ == "__main__":
  seed = 0

  # set seeds
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  demo_dir = "/home/sangwon/Projects/ai_coach/train_dnn/test_data/"
  demo_file = demo_dir + "EnvMovers_v0_66.pkl"

  expert_dataset, traj_labels, cnt_label = load_expert_data_w_labels(
      demo_file, 66, 0, seed)

  trajectories = expert_dataset.trajectories
  print(trajectories["rewards"][0])
  rewards = []
  n_traj = len(trajectories["rewards"])
  for i in range(n_traj):
    rewards.append([-1] * len(trajectories["rewards"][i]))

  trajectories["rewards"] = rewards
  print(trajectories["rewards"][0])

  file_path = demo_file
  with open(demo_file, 'wb') as f:
    pickle.dump(trajectories, f)

  # n_expert_trj = len(trajectories["rewards"])
  # print(trajectories["rewards"][0])

  # print(f'Demo reward: {expert_return_avg} +- {expert_return_std}')
