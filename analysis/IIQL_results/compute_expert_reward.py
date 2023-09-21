import random
import numpy as np
import torch
from aic_ml.MentalIQL.train_miql import load_expert_data_w_labels

if __name__ == "__main__":
  seed = 0

  # set seeds
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  demo_dir = "/home/sangwon/Projects/ai_coach/train_dnn/experts/"
  demo_file = demo_dir + "AntPush-v0_400_original.pkl"

  expert_dataset, traj_labels, cnt_label = load_expert_data_w_labels(
      demo_file, 253, 0, seed)

  trajectories = expert_dataset.trajectories

  n_expert_trj = len(trajectories["rewards"])
  print(trajectories["rewards"][0])

  # print(f'Demo reward: {expert_return_avg} +- {expert_return_std}')
