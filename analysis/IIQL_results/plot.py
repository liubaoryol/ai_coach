import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np


def fill_na(res):

  idx = -1
  val = None
  # find the closest value
  for i in range(len(res)):
    if not np.isnan(res[i]):
      idx = i
      val = res[i]
      break

  res[:idx] = res[idx]
  for i in range(idx, len(res)):
    if np.isnan(res[i]):
      res[i] = val
    else:
      val = res[i]

  return res


def plot(cur_dir,
         domain_name,
         title,
         max_step,
         ogail_names,
         iql_names,
         iiql_names,
         oiql_names,
         down_sampling=1):
  csv_files = glob.glob(os.path.join(cur_dir, domain_name + "_*.csv"))

  csv_files = sorted(csv_files)

  data_frames = []
  labels = []
  names = {}

  for csv_file in csv_files:
    if 'gail' in csv_file:
      labels.append('OGAIL')
      names[labels[-1]] = ogail_names
    elif 'iiql' in csv_file:
      labels.append('IDIL')
      names[labels[-1]] = iiql_names
    elif 'oiql' in csv_file:
      labels.append('IDIL-J')
      names[labels[-1]] = oiql_names
    elif 'iql' in csv_file:
      labels.append('IQL')
      names[labels[-1]] = iql_names
    df = pd.read_csv(csv_file)
    if 'global_step' in df.columns:
      data_frames.append(df)

  epi_steps = []
  means = []
  stds = []
  for idx, df in enumerate(data_frames):

    res_group = []
    for run_name in names[labels[idx]]:
      res = df[run_name + ' - eval/episode_reward']
      res = fill_na(res.values)
      res_group.append(res)
      epi_step = df['global_step'].values

    res_group = np.array(res_group)
    mean_res = np.mean(res_group, axis=0)
    std_res = np.std(res_group, axis=0)

    means.append(mean_res)
    stds.append(std_res)
    epi_steps.append(epi_step)

  plt.figure(figsize=(10, 6))
  for idx in range(len(means)):
    mean_res = means[idx][::down_sampling]
    std_res = stds[idx][::down_sampling]
    epi_step = epi_steps[idx][::down_sampling]
    plt.plot(epi_step, mean_res, label=labels[idx])
    plt.fill_between(epi_step,
                     mean_res - std_res,
                     mean_res + std_res,
                     alpha=0.2)

  plt.xlabel('Exploration Steps', fontsize=16)
  plt.ylabel('Task Reward', fontsize=16)
  plt.xlim(0, max_step)
  plt.title(title, fontsize=16)
  plt.legend()
  plt.show()


if __name__ == "__main__":

  cur_dir = '/home/sangwon/Projects/ai_coach/analysis/IIQL_results/data/'

  if True:
    ogail_names = ['ogail_sep19', 'ogail_oct3seed1', 'ogail_oct3seed2']
    iql_names = ['iql_oct8seed0', 'iql_oct8seed1', 'iql_oct8seed2']
    iiql_names = ['miql_sep24', 'miql_Ttx1Tpi001', 'miql_oct7seed2']
    oiql_names = ['oiql_oct3seed1', 'oiql_1tjv0T001', 'oiql_oct3seed2']
    plot(cur_dir, 'hopper', 'Hopper', 1.e6, ogail_names, iql_names, iiql_names,
         oiql_names, 1)

  if True:
    ogail_names = ['ogail_sep19', 'ogail_oct3seed1', 'ogail_oct3seed2']
    iql_names = ['iql_oct3seed1', 'iql_oct3seed2', 'iql_sep19']
    iiql_names = ['miql_oct7seed0', 'miql_oct7seed2', 'miql_oct7seed1']
    oiql_names = ['oiql_sep21', 'oiql_oct3seed1', 'oiql_oct3seed2']
    plot(cur_dir, 'walker', 'Walker', 3e6, ogail_names, iql_names, iiql_names,
         oiql_names, 1)

  if True:
    ogail_names = ['ogail_oct3seed2', 'ogail_oct3seed1', 'ogail_sep19']
    iql_names = ['iql_sep19', 'iql_oct3seed2', 'iql_oct3seed1']
    iiql_names = ['miql_oct7seed0', 'miql_oct7seed1', 'miql_oct7seed2']
    oiql_names = ['oiql_sep21', 'oiql_oct3seed2', 'oiql_10tjv0T03']
    plot(cur_dir, 'humanoid', 'Humanoid', 2.5e6, ogail_names, iql_names,
         iiql_names, oiql_names, 1)

  if True:
    ogail_names = ['ogail_oct3seed2', 'ogail_oct3seed1', 'ogail_clip']
    iql_names = ['iql_oct3seed2', 'iql_oct3seed1', 'iql_sep19']
    iiql_names = ['miql_oct7seed2', 'miql_Ttx001Tpi0001', 'miql_sep24']
    oiql_names = ['oiql_sep21', 'oiql_oct3seed2', 'oiql_oct3seed1']
    plot(cur_dir, 'antpush', 'AntPush', 8e5, ogail_names, iql_names, iiql_names,
         oiql_names, 1)

  if True:
    ogail_names = ['ogail_seed2Sv0', 'ogail_seed1Sv0', 'ogail_tol5Sv2']
    iql_names = ['iql_seed1Sv0', 'iql_seed2Sv0', 'iql_Tpi001tol5Sv0']
    iiql_names = ['miql_seed2Sv0', 'miql_seed1Sv0', 'miql_Ttx001Tpi001tol5Sv0']
    oiql_names = ['oiql_seed2Sv0', 'oiql_seed1Sv0', 'oiql_T001tol5Sv2']
    plot(cur_dir, 'mg5', 'MultiGoals-5', 5e5, ogail_names, iql_names,
         iiql_names, oiql_names, 1)