# from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os


def conv_tfevent_csv(path_tfevent, save_path):
  size_guide = {
      event_accumulator.COMPRESSED_HISTOGRAMS: 500,
      event_accumulator.IMAGES: 4,
      event_accumulator.AUDIO: 4,
      event_accumulator.SCALARS: 0,
      event_accumulator.HISTOGRAMS: 1,
  }

  ea = event_accumulator.EventAccumulator(path, size_guidance=size_guide)
  ea.Reload()
  tags = ea.Tags()
  if 'Test/r-avg' in tags["scalars"]:
    pd.DataFrame(ea.Scalars('Test/r-avg')).to_csv(save_name)
    print('saved from Test/r-avg tag')
  elif 'eval/episode_reward' in tags["scalars"]:
    pd.DataFrame(ea.Scalars('eval/episode_reward')).to_csv(save_name)
    print('saved from eval/episode_reward tag')


def read_cvs(path_csv):
  return pd.read_csv(path_csv)


def get_df_historical_best(df_data, sample_step, max_step):
  min_val = min(df_data['value'])
  list_history_best = []
  for step in range(0, max_step, sample_step):
    partial_df = df_data[df_data['step'] <= step]['value']
    if len(partial_df) > 0:
      max_val = max(df_data[df_data['step'] <= step]['value'])
    else:
      max_val = min_val
    list_history_best.append((step, max_val))
  return list_history_best


if __name__ == "__main__":
  cur_dir = os.path.dirname(__file__)

  save_name = os.path.join(cur_dir, "tfevent_test.csv")

  # path = "/home/sangwon/Projects/ai_coach/test_algs/result/CleanupSingle-v0/oiql/oiqlstrm_256_3e-5_boundnnstd_extraD/2023-08-14_19-09-42/log/oiql/events.out.tfevents.1692058183.sangwon-XPS-15-9500.80313.0"
  path = "/home/sangwon/Projects/ai_coach/test_algs/result/CleanupSingle-v0/ogail/ogail_64_3e-5_value/2023-08-15_11-10-52/log/events.out.tfevents.1692115852.sangwon-XPS-15-9500.16042.0"

  conv_tfevent_csv(path, save_name)
  df_data = read_cvs(save_name)
  # print(df_data[df_data['step'] < 60000])
  # print(df_data['value'])
  print(get_df_historical_best(df_data, 5000, 100001))
  print('best', max(df_data['value']))