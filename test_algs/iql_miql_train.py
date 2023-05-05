from aicoach_baselines.option_gail.utils.config import ARGConfig
from ai_coach_core.model_learning.IQLearn.iql import run_iql
from ai_coach_core.model_learning.LatentIQL.train_mental_iql_pond import (
    train_mental_iql_pond)


def learn_iql(config: ARGConfig, log_dir, output_dir, path_iq_data, num_traj):

  log_interval = 500
  eval_interval = 10000

  run_iql(config,
          path_iq_data,
          num_traj,
          log_dir,
          output_dir,
          log_interval=log_interval,
          eval_interval=eval_interval)


def learn_miql(config: ARGConfig, log_dir, output_dir, path_iq_data, num_traj):

  log_interval = 500
  eval_interval = 10

  train_mental_iql_pond(config,
                        path_iq_data,
                        num_traj,
                        log_dir,
                        output_dir,
                        log_interval=log_interval,
                        eval_epoch_interval=eval_interval)
