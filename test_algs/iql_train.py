import os
from aicoach_baselines.option_gail.utils.config import ARGConfig
from ai_coach_core.model_learning.IQLearn.iql import run_iql
from aicoach_baselines.option_gail.utils.mujoco_env import load_demo
from iql_helper import get_dirs, conv_torch_trajs_2_iql_format


def learn(config: ARGConfig):

  log_dir, output_dir, sample_name = get_dirs(config.seed, config.base_dir,
                                              "iql", config.env_type,
                                              config.env_name)

  if config.data_path == "":
    trajs, _ = load_demo(sample_name, config.n_demo)
    data_dir = os.path.dirname(sample_name)
    num_traj = len(trajs)

    path_iq_data = os.path.join(data_dir, f"{config.env_name}_{num_traj}.pkl")
    conv_torch_trajs_2_iql_format(trajs, path_iq_data)
  else:
    path_iq_data = os.path.join(config.base_dir, config.data_path)
    num_traj = config.n_traj

  n_sample = config.n_sample
  n_step = 10
  batch_size = config.mini_batch_size
  clip_grad_val = 0.5
  learn_alpha = True

  update_per_epoch = int(n_sample / batch_size) * n_step
  num_iter = config.n_epoch * update_per_epoch
  log_interval = update_per_epoch
  eval_interval = 20 * update_per_epoch

  run_iql(config.env_name, {},
          config.seed,
          batch_size,
          path_iq_data,
          num_traj,
          log_dir,
          output_dir,
          replay_mem=n_sample,
          eps_window=10,
          agent_name="sac",
          num_learn_steps=num_iter,
          log_interval=log_interval,
          eval_interval=eval_interval,
          list_hidden_dims=config.hidden_policy,
          clip_grad_val=clip_grad_val,
          learn_alpha=learn_alpha,
          learning_rate=config.optimizer_lr_policy,
          bounded_actor=config.bounded_actor,
          method_loss=config.method_loss)
