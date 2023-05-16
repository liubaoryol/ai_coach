import os
import torch
import json
from aicoach_baselines.option_gail.utils.config import ARGConfig
from aicoach_baselines.option_gail.utils.mujoco_env import load_demo
from ai_coach_core.model_learning.IQLearn.dataset.expert_dataset import (
    read_file)
from default_config import rlbench_config, mujoco_config, default_config
from iql_helper import (get_dirs, conv_torch_trajs_2_iql_format,
                        conv_iql_trajs_2_optiongail_format)


def get_pickle_datapath_n_traj(config):
  data_path = os.path.join(config.base_dir, config.data_path)
  num_traj = config.n_traj
  if data_path.endswith("torch"):
    trajs, _ = load_demo(data_path, config.n_demo)
    data_dir = os.path.dirname(data_path)
    num_traj = len(trajs)

    data_path = os.path.join(data_dir, f"temp_{config.env_name}_{num_traj}.pkl")
    conv_torch_trajs_2_iql_format(trajs, data_path)

  return data_path, num_traj


def get_torch_datapath(config):
  data_path = os.path.join(config.base_dir, config.data_path)
  if not data_path.endswith("torch"):
    with open(data_path, 'rb') as f:
      trajs = read_file(data_path, f)
    num_traj = len(trajs["states"])

    data_dir = os.path.dirname(data_path)
    data_path = os.path.join(data_dir,
                             f"temp_{config.env_name}_{num_traj}.torch")

    conv_iql_trajs_2_optiongail_format(trajs, data_path)

  return data_path


def run_alg(config):
  alg_name = config.alg_name
  msg = f"{config.tag}"

  log_dir, output_dir, log_dir_root = get_dirs(config.base_dir, alg_name,
                                               config.env_type, config.env_name,
                                               msg)
  pretrain_name = os.path.join(config.base_dir, config.pretrain_path)

  # save config
  config_path = os.path.join(log_dir_root, "config.txt")
  with open(config_path, "w") as outfile:
    outfile.write(str(config))

  log_interval, eval_interval = 500, 10000
  if (config.data_path.endswith("torch") or config.data_path.endswith("pt")
      or config.data_path.endswith("pkl") or config.data_path.endswith("npy")):
    sample_name = get_torch_datapath(config)
    path_iq_data, num_traj = get_pickle_datapath_n_traj(config)
  else:
    print(f"Data path not exists: {config.data_path}")

  if alg_name == "obc":
    from aicoach_baselines.option_gail.option_bc_learn import learn
    learn(config, log_dir, output_dir, sample_name, pretrain_name, msg)
  elif alg_name == "ogail":
    from aicoach_baselines.option_gail.option_gail_learn import learn
    learn(config, log_dir, output_dir, sample_name, pretrain_name, msg)
  elif alg_name == "ogailv2":
    from aicoach_baselines.option_gail.option_gail_learn_v2 import learn
    learn(config, log_dir, output_dir, sample_name, pretrain_name, msg)
  elif alg_name == "oppo":
    from aicoach_baselines.option_gail.option_ppo_learn import learn
    learn(config, log_dir, output_dir, msg)
  elif alg_name == "oppov2":
    from aicoach_baselines.option_gail.option_ppo_learn_v2 import learn
    learn(config, log_dir, output_dir, msg)
  elif alg_name == "miql" and config.miql_stream:
    from ai_coach_core.model_learning.LatentIQL.train_mental_iql_stream import (
        train_mental_iql_stream)
    train_mental_iql_stream(config, path_iq_data, num_traj, log_dir, output_dir,
                            log_interval, eval_interval)
  elif alg_name == "miql" and not config.miql_stream:
    from ai_coach_core.model_learning.LatentIQL.train_mental_iql_pond import (
        train_mental_iql_pond)
    train_mental_iql_pond(config, path_iq_data, num_traj, log_dir, output_dir,
                          log_interval, eval_interval)
  elif alg_name == "miqlv2":
    from ai_coach_core.model_learning.LatentIQL_v2.train_mental_iql_v2 import (
        learn)
    learn(config, log_dir, output_dir, sample_name, pretrain_name, msg)
  elif alg_name == "iql":
    from ai_coach_core.model_learning.IQLearn.iql import run_iql
    run_iql(config, log_dir, output_dir, path_iq_data, num_traj, log_interval,
            eval_interval)
  elif alg_name == "sac":
    from ai_coach_core.model_learning.IQLearn.iql import run_sac
    run_sac(config, log_dir, output_dir, log_interval, eval_interval)
  elif alg_name == "msac" and config.miql_stream:
    from ai_coach_core.model_learning.LatentIQL.train_mental_iql_stream import (
        train_mental_sac_stream)
    train_mental_sac_stream(config, log_dir, output_dir, log_interval,
                            eval_interval)
  elif alg_name == "msac" and not config.miql_stream:
    from ai_coach_core.model_learning.LatentIQL.train_mental_iql_pond import (
        train_mental_sac_pond)
    train_mental_sac_pond(config, log_dir, output_dir, log_interval,
                          eval_interval)
  elif alg_name == "sb3_sac":
    from sb3_algs import sb3_run_sac
    log_interval, eval_interval = 10, 5000
    sb3_run_sac(config, log_dir, output_dir, log_interval, eval_interval)
  else:
    raise ValueError("Invalid alg_name")


if __name__ == "__main__":
  import torch.multiprocessing as multiprocessing
  multiprocessing.set_start_method('spawn')

  arg = ARGConfig()
  for key, value in default_config.items():
    arg.add_arg(key, value)

  arg.add_arg("alg_name", "gail", "bc / gail /  ppo")
  # We can prove that using MLE is equivalent to MAP mathematically, but the former has higher computational efficiency
  # so here we recommend using MLE instead of MAP in your future work
  arg.add_arg("loss_type", "L2",
              "Pretraining method [L2, MLE, MAP, subpart MAP]")
  arg.add_arg("data_path", "", "data path")
  arg.add_arg("pretrain_path", "", "pretrain path")
  arg.parser()
  if arg.clip_grad_val == 0:
    arg.clip_grad_val = None

  if arg.env_type == "rlbench":
    config = rlbench_config
  elif arg.env_type == "mujoco":
    config = mujoco_config
  else:
    raise ValueError(
        "mini for circle env; rlbench for rlbench env; mujoco for mujoco env")

  config.base_dir = os.path.dirname(__file__)

  config.update(arg)
  # Training Humanoid.* envs with larger policy network size
  if config.env_name.startswith("Humanoid"):
    config.hidden_policy = (512, 512)
    config.hidden_critic = (512, 512)

  # Training RLBench.* envs with larger policy network size
  if config.env_type == "rlbench":
    config.hidden_policy = (128, 128)
    config.hidden_option = (128, 128)
    config.hidden_critic = (128, 128)
    config.log_std_bounds = (-20., -2.)

  if config.alg_name in ["obc", "ogail", "ogailv2", "oppo", "miqlv2"]:
    dim_c = config.dim_c
    hp1, hp2 = config.hidden_policy
    ho1, ho2 = config.hidden_option
    hc1, hc2 = config.hidden_critic

    config.hidden_policy = (hp1 // dim_c, hp2 // dim_c)
    config.hidden_option = (ho1 // dim_c, ho2 // dim_c)
    config.hidden_critic = (hc1 // dim_c, hc2 // dim_c)

  print(f"Hidden_policy: {config.hidden_policy}",
        f"Hidden_option: {config.hidden_option}",
        f"Hidden_critic: {config.hidden_critic}")

  print(f">>>> Training {'Option-' if config.use_option else ''} "
        f"{config.alg_name} using {config.env_name} "
        f"environment on {config.device}")
  run_alg(config)
