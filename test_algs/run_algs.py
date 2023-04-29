import os
import torch
from aicoach_baselines.option_gail.utils.config import ARGConfig
from default_config import rlbench_config, mujoco_config


def run_alg(config):
  alg_name = config.alg_name
  msg = f"{config.loss_type}_{config.tag}"
  if alg_name == "bc":
    from aicoach_baselines.option_gail.option_bc_learn import learn
    learn(config, msg)
  elif alg_name == "gail":
    from aicoach_baselines.option_gail.option_gail_learn import learn
    learn(config, msg)
  elif alg_name == "gailmoe":
    from aicoach_baselines.option_gail.option_gail_learn_moe import learn
    learn(config, msg)
  elif alg_name == "ppo":
    from aicoach_baselines.option_gail.option_ppo_learn import learn
    learn(config, msg)
  elif alg_name == "miql":
    from miql_train import learn
    learn(config)
  elif alg_name == "miql_v2":
    from ai_coach_core.model_learning.LatentIQL_v2.train_mental_iql_v2 import (
        learn)
    learn(config, msg)
  elif alg_name == "iql":
    from iql_train import learn
    learn(config)
  else:
    raise ValueError("Invalid alg_name")


if __name__ == "__main__":
  import torch.multiprocessing as multiprocessing
  multiprocessing.set_start_method('spawn')

  arg = ARGConfig()
  arg.add_arg("alg_name", "gail", "bc / gail / gailmoe / ppo")
  arg.add_arg("use_option", True, "Use Option when training or not")
  arg.add_arg("n_pretrain_epoch", 5000, "Pretrain epoches")
  arg.add_arg("pretrain_log_interval", 20, "Pretrain logging logging interval")
  # We can prove that using MLE is equivalent to MAP mathematically, but the former has higher computational efficiency
  # so here we recommend using MLE instead of MAP in your future work
  arg.add_arg("loss_type", "L2",
              "Pretraining method [L2, MLE, MAP, subpart MAP]")
  arg.add_arg("dim_c", 4, "Number of Options")
  arg.add_arg("env_type", "mujoco",
              "Environment type, can be [mujoco, rlbench, mini]")
  arg.add_arg("env_name", "AntPush-v0", "Environment name")
  arg.add_arg("device", "cuda:0", "Computing device")
  arg.add_arg("tag", "default", "Experiment tag")
  arg.add_arg("n_demo", 5000, "Number of demonstration s-a")
  arg.add_arg("n_epoch", 4000, "Number of training epochs")
  arg.add_arg("seed", torch.randint(100, ()).item(), "Random seed")
  arg.add_arg("use_c_in_discriminator", True,
              "Use (s,a) or (s,c,a) as occupancy measurement")
  arg.add_arg("use_d_info_gail", False, "Use directed-info gail or not")
  arg.add_arg(
      "use_pretrain", False,
      "Use pretrained master policy or not (only true when using D-info-GAIL)")
  arg.add_arg("train_option", True,
              "Train master policy or not (only false when using D-info-GAIL)")
  arg.add_arg("use_state_filter", True, "Use state filter")
  arg.add_arg("bounded_actor", True, "use bounded actor")
  arg.add_arg("data_path", "", "data path")
  arg.add_arg("use_prev_action", True, "use prev action in trans")
  arg.parser()

  if arg.env_type == "rlbench":
    config = rlbench_config
  elif arg.env_type == "mujoco":
    config = mujoco_config
  else:
    raise ValueError(
        "mini for circle env; rlbench for rlbench env; mujoco for mujoco env")

  config.base_dir = os.path.dirname(__file__)

  config.update(arg)
  if config.env_name.startswith("Humanoid"):
    config.hidden_policy = (512, 512)
    config.hidden_critic = (512, 512)
    print(
        f"Training Humanoid.* envs with larger policy network size :{config.hidden_policy}"
    )
  if config.env_type == "rlbench":
    config.hidden_policy = (128, 128)
    config.hidden_option = (128, 128)
    config.hidden_critic = (128, 128)
    config.log_clamp_policy = (-20., -2.)
    print(
        f"Training RLBench.* envs with larger policy network size :{config.hidden_policy}"
    )

  print(
      f">>>> Training {'Option-' if config.use_option else ''} {config.alg_name} using {config.env_name} environment on {config.device}"
  )
  run_alg(config)
