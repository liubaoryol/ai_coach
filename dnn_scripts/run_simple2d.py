import os
from aic_ml.baselines.option_gail.utils.config import ARGConfig
from default_config import simple2d_config
from run_algs import run_alg
import gym_custom

if __name__ == "__main__":
  import torch.multiprocessing as multiprocessing
  multiprocessing.set_start_method('spawn')

  arg = ARGConfig()
  for key, value in simple2d_config.items():
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
  if arg.thinker_clip_grad_val == 0:
    arg.thinker_clip_grad_val = None
  if arg.miql_pi_clip_grad_val == 0:
    arg.miql_pi_clip_grad_val = None
  if arg.miql_tx_clip_grad_val == 0:
    arg.miql_tx_clip_grad_val = None

  config = simple2d_config
  config.base_dir = os.path.dirname(__file__)

  config.update(arg)
  # Training Humanoid.* envs with larger policy network size
  if config.env_name.startswith("Humanoid"):
    config.hidden_policy = (512, 512)
    config.hidden_critic = (512, 512)

  if config.alg_name in ["obc", "ogail", "ogailv2", "oppo", "oppov2", "oiqlv2"]:
    if config.use_option:
      dim_c = config.dim_c
      hp1, hp2 = config.hidden_policy
      ho1, ho2 = config.hidden_option
      hc1, hc2 = config.hidden_critic

      config.hidden_policy = (hp1 // dim_c, hp2 // dim_c)
      config.hidden_option = (ho1 // dim_c, ho2 // dim_c)
      config.hidden_critic = (hc1 // dim_c, hc2 // dim_c)

  if config.alg_name != "miql":
    print(f"Hidden_policy: {config.hidden_policy}",
          f"Hidden_option: {config.hidden_option}",
          f"Hidden_critic: {config.hidden_critic}")

  print(f">>>> Training "
        f"{config.alg_name} using {config.env_name} "
        f"environment on {config.device}")
  run_alg(config, log_interval=500, eval_interval=2000)
