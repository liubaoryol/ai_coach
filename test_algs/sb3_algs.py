import os
import stable_baselines3 as sb3
import gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback


def sb3_run_sac(config, log_dir, output_dir, log_interval, eval_interval):
  env_name = config.env_name
  env = gym.make(env_name)
  eval_env = gym.make(env_name)

  logger = configure(log_dir, ("log", "csv", "tensorboard"))
  best_output_path = os.path.join(output_dir, "best_sb3_sac_" + f"{env_name}")
  eval_callback = EvalCallback(eval_env,
                               best_model_save_path=best_output_path,
                               n_eval_episodes=10,
                               log_path=log_dir,
                               eval_freq=eval_interval,
                               verbose=0)

  model = sb3.SAC("MlpPolicy",
                  env,
                  verbose=0,
                  learning_rate=config.optimizer_lr_policy,
                  buffer_size=config.n_sample,
                  batch_size=config.mini_batch_size,
                  gamma=config.gamma)
  model.set_logger(logger)
  model.learn(total_timesteps=config.max_explore_step,
              log_interval=log_interval,
              callback=eval_callback)

  output_path = os.path.join(output_dir, "sb3_sac_" + f"{env_name}")
  model.save(output_path)
