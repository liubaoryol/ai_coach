import os
import stable_baselines3 as sb3
import gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback


def sb3_run(config, log_dir, output_dir, log_interval, eval_interval, alg_name):
  env_name = config.env_name
  env = gym.make(env_name)
  eval_env = gym.make(env_name)

  logger = configure(log_dir, ("log", "csv", "tensorboard"))
  best_output_path = os.path.join(output_dir, f"best_sb3_{alg_name}_{env_name}")
  eval_callback = EvalCallback(eval_env,
                               best_model_save_path=best_output_path,
                               n_eval_episodes=10,
                               log_path=log_dir,
                               eval_freq=eval_interval,
                               verbose=0)

  if alg_name == "sac":
    policy_kwargs = dict(
        net_arch=dict(pi=config.hidden_policy, qf=config.hidden_critic))
    model = sb3.SAC("MlpPolicy",
                    env,
                    learning_rate=config.optimizer_lr_policy,
                    buffer_size=config.n_sample,
                    batch_size=config.mini_batch_size,
                    gamma=config.gamma,
                    seed=config.seed,
                    device=config.device,
                    policy_kwargs=policy_kwargs)
    if hasattr(env.spec, "max_episode_steps"):
      max_epi_steps = env.spec.max_episode_steps
    else:
      max_epi_steps = 1000
    log_interval = max(int(log_interval / max_epi_steps), 1)

  elif alg_name == "ppo":
    policy_kwargs = dict(
        net_arch=dict(pi=config.hidden_policy, vf=config.hidden_critic))

    if config.clip_grad_val:
      max_grad_norm = config.clip_grad_val
    else:
      max_grad_norm = 10

    model = sb3.PPO("MlpPolicy",
                    env,
                    learning_rate=config.optimizer_lr_policy,
                    n_steps=config.n_sample,
                    n_epochs=config.n_update_rounds,
                    batch_size=config.mini_batch_size,
                    gamma=config.gamma,
                    max_grad_norm=max_grad_norm,
                    clip_range=config.clip_eps,
                    clip_range_vf=config.clip_eps,
                    ent_coef=config.lambda_entropy_policy,
                    seed=config.seed,
                    device=config.device,
                    policy_kwargs=policy_kwargs)
    log_interval = max(int(log_interval / config.n_sample), 1)
  else:
    raise NotImplementedError

  model.set_logger(logger)
  model.learn(total_timesteps=config.max_explore_step,
              log_interval=log_interval,
              callback=eval_callback,
              tb_log_name=alg_name)

  output_path = os.path.join(output_dir, f"sb3_{alg_name}_{env_name}")
  model.save(output_path)
