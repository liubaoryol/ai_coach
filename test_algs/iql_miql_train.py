from aicoach_baselines.option_gail.utils.config import ARGConfig
from ai_coach_core.model_learning.IQLearn.iql import run_iql
from ai_coach_core.model_learning.LatentIQL.train_mental_iql_pond import (
    train_mental_iql_pond)


def learn_iql(config: ARGConfig, log_dir, output_dir, path_iq_data, num_traj):

  n_sample = config.n_sample
  n_step = 10
  batch_size = config.mini_batch_size
  clip_grad_val = 0.5
  learn_alpha = False

  num_iter = config.max_explore_step
  log_interval = 500
  eval_interval = 10000

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
          list_actor_hidden_dims=config.hidden_policy,
          list_critic_hidden_dims=config.hidden_critic,
          clip_grad_val=clip_grad_val,
          learn_alpha=learn_alpha,
          critic_lr=config.optimizer_lr_policy,
          actor_lr=config.optimizer_lr_critic,
          alpha_lr=config.optimizer_lr_alpha,
          bounded_actor=config.bounded_actor,
          method_loss=config.method_loss)


def learn_miql(config: ARGConfig, log_dir, output_dir, path_iq_data, num_traj):

  n_sample = config.n_sample
  n_step = 10
  batch_size = config.mini_batch_size
  clip_grad_val = 0.5
  learn_alpha = False

  log_interval = 500
  eval_interval = 10

  train_mental_iql_pond(config.env_name, {},
                        config.seed,
                        batch_size,
                        config.dim_c,
                        path_iq_data,
                        num_traj,
                        log_dir,
                        output_dir,
                        replay_mem=n_sample,
                        initial_mem=n_sample,
                        max_explore_step=config.max_explore_step,
                        log_interval=log_interval,
                        eval_epoch_interval=eval_interval,
                        list_critic_hidden_dims=config.hidden_critic,
                        list_actor_hidden_dims=config.hidden_policy,
                        list_thinker_hidden_dims=config.hidden_option,
                        clip_grad_val=clip_grad_val,
                        learn_alpha=learn_alpha,
                        critic_lr=config.optimizer_lr_critic,
                        actor_lr=config.optimizer_lr_policy,
                        thinker_lr=config.optimizer_lr_option,
                        alpha_lr=config.optimizer_lr_alpha,
                        gumbel_temperature=1.0,
                        bounded_actor=config.bounded_actor,
                        method_loss=config.method_loss,
                        method_regularize=config.method_regularize,
                        use_prev_action=config.use_prev_action)
