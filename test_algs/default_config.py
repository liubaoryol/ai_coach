import os
from aicoach_baselines.option_gail.utils.config import Config

default_config = Config({
    # global program config
    "seed": 0,
    "tag": "default",
    "device": "cuda:0",
    "n_thread": 1,
    "n_sample": 4096,
    "n_epoch": 5000,
    "max_explore_step": 5e4,
    "base_dir": os.path.dirname(__file__),
    "supervision": 0.0,

    # global task config
    "env_type": "mujoco",
    "env_name": "HalfCheetah-v2",
    "use_state_filter": False,

    # global policy config
    "activation": "relu",
    "hidden_policy": (256, 256),
    "shared_policy": False,
    "log_std_bounds": (-5., 2.),
    "optimizer_lr_policy": 3.e-4,
    "dim_c": 4,
    "use_option": True,
    "hidden_option": (256, 256),
    "optimizer_lr_option": 3.e-4,
    "clip_grad_val": 0.0,

    # ppo config
    "hidden_critic": (256, 256),
    "shared_critic": False,
    "train_policy": True,
    "train_option": True,
    "optimizer_lr_critic": 3.e-4,
    "use_gae": True,
    "gamma": 0.99,
    "gae_tau": 0.95,
    "clip_eps": 0.2,
    "mini_batch_size": 256,
    "lambda_entropy_policy": 0.,
    "lambda_entropy_option": 1.e-2,

    # pre-train config
    "n_pretrain_epoch": 1000,
    "pretrain_log_interval": 500,

    # gail config
    "use_pretrain": False,
    "hidden_discriminator": (256, 256),
    "shared_discriminator": False,
    "use_c_in_discriminator": True,
    "optimizer_lr_discriminator": 3.e-4,
    "use_d_info_gail": False,

    # gail debug
    "gail_option_entropy_orig": True,
    "gail_option_sample_orig": True,
    "gail_orig_log_opt": True,
    "clamp_action_logstd": True,  # True: use clamp() / False: use tanh
    "use_nn_logstd": False,

    # oiql/iql config
    "iql_agent_name": "sac",  # softq \ sac \ sacd
    "bounded_actor": True,
    "method_loss": "v0",
    "n_traj": 1,
    "method_regularize": True,
    "use_prev_action": False,
    "optimizer_lr_alpha": 3.e-4,
    "num_critic_update": 1,
    "num_actor_update": 1,
    "gumbel_temperature": 1.0,
    "use_prev_action_dim": True,
    "use_prev_option_dim": True,
    "demo_latent_infer_interval": 4096,
    "separate_policy_update": False,
    "init_temp": 1e-2,
    "learn_temp": False,
    "thinker_clip_grad_val": 0.0,
    "stream_training": True,
    "n_update_rounds": 256,
    "iql_single_critic": True,

    # miql config
    "miql_update_strategy":
    1,  # 1: always update both / 2: update in order / 3: update alternatively
    "miql_tx_after_pi": True,
    "miql_alter_update_n_pi_tx": (10, 5),
    "miql_order_update_pi_ratio": 0.5,
    # tx
    "miql_tx_method_loss": "value",
    "miql_tx_method_regularize": True,
    "miql_tx_init_temp": 1e-4,
    "miql_tx_clip_grad_val": 0.0,
    "miql_tx_num_critic_update": 1,
    "miql_tx_num_actor_update": 1,
    "miql_tx_activation": "relu",
    "miql_tx_hidden_critic": (64, 64),
    "miql_tx_optimizer_lr_critic": 3.e-4,
    "miql_tx_tx_batch_size": 64,
    # pi
    "miql_pi_method_loss": "value",
    "miql_pi_method_regularize": True,
    "miql_pi_init_temp": 1e-2,
    "miql_pi_learn_temp": True,
    "miql_pi_clip_grad_val": 0.0,
    "miql_pi_num_critic_update": 1,
    "miql_pi_num_actor_update": 1,
    "miql_pi_activation": "relu",
    "miql_pi_hidden_critic": (64, 64),
    "miql_pi_hidden_policy": (64, 64),
    "miql_pi_optimizer_lr_critic": 3.e-4,
    "miql_pi_optimizer_lr_policy": 3.e-4,
    "miql_pi_optimizer_lr_alpha": 3.e-4,
    "miql_pi_log_std_bounds": (-5., 2.),
    "miql_pi_bounded_actor": True,
    "miql_pi_use_nn_logstd": True,
    "miql_pi_clamp_action_logstd": False,  # True: use clamp() / False: use tanh
    "miql_pi_single_critic": True,
})

mujoco_config = default_config.copy()
mujoco_config.update({
    # global task config
    "env_type": "mujoco",
    "n_traj": 1,
    "dim_c": 4,
    "device": "cuda:0",
    "max_explore_step": 1e6,
    "mini_batch_size": 256,
    "miql_tx_tx_batch_size": 64,
    "demo_latent_infer_interval": 5000,
    "stream_training": True,
    "miql_tx_optimizer_lr_critic": 1.e-4,
    "miql_pi_optimizer_lr_critic": 3.e-4,
    "miql_pi_optimizer_lr_policy": 3.e-5,
    "miql_tx_hidden_critic": (32, 32),
    "miql_pi_hidden_critic": (64, 64),
    "miql_pi_hidden_policy": (64, 64),
    "miql_tx_init_temp": 1e-4,
    "miql_pi_learn_temp": False,
    "miql_tx_method_regularize": True,
    "miql_pi_method_regularize": True,
    "miql_pi_single_critic": True,
})
