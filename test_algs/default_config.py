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

    # global task config
    "env_type": "mujoco",
    "env_name": "HalfCheetah-v2",
    "use_state_filter": False,

    # global policy config
    "activation": "relu",
    "hidden_policy": (64, 64),  # per option
    "shared_policy": False,
    "log_std_bounds": (-5., 2.),
    "optimizer_lr_policy": 3.e-4,
    "dim_c": 4,
    "use_option": True,
    "hidden_option": (64, 64),  # per option
    "optimizer_lr_option": 3.e-4,

    # ppo config
    "hidden_critic": (64, 64),  # per option
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
    "n_demo": 2048,
    "n_pretrain_epoch": 1000,
    "pretrain_log_interval": 500,

    # gail config
    "use_pretrain": False,
    "hidden_discriminator": (256, 256),
    "shared_discriminator": False,
    "use_c_in_discriminator": True,
    "optimizer_lr_discriminator": 3.e-4,
    "use_d_info_gail": False,

    # miql/iql config
    "iql_agent_name": "sac",
    "bounded_actor": True,
    "method_loss": "v0",
    "n_traj": 1,
    "method_regularize": True,
    "use_prev_action": False,
    "optimizer_lr_alpha": 3.e-4,
    "num_critic_update": 1,
    "num_actor_update": 1,
    "clip_grad_val": 0.0,
    "gumbel_temperature": 1.0,
    "use_prev_action_dim": True,
    "use_prev_option_dim": True,

    # gail debug
    "gail_option_entropy_orig": True,
    "gail_option_sample_orig": True,
    "gail_orig_log_opt": True,
    "clamp_action_logstd": True,  # True: use clamp() / False: use tanh
    "use_nn_logstd": False,
})

mujoco_config = default_config.copy()
mujoco_config.update({
    # global task config
    "env_type": "mujoco",
    "env_name": "HalfCheetah-v2",

    # pre-train config
    "n_demo": 5000,
    "n_pretrain_epoch": 1000,
    "pretrain_log_interval": 500,
})

rlbench_config = default_config.copy()
rlbench_config.update({
    # global task config
    "n_thread": 3,
    "env_type": "rlbench",
    "env_name": "PlaceHangerOnRack",

    # pre-train config
    "n_demo": 5000,
    "n_pretrain_epoch": 50000,
    "pretrain_log_interval": 500,
})

mini_config = default_config.copy()
mini_config.update({
    # global task config
    "env_type": "mini",
    "env_name": "Circle",

    # pre-train config
    "n_demo": 2048,
    "n_pretrain_epoch": 750,
    "pretrain_log_interval": 200,
})
