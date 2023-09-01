from .default_config import default_config

ogail_config = default_config.copy()
ogail_config.update({
    "n_epoch": 5000,
    "use_state_filter": False,
    "use_option": True,
    "activation": "relu",
    "use_d_info_gail": False,
    "clip_grad_val": 0.0,
    # critic
    "hidden_critic": (256, 256),
    "optimizer_lr_critic": 3.e-4,
    "shared_critic": False,
    # policy
    "hidden_policy": (256, 256),
    "optimizer_lr_policy": 3.e-4,
    "shared_policy": False,
    "train_policy": True,
    "log_std_bounds": (-5., 2.),
    # option
    "hidden_option": (256, 256),
    "optimizer_lr_option": 3.e-4,
    "train_option": True,
    # discriminator
    "hidden_discriminator": (256, 256),
    "optimizer_lr_discriminator": 3.e-4,
    "shared_discriminator": False,
    "use_c_in_discriminator": True,
    # ppo
    "use_gae": True,
    "gae_tau": 0.95,
    "clip_eps": 0.2,
    "lambda_entropy_policy": 0.,
    "lambda_entropy_option": 1.e-2,
    # pre-train config
    "n_pretrain_epoch": 1000,
    "pretrain_log_interval": 500,
    "use_pretrain": False,
})
