from .default_config import default_config

miql_config = default_config.copy()
miql_config.update({
    "stream_training": True,
    "n_update_rounds": 256,  # only when stream_training=False
    "demo_latent_infer_interval": 4096,  # only when stream_training=True
    "miql_update_strategy": 1,  # 1: always update both / 2: update in order /
    #                             3: update alternatively
    "miql_tx_after_pi": True,
    "miql_alter_update_n_pi_tx": (10, 5),
    "miql_order_update_pi_ratio": 0.5,
    # tx
    "miql_tx_hidden_critic": (64, 64),
    "miql_tx_optimizer_lr_critic": 3.e-4,
    "miql_tx_activation": "relu",
    "miql_tx_method_loss": "value",
    "miql_tx_method_regularize": True,
    "miql_tx_init_temp": 1e-4,
    "miql_tx_clip_grad_val": 0.0,
    "miql_tx_num_critic_update": 1,
    "miql_tx_tx_batch_size": 64,
    # pi
    "miql_pi_activation": "relu",
    # pi - Q-network
    "miql_pi_hidden_critic": (64, 64),
    "miql_pi_optimizer_lr_critic": 3.e-4,
    "miql_pi_method_loss": "value",
    "miql_pi_method_regularize": True,
    "miql_pi_num_critic_update": 1,
    "miql_pi_single_critic": True,
    # pi - actor
    "miql_pi_hidden_policy": (64, 64),
    "miql_pi_optimizer_lr_policy": 3.e-4,
    "miql_pi_log_std_bounds": (-5., 2.),
    "miql_pi_num_actor_update": 1,
    "miql_pi_clip_grad_val": 0.0,
    "miql_pi_bounded_actor": True,
    "miql_pi_use_nn_logstd": True,
    "miql_pi_clamp_action_logstd": False,  # True: use clamp() / False: use tanh
    # pi - alpha
    "miql_pi_optimizer_lr_alpha": 3.e-4,
    "miql_pi_init_temp": 1e-2,
    "miql_pi_learn_temp": True,
})
