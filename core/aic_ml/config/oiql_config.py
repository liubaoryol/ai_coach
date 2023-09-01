from .default_config import default_config

oiql_config = default_config.copy()
oiql_config.update({
    "clip_grad_val": 0.0,
    "gumbel_temperature": 1.0,
    "separate_policy_update": False,
    "stream_training": True,
    "n_update_rounds": 256,  # only when stream_training=False
    "demo_latent_infer_interval": 4096,  # only when stream_training=True
    # Q-net
    "hidden_critic": (256, 256),
    "optimizer_lr_critic": 3.e-4,
    "method_loss": "v0",
    "method_regularize": True,
    "num_critic_update": 1,
    "iql_single_critic": True,
    # policy
    "hidden_policy": (256, 256),
    "optimizer_lr_policy": 3.e-4,
    "bounded_actor": True,
    "use_nn_logstd": True,
    "clamp_action_logstd": False,  # True: use clamp() / False: use tanh
    "log_std_bounds": (-5., 2.),
    "num_actor_update": 1,
    # option
    "hidden_option": (256, 256),
    "optimizer_lr_option": 3.e-4,
    "use_prev_action": False,
    "use_prev_action_dim": True,  # use extra dim to represent initial value
    "use_prev_option_dim": True,  # use extra dim to represent initial value
    "thinker_clip_grad_val": 0.0,
    # alpha
    "optimizer_lr_alpha": 3.e-4,
    "init_temp": 1e-2,
    "learn_temp": False,
})
