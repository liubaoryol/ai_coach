from .default_config import default_config

iql_config = default_config.copy()
iql_config.update({
    "iql_agent_name": "sac",  # softq \ sac \ sacd
    "clip_grad_val": 0.0,
    # Q-network
    "hidden_critic": (256, 256),
    "optimizer_lr_critic": 3.e-4,
    "iql_single_critic": True,
    "method_loss": "v0",
    "method_regularize": True,
    "num_critic_update": 1,
    # policy
    "hidden_policy": (256, 256),
    "optimizer_lr_policy": 3.e-4,
    "bounded_actor": True,
    "use_nn_logstd": True,
    "clamp_action_logstd": False,  # True: use clamp() / False: use tanh
    "log_std_bounds": (-5., 2.),
    "gumbel_temperature": 1.0,  # only for sac with discrete action agent
    "num_actor_update": 1,
    # alpha
    "optimizer_lr_alpha": 3.e-4,
    "init_temp": 1e-2,
    "learn_temp": False,
})
