import os
from .default_config import default_config

iql_config = default_config.copy()
iql_config.update({
    "iql_agent_name": "sac",  # softq \ sac \ sacd
    "hidden_critic": (256, 256),
    "hidden_policy": (256, 256),
    "hidden_option": (256, 256),
    "optimizer_lr_critic": 3.e-4,
    "optimizer_lr_policy": 3.e-4,
    "optimizer_lr_option": 3.e-4,
    "optimizer_lr_alpha": 3.e-4,
    "bounded_actor": True,
    "log_std_bounds": (-5., 2.),
    "clip_grad_val": 0.0,
    "method_loss": "v0",
    "method_regularize": True,
    "use_prev_action": False,
    "num_critic_update": 1,
    "num_actor_update": 1,
    "gumbel_temperature": 1.0,
    "use_prev_action_dim": True,
    "use_prev_option_dim": True,
    "init_temp": 1e-2,
    "learn_temp": False,
    "iql_single_critic": True,
})
