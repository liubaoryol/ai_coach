import os
from aic_ml.baselines.option_gail.utils.config import Config

default_config = Config({
    # global program config
    "seed": 0,
    "tag": "default",
    "device": "cuda:0",
    "n_thread": 1,
    "n_sample": 4096,  # replay buffer size
    "max_explore_step": 5e4,
    "base_dir": os.path.dirname(__file__),

    # global task config
    "env_type": "mujoco",
    "env_name": "HalfCheetah-v2",
    "dim_c": 4,
    "n_traj": 1,
    "supervision": 0.0,

    # common config
    "mini_batch_size": 256,
    "gamma": 0.99,
})
