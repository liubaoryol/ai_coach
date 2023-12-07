import os
import numpy as np
import torch
from itertools import count
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.envs.wrappers import EnvWrapper
import gym_custom.envs.robomimic_env.pickplacecan_config as rbm_cfg


class FlatteningEnvWrapper(EnvWrapper):

  def __init__(self, env):
    super(FlatteningEnvWrapper, self).__init__(env)

  def seed(self, seed):
    Warning("Robomimic envs do not support seed")
    pass

  @classmethod
  def flatten_obs_dict(cls, obs_dict):
    obs_dict = TensorUtils.to_tensor(obs_dict)
    feats = []
    for k in rbm_cfg.SHAPE_META["all_shapes"]:
      x = obs_dict[k]
      if obs_dict[k].ndim == 1:
        x = x.unsqueeze(0)
      x = TensorUtils.flatten(x, begin_axis=1)
      feats.append(x)

    return torch.cat(feats, dim=-1)

  def reset(self):
    obs_dict = self.env.reset()

    return FlatteningEnvWrapper.flatten_obs_dict(obs_dict)[0]

  def step(self, action):
    obs_dict, reward, done, info = self.env.step(action)

    obs = FlatteningEnvWrapper.flatten_obs_dict(obs_dict)[0]

    return obs, reward, done, info


if __name__ == "__main__":

  # robomimic_path = '/home/sangwon/Projects/external/robomimic/'
  # data_path = os.path.join(robomimic_path, "datasets/can/ph/low_dim_v141.hdf5")

  # all_obs_keys = [
  #     'object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'
  # ]

  modalities = {
      "obs": {
          "low_dim": [
              "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
              "object"
          ]
      }
  }

  ObsUtils.initialize_obs_utils_with_obs_specs(modalities)

  # env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=data_path)
  # print(env_meta)
  # shape_meta = FileUtils.get_shape_metadata_from_dataset(
  #     dataset_path=data_path, all_obs_keys=all_obs_keys, verbose=True)
  # print(shape_meta)

  ac_dim = rbm_cfg.SHAPE_META["ac_dim"]
  all_shapes = rbm_cfg.SHAPE_META["all_shapes"]
  horizon = 400

  env_name = rbm_cfg.ENV_META["env_name"]
  env = EnvUtils.create_env_from_metadata(env_meta=rbm_cfg.ENV_META,
                                          env_name=env_name)
  env = FlatteningEnvWrapper(env)

  obs = env.reset()
  total_reward = 0
  for step_i in range(horizon):
    # policy action
    action = np.random.randn(ac_dim)

    obs, r, done, _ = env.step(action)  # take action in the environment

    # compute reward
    total_reward += r

    success = env.is_success()

    if done or success["task"]:
      break
