import os
import numpy as np
import torch
import h5py
from collections import defaultdict
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import gym
import pickle
from gym import spaces
from gym_custom.envs.robomimic_env.pickplacecan_config import (ENV_META,
                                                               SHAPE_META)


class RMPickPlaceCan(gym.Env):

  def __init__(self) -> None:
    super().__init__()

    obs_dim = 0
    all_shapes = SHAPE_META["all_shapes"]
    for k in all_shapes:
      obs_dim += all_shapes[k][0]

    act_dim = SHAPE_META["ac_dim"]

    self.observation_space = spaces.Box(low=-np.inf,
                                        high=np.inf,
                                        shape=(obs_dim, ),
                                        dtype=np.float32)
    self.action_space = spaces.Box(low=-1,
                                   high=1,
                                   shape=(act_dim, ),
                                   dtype=np.float32)

    if ObsUtils.OBS_KEYS_TO_MODALITIES is None:
      modalities = {"obs": {"low_dim": SHAPE_META["all_obs_keys"]}}
      ObsUtils.initialize_obs_utils_with_obs_specs(modalities)
    else:
      Warning("A robomimic env has already been initialized.")

    self.env_impl = EnvUtils.create_env_from_metadata(
        env_meta=ENV_META, env_name=ENV_META["env_name"])

  def seed(self, seed):
    Warning("Robomimic envs do not support seed")
    pass

  def reset(self):
    obs_dict = self.env_impl.reset()
    return RMPickPlaceCan.flatten_obs_dict(obs_dict)[0]

  def step(self, action):
    obs_dict, reward, done, info = self.env_impl.step(action)
    success = self.env_impl.is_success()

    done = done or success["task"]
    obs = RMPickPlaceCan.flatten_obs_dict(obs_dict)[0]
    info["task_success"] = success["task"]

    return obs, reward, done, info

  @classmethod
  def flatten_obs_dict(cls, obs_dict):
    # obs_dict = TensorUtils.to_tensor(obs_dict)
    feats = []
    for k in SHAPE_META["all_shapes"]:
      shape = np.prod(SHAPE_META["all_shapes"][k])
      x = obs_dict[k]
      x = x.reshape((-1, shape))
      feats.append(x)

    return np.concatenate(feats, axis=-1)


def test_env():
  env = RMPickPlaceCan()

  obs = env.reset()
  total_reward = 0
  horizon = 400
  for step_i in range(horizon):
    # policy action
    action = np.random.randn(env.action_space.shape[0])

    obs, r, done, _ = env.step(action)  # take action in the environment

    # compute reward
    total_reward += r

    if done:
      break

  print("num steps:", step_i + 1)


def conv_dataset(data_path, save_dir, demo_start_end_idx=None):
  data_file = h5py.File(data_path, "r")

  # each demonstration is a group under "data"
  demos = list(data_file["data"].keys())
  if demo_start_end_idx is None:
    idx_st, idx_ed = 0, len(demos)
  else:
    idx_st, idx_ed = demo_start_end_idx
  num_demos = idx_ed - idx_st

  # each demonstration is named "demo_#" where # is a number.
  # Let's put the demonstration list in increasing episode order
  inds = np.argsort([int(elem[5:]) for elem in demos])
  demos = [demos[i] for i in inds]

  expert_trajs = defaultdict(list)
  for ep in demos[idx_st:idx_ed]:
    demo_grp = data_file["data/{}".format(ep)]

    num_samples = data_file["data/{}/actions".format(ep)].shape[0]
    obs_dict = dict()
    next_obs_dict = dict()
    demo_grp_obs = demo_grp['obs']
    for k in demo_grp_obs:
      obs_dict[k] = demo_grp["obs"][k][:]  # numpy array
      next_obs_dict[k] = demo_grp["next_obs"][k][:]  # numpy array
    # TODO: convert obs, next_obs
    obs = RMPickPlaceCan.flatten_obs_dict(obs_dict)
    next_obs = RMPickPlaceCan.flatten_obs_dict(next_obs_dict)

    actions = demo_grp["actions"][:]
    dones = demo_grp["dones"][:]
    rewards = demo_grp["rewards"][:]

    itemidx = np.where(actions[:, 6] == 1.0)
    itemidx = itemidx[0][0]

    latents = np.zeros(num_samples, dtype=np.int32)
    latents[itemidx:] = 1

    expert_trajs["states"].append(obs)
    expert_trajs["next_states"].append(next_obs)
    expert_trajs["latents"].append(latents)
    expert_trajs["actions"].append(actions)
    expert_trajs["rewards"].append(rewards)
    expert_trajs["dones"].append(dones)
    expert_trajs["lengths"].append(num_samples)

    file_path = os.path.join(save_dir, f"RMPickPlaceCan-v0_{num_demos}.pkl")
    with open(file_path, 'wb') as f:
      pickle.dump(expert_trajs, f)


if __name__ == "__main__":
  # test_env()

  GEN_DATA = True
  if GEN_DATA:
    robomimic_path = '/home/sangwon/Projects/external/robomimic/'
    data_path = os.path.join(robomimic_path,
                             "datasets/can/ph/low_dim_v141.hdf5")
    cur_dir = os.path.dirname(__file__)
    conv_dataset(data_path, cur_dir, demo_start_end_idx=(150, 200))
