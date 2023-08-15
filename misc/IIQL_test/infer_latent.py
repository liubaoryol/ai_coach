import os
from ai_coach_core.model_learning.MentalIQL.agent.make_agent import (
    make_miql_agent)
from ai_coach_core.model_learning.OptionIQL.agent.make_agent import (
    make_oiql_agent)
from aicoach_baselines.option_gail.utils.config import Config
from ai_coach_core.model_learning.IQLearn.utils.utils import make_env
from ai_coach_core.model_learning.MentalIQL.train_miql import (
    load_expert_data_w_labels)
import gym_custom
import importlib.util
import sys
import numpy as np
from ai_coach_core.utils.result_utils import hamming_distance
from ai_coach_core.model_learning.OptionIQL.helper.utils import (
    infer_mental_states)

spec = importlib.util.spec_from_file_location(
    "default_config",
    "/home/sangwon/Projects/ai_coach/test_algs/default_config.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)


def miql_model_load(env_name, model_path):
  root_dir = os.path.dirname(os.path.dirname(model_path))
  config_path = os.path.join(root_dir, 'config.txt')

  config = foo.default_config
  config.load_saved(config_path)

  env = make_env(env_name, env_make_kwargs={})
  agent = make_miql_agent(config, env)

  agent.load(model_path)
  return agent, config


def oiql_model_load(env_name, model_path):
  root_dir = os.path.dirname(os.path.dirname(model_path))
  config_path = os.path.join(root_dir, 'config.txt')

  config = foo.default_config
  config.load_saved(config_path)

  env = make_env(env_name, env_make_kwargs={})
  agent = make_oiql_agent(config, env)

  agent.load(model_path)
  return agent, config


def miql_infer_latent(agent, trajectories):
  list_inferred_x = []
  for i_e in range(len(trajectories["states"])):
    states = trajectories["states"][i_e]
    actions = trajectories["actions"][i_e]
    inferred_x, _ = agent.infer_mental_states(states, actions)
    list_inferred_x.append(inferred_x)

  return list_inferred_x


def oiql_infer_latent(agent, trajectories):
  return infer_mental_states(agent, trajectories, agent.lat_dim, None)


def get_stats_about_x(list_inferred_x, list_true_x):
  dis_array = []
  length_array = []
  for i_e, inferred_x in enumerate(list_inferred_x):
    res = hamming_distance(inferred_x, list_true_x[i_e])
    dis_array.append(res)
    length_array.append(len(inferred_x))

  dis_array = np.array(dis_array)
  length_array = np.array(length_array)
  return dis_array, length_array


if __name__ == "__main__":

  if False:
    # load model
    model_path = "/home/sangwon/Projects/ai_coach/test_algs/result/EnvMovers-v0/miql/miql_256_3e-5_value/2023-08-15_00-53-14/model/miql_iq_EnvMovers-v0_n44_l44_best"
    env_name = "EnvMovers-v0"

    agent, config = miql_model_load(env_name, model_path)

    # load data
    num_data = 22
    data_path = "/home/sangwon/Projects/ai_coach/test_algs/experts/EnvMovers_v0_22.pkl"
    expert_dataset, traj_labels, cnt_label = load_expert_data_w_labels(
        data_path, num_data, 0, 0)

    list_inferred_x = miql_infer_latent(agent, expert_dataset.trajectories)
    hd_array, len_array = get_stats_about_x(
        list_inferred_x, expert_dataset.trajectories["latents"])
    print(np.mean(hd_array / len_array))

  if False:
    model_path = "/home/sangwon/Projects/ai_coach/test_algs/result/CleanupSingle-v0/oiql/oiqlstrm_256_3e-5_boundnnstd_extraD/2023-08-14_19-09-42/model/oiql_iq_CleanupSingle-v0_n10_l0_best"
    env_name = 'CleanupSingle-v0'
    agent, config = oiql_model_load(env_name, model_path)

    # load data
    num_data = 30
    data_path = "/home/sangwon/Projects/ai_coach/test_algs/experts/CleanupSingle-v0_100.pkl"
    expert_dataset, traj_labels, cnt_label = load_expert_data_w_labels(
        data_path, num_data, 0, 0)

    list_inferred_x = oiql_infer_latent(agent, expert_dataset.trajectories)
    hd_array, len_array = get_stats_about_x(
        list_inferred_x, expert_dataset.trajectories["latents"])
    print(np.mean(hd_array / len_array))
