import os
from ai_coach_core.model_learning.MentalIQL.agent.make_agent import (
    make_miql_agent)
from aicoach_baselines.option_gail.utils.config import Config
from ai_coach_core.model_learning.IQLearn.utils.utils import make_env
from ai_coach_core.model_learning.MentalIQL.train_miql import (
    load_expert_data_w_labels)
import gym_custom
import importlib.util
import sys
import numpy as np
from ai_coach_core.utils.result_utils import norm_hamming_distance

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


if __name__ == "__main__":

  # load model
  model_path = "/home/sangwon/Projects/ai_coach/test_algs/result/EnvMovers-v0/miql/miql_256_3e-5_value/2023-08-15_00-53-14/model/miql_iq_EnvMovers-v0_n44_l44_best"
  env_name = "EnvMovers-v0"

  agent, config = miql_model_load(env_name, model_path)

  # load data
  num_data = 22
  data_path = "/home/sangwon/Projects/ai_coach/test_algs/experts/EnvMovers_v0_22.pkl"
  expert_dataset, traj_labels, cnt_label = load_expert_data_w_labels(
      "/home/sangwon/Projects/ai_coach/test_algs/experts/EnvMovers_v0_22.pkl",
      num_data, 0, 0)

  dis_array = []
  for i_e in range(len(expert_dataset.trajectories["states"])):
    states = expert_dataset.trajectories["states"][i_e]
    actions = expert_dataset.trajectories["actions"][i_e]
    inferred_x, _ = agent.infer_mental_states(states, actions)
    res = norm_hamming_distance(inferred_x,
                                expert_dataset.trajectories["latents"][i_e])
    dis_array.append(res)

  print(np.mean(dis_array))
