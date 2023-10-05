from aic_ml.MentalIQL.agent.make_agent import make_miql_agent
from aic_ml.OptionIQL.agent.make_agent import make_oiql_agent
from aic_ml.baselines.option_gail.option_gail_learn import make_gail
from aic_ml.baselines.IQLearn.utils.utils import make_env
from aic_ml.MentalIQL.train_miql import (load_expert_data_w_labels)
import numpy as np
from aic_core.utils.result_utils import hamming_distance
from omegaconf import OmegaConf
import gym_custom
import gym
import torch
from gym.spaces import Discrete, Box
import os
import click


def get_s_a_dim(env: gym.Env):
  if isinstance(env.observation_space, Discrete):
    obs_dim = env.observation_space.n
    discrete_obs = True
  else:
    obs_dim = env.observation_space.shape[0]
    discrete_obs = False

  if isinstance(env.action_space, Discrete):
    action_dim = env.action_space.n
    discrete_act = True
  else:
    action_dim = env.action_space.shape[0]
    discrete_act = False

  return obs_dim, action_dim, discrete_obs, discrete_act


def infer_latent(agent, trajectories):
  list_inferred_x = []
  for i_e in range(len(trajectories["states"])):
    states = trajectories["states"][i_e]
    actions = trajectories["actions"][i_e]
    inferred_x, _ = agent.infer_mental_states(states, actions)
    list_inferred_x.append(inferred_x)

  return list_inferred_x


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


def load_model(config, env_obj, modelpath):
  agent = None
  alg = config.alg_name
  if alg == "miql":
    agent = make_miql_agent(config, env_obj)
    agent.load(modelpath)
  elif alg == "oiql":
    agent = make_oiql_agent(config, env_obj)
    agent.load(modelpath)
  elif alg == "ogail":
    dim_s, dim_a, discrete_s, discrete_a = get_s_a_dim(env_obj)
    agent, ppo = make_gail(config,
                           dim_s=dim_s,
                           dim_a=dim_a,
                           discrete_s=discrete_s,
                           discrete_a=discrete_a)
    param, filter_state = torch.load(modelpath)
    agent.load_state_dict(param)

  return agent


@click.command()
@click.option("--alg", type=str, default="miql", help="miql, oiql, ogail")
@click.option("--modelpath", type=str, default="", help="")
@click.option("--env", type=str, default="MultiGoals2D_2-v0", help="")
@click.option("--ndata", type=int, default=True, help="")
@click.option("--logroot", type=str, default="result", help="")
def main(alg, modelpath, env, ndata, logroot):
  datadir = "/home/sangwon/Projects/ai_coach/train_dnn/test_data/"
  resdir = f"/home/sangwon/Projects/ai_coach/train_dnn/{logroot}/{env}/{alg}/"
  modelpath = resdir + modelpath

  logdir = os.path.dirname(os.path.dirname(modelpath))

  config_path = os.path.join(logdir, "log/config.yaml")
  config = OmegaConf.load(config_path)

  # add updated keys
  if alg == "miql":
    if 'miql_tx_method_div' not in config.keys():
      config['miql_tx_method_div'] = ""
      print("Missing key - miql_tx_method_div is added as \"\".")
    if 'miql_pi_method_div' not in config.keys():
      config['miql_pi_method_div'] = ""
      print("Missing key - miql_pi_method_div is added as \"\".")
  elif alg == "oiql":
    if 'method_div' not in config.keys():
      config['method_div'] = ""
      print("Missing key - method_div is added as \"\".")

  env_name = env
  env_obj = make_env(env_name, env_make_kwargs={})

  # load model
  agent = load_model(config, env_obj, modelpath)

  # load data
  data_path = datadir + f"{env_name}_{ndata}.pkl"

  expert_dataset, traj_labels, cnt_label = load_expert_data_w_labels(
      data_path, ndata, 0, 0)

  list_inferred_x = infer_latent(agent, expert_dataset.trajectories)
  hd_array, len_array = get_stats_about_x(
      list_inferred_x, expert_dataset.trajectories["latents"])

  norm_hd = np.mean(hd_array / len_array)
  accuracy = 1 - np.sum(hd_array) / np.sum(len_array)
  print(f"{env_name}-{alg} - D_hamming_norm: {norm_hd} -- Accuracy: {accuracy}")


if __name__ == "__main__":
  main()

  if False:
    # load model
    env_name = "MultiGoals2D_2-v0"
    num_data = 50
    model_path = (
        "/home/sangwon/Projects/ai_coach/train_dnn/result/" +
        "MultiGoals2D_2-v0/miql/Ttx001Tpi001tol5Sv2/2023-09-20_10-23-11/" +
        "model/iq_MultiGoals2D_2-v0_n50_l10_best")

    main("miql", model_path, env_name, 50)

    model_path = ("/home/sangwon/Projects/ai_coach/train_dnn/result/" +
                  "MultiGoals2D_2-v0/ogail/tol5Sv2/2023-09-20_15-35-27/" +
                  "model/MultiGoals2D_2-v0_n50_l10_best.torch")

    main("ogail", model_path, env_name, 50)

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
