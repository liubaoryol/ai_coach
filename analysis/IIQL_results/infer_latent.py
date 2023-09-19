from aic_ml.MentalIQL.agent.make_agent import (make_miql_agent)
from aic_ml.OptionIQL.agent.make_agent import (make_oiql_agent)
from aic_ml.baselines.IQLearn.utils.utils import make_env
from aic_ml.MentalIQL.train_miql import (load_expert_data_w_labels)
import numpy as np
from aic_core.utils.result_utils import hamming_distance
from aic_ml.OptionIQL.helper.utils import (infer_mental_states)
from omegaconf import OmegaConf


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

  if True:
    # load model
    log_path = (
        "/home/sangwon/Projects/ai_coach/train_dnn/result/" +
        "MultiGoals2D_2-v0/miql/Ttx001Tpi001valSv0/2023-09-18_10-14-11/")
    config_path = log_path + "log/config.yaml"
    model_path = log_path + "model/miql_iq_EnvMovers-v0_n44_l44_best"
    env_name = "MultiGoals2D_2-v0"
    config = OmegaConf.load(config_path)

    env = make_env(env_name, env_make_kwargs={})
    agent = make_miql_agent(config, env)
    agent.load(model_path)

    # load data
    num_data = 22
    data_path = "/home/sangwon/Projects/ai_coach/train_dnn/experts/EnvMovers_v0_22.pkl"
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
