import os
import glob
import numpy as np
import random
import pandas as pd
from aic_core.utils.decoding import forward_inference
import load_domain


def prediction_result(domain_name: str,
                      num_train: int,
                      supervision: float,
                      gen_testset: bool = False,
                      num_testset: int = 100,
                      fix_illegal: bool = False):
  sup_txt = ("%.2f" % supervision).replace('.', ',')
  if domain_name == "rescue_3":
    tx_dependency = "FTTTT"
  else:
    tx_dependency = "FTTT"

  policy1_file = (
      domain_name +
      f"_btil2_policy_synth_woTx_{tx_dependency}_{num_train}_{sup_txt}_a1.npy")
  policy2_file = (
      domain_name +
      f"_btil2_policy_synth_woTx_{tx_dependency}_{num_train}_{sup_txt}_a2.npy")
  policy3_file = (
      domain_name +
      f"_btil2_policy_synth_woTx_{tx_dependency}_{num_train}_{sup_txt}_a3.npy")

  tx1_file = (domain_name +
              f"_btil2_tx_synth_{tx_dependency}_{num_train}_{sup_txt}_a1.npy")
  tx2_file = (domain_name +
              f"_btil2_tx_synth_{tx_dependency}_{num_train}_{sup_txt}_a2.npy")
  tx3_file = (domain_name +
              f"_btil2_tx_synth_{tx_dependency}_{num_train}_{sup_txt}_a3.npy")

  # =========== LOAD ENV ============
  if domain_name == "movers":
    vec_domain_data = load_domain.load_movers()
  elif domain_name == "cleanup_v2":
    vec_domain_data = load_domain.load_cleanup_v2()
  elif domain_name == "cleanup_v3":
    vec_domain_data = load_domain.load_cleanup_v3()
  elif domain_name == "rescue_2":
    vec_domain_data = load_domain.load_rescue_2()
  elif domain_name == "rescue_3":
    vec_domain_data = load_domain.load_rescue_3()
  else:
    raise NotImplementedError

  game, agents, _, test_data_handler, game_map = vec_domain_data

  game.init_game(**game_map)
  game.set_autonomous_agent(*agents)

  data_dir = os.path.join(os.path.dirname(__file__), "data/")
  model_dir = os.path.join(data_dir, "learned_models/")

  # =========== LOAD MODELS ============
  np_policy1 = np.load(model_dir + policy1_file)
  np_tx1 = np.load(model_dir + tx1_file)
  np_policy2 = np.load(model_dir + policy2_file)
  np_tx2 = np.load(model_dir + tx2_file)
  list_np_policy = [np_policy1, np_policy2]
  list_np_tx = [np_tx1, np_tx2]

  if game.get_num_agents() > 2:
    np_policy3 = np.load(model_dir + policy3_file)
    np_tx3 = np.load(model_dir + tx3_file)

    list_np_policy.append(np_policy3)
    list_np_tx.append(np_tx3)

  # =========== GEN DATA ============
  test_dir = os.path.join(data_dir, domain_name + "_test/")
  test_prefix = "test_"
  if gen_testset:
    file_names = glob.glob(os.path.join(test_dir, test_prefix + '*.txt'))
    for fmn in file_names:
      os.remove(fmn)
    game.run_simulation(num_testset, os.path.join(test_dir, test_prefix),
                        "header")

  # =========== LOAD DATA ============
  file_names = glob.glob(os.path.join(test_dir, "*.txt"))
  random.shuffle(file_names)

  test_data_handler.load_from_files(file_names)
  list_trajs = test_data_handler.get_as_column_lists(include_terminal=True)

  # =========== INFERENCE ============
  def policy_nxsa(nidx, xidx, sidx, tuple_aidx):
    return list_np_policy[nidx][xidx, sidx, tuple_aidx[nidx]]

  def Tx_nxsasx(nidx, xidx, sidx, tuple_aidx, sidx_n, xidx_n):
    np_idx = tuple([xidx, *tuple_aidx, sidx_n])
    np_dist = list_np_tx[nidx][np_idx]

    # NOTE: for illegal or unencountered states,
    #       we assume mental model was maintained.
    if fix_illegal:
      if np.all(np_dist == np_dist[0]):
        np_dist = np.zeros_like(np_dist)
        np_dist[xidx] = 1

    return np_dist[xidx_n]

  def init_latent_nxs(nidx, xidx, sidx):
    agent = agents[nidx]

    num_latents = agent.agent_model.policy_model.get_num_latent_states()
    return 1 / num_latents  # uniform

  prediction_results = []
  list_prev_np_x_dist = None
  for epi in list_trajs:
    list_states, list_actions, list_latents = epi

    episode_result = []
    for t in range(len(list_actions) - 1):
      num_latents = agents[0].agent_model.policy_model.get_num_latent_states()
      _, list_np_x_dist = forward_inference(list_states[:t + 2],
                                            list_actions[:t + 1],
                                            game.get_num_agents(), num_latents,
                                            policy_nxsa, Tx_nxsasx,
                                            init_latent_nxs,
                                            list_prev_np_x_dist)
      list_prev_np_x_dist = list_np_x_dist
      list_result = []
      for idx, np_dist in enumerate(list_np_x_dist):
        list_same_idx = np.argwhere(np_dist == np.max(np_dist))
        xhat = random.choice(list_same_idx)[0]
        list_result.append(xhat == list_latents[t + 1][idx])

      episode_result.append(tuple(list_result))

    epi_accuracy = np.array(episode_result, dtype=np.int32).mean()
    prediction_results.append(epi_accuracy)

  return prediction_results


if __name__ == "__main__":
  DO_TEST = False
  if DO_TEST:
    res = prediction_result("rescue_2", 500, 0.3, True, 100, False)
    print(np.array(res).mean())

    raise RuntimeError

  domains = ["movers", "cleanup_v3", "rescue_2", "rescue_3"]
  train_setup = [(150, 1), (500, 0.3), (500, 1)]

  rows = []
  for dom in domains:
    for num_data, supervision in train_setup:
      list_predictions = prediction_result(dom,
                                           num_data,
                                           supervision,
                                           fix_illegal=False)
      rows = rows + [(dom, "%d_%d%%" % (num_data, int(supervision * 100)), item)
                     for item in list_predictions]

  df = pd.DataFrame(rows, columns=['domain', 'train_setup', 'value'])

  data_dir = os.path.join(os.path.dirname(__file__), "data/")
  df.to_csv(data_dir + "eval_result3.csv", index=False)
