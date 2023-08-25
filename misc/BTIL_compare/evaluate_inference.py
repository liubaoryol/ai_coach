import os
import glob
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from aic_core.utils.decoding import (forward_inference,
                                                smooth_inference_zx,
                                                smooth_inference_max_z,
                                                smooth_inference_sa)
import click


@click.command()
@click.option("--domain-name", type=str, default="movers", help="")
@click.option("--num-train", type=int, default=500, help="")
@click.option("--supervision", type=float, default=0.3, help="")
@click.option("--alg", type=str, default='abs', help="")
@click.option("--gen-testset", type=bool, default=False, help="")
@click.option("--num-testset", type=int, default=100, help="")
@click.option("--fix-illegal", type=bool, default=False, help="")
@click.option("--num-x", type=int, default=4, help="")
@click.option("--num-abs", type=int, default=30, help="")
def prediction_result(domain_name: str,
                      num_train: int,
                      supervision: float,
                      alg: str,
                      gen_testset: bool = False,
                      num_testset: int = 100,
                      fix_illegal: bool = False,
                      num_x: int = 4,
                      num_abs: int = 30):
  sup_txt = ("%.2f" % supervision).replace('.', ',')
  if domain_name == "rescue_3":
    tx_dependency = "FTTTT"
  else:
    tx_dependency = "FTTT"

  if alg == "coord":
    mid_text = f"{tx_dependency}_{num_train}_{sup_txt}"
    policy1_file = (domain_name +
                    f"_btil_dec_policy_synth_woTx_{mid_text}_a1.npy")
    policy2_file = (domain_name +
                    f"_btil_dec_policy_synth_woTx_{mid_text}_a2.npy")
    policy3_file = (domain_name +
                    f"_btil_dec_policy_synth_woTx_{mid_text}_a3.npy")

    tx1_file = (domain_name + f"_btil_dec_tx_synth_{mid_text}_a1.npy")
    tx2_file = (domain_name + f"_btil_dec_tx_synth_{mid_text}_a2.npy")
    tx3_file = (domain_name + f"_btil_dec_tx_synth_{mid_text}_a3.npy")
    bx1_file = None
  elif alg == "svi":
    mid_text = f"{tx_dependency}_{num_train}"
    policy1_file = (domain_name + f"_btil_svi_policy_{mid_text}_f_a1.npy")
    policy2_file = (domain_name + f"_btil_svi_policy_{mid_text}_f_a2.npy")
    policy3_file = (domain_name + f"_btil_svi_policy_{mid_text}_f_a3.npy")

    tx1_file = (domain_name + f"_btil_svi_tx_{mid_text}__a1.npy")
    tx2_file = (domain_name + f"_btil_svi_tx_{mid_text}__a2.npy")
    tx3_file = (domain_name + f"_btil_svi_tx_{mid_text}__a3.npy")

    bx1_file = (domain_name + f"_btil_svi_bx_{mid_text}__a1.npy")
    bx2_file = (domain_name + f"_btil_svi_bx_{mid_text}__a2.npy")
    bx3_file = (domain_name + f"_btil_svi_bx_{mid_text}__a3.npy")
  elif alg == "abs" or alg == "maxz":
    mid_text = f"{tx_dependency}_{num_train}_{num_x}_{num_abs}"
    policy1_file = (domain_name + f"_btil_abs_{mid_text}_pi_a1.npy")
    policy2_file = (domain_name + f"_btil_abs_{mid_text}_pi_a2.npy")
    policy3_file = (domain_name + f"_btil_abs_{mid_text}_pi_a3.npy")

    tx1_file = (domain_name + f"_btil_abs_{mid_text}_tx_a1.npy")
    tx2_file = (domain_name + f"_btil_abs_{mid_text}_tx_a2.npy")
    tx3_file = (domain_name + f"_btil_abs_{mid_text}_tx_a3.npy")

    bx1_file = (domain_name + f"_btil_abs_{mid_text}_bx_a1.npy")
    bx2_file = (domain_name + f"_btil_abs_{mid_text}_bx_a2.npy")
    bx3_file = (domain_name + f"_btil_abs_{mid_text}_bx_a3.npy")
    abs_file = (domain_name + f"_btil_abs_{mid_text}_abs.npy")

  # =========== LOAD ENV ============
  if domain_name == "movers":
    from ai_coach_domain.box_push_v2.agent import BoxPushAIAgent_Team
    from ai_coach_domain.box_push_v2.maps import MAP_MOVERS
    from ai_coach_domain.box_push_v3.policy import Policy_MoversV3
    from ai_coach_domain.box_push_v3.mdp import MDP_MoversV3_Agent
    from ai_coach_domain.box_push_v3.mdp import MDP_MoversV3_Task
    from ai_coach_domain.box_push_v3.simulator import BoxPushSimulatorV3
    from ai_coach_domain.box_push.utils import BoxPushTrajectories
    game_map = MAP_MOVERS
    temperature = 0.3

    MDP_Task = MDP_MoversV3_Task(**game_map)
    MDP_Agent = MDP_MoversV3_Agent(**game_map)

    policy1 = Policy_MoversV3(MDP_Task, MDP_Agent, temperature, 0)
    policy2 = Policy_MoversV3(MDP_Task, MDP_Agent, temperature, 1)
    agent_1 = BoxPushAIAgent_Team(policy1, agent_idx=0)
    agent_2 = BoxPushAIAgent_Team(policy2, agent_idx=1)
    agents = [agent_1, agent_2]

    game = BoxPushSimulatorV3(False)
    test_data_handler = BoxPushTrajectories(MDP_Task, MDP_Agent)

  game.init_game(**game_map)
  game.set_autonomous_agent(*agents)

  data_dir = os.path.join(os.path.dirname(__file__), "data/")
  model_dir = os.path.join(data_dir, "learned_models/")

  num_agents = len(agents)
  tup_num_latents = tuple([MDP_Agent.num_latents] * num_agents)

  # =========== LOAD MODELS ============
  np_policy1 = np.load(model_dir + policy1_file)
  np_tx1 = np.load(model_dir + tx1_file)
  np_policy2 = np.load(model_dir + policy2_file)
  np_tx2 = np.load(model_dir + tx2_file)
  list_np_policy = [np_policy1, np_policy2]
  list_np_tx = [np_tx1, np_tx2]
  if bx1_file is None:
    list_bx = [
        np.ones((list_np_policy[0].shape[1], tup_num_latents[idx])) /
        tup_num_latents[idx] for idx in range(game.get_num_agents())
    ]
  else:
    np_bx1 = np.load(model_dir + bx1_file)
    np_bx2 = np.load(model_dir + bx2_file)
    list_bx = [np_bx1, np_bx2]

  if alg == "abs" or alg == "maxz":
    np_abs = np.load(model_dir + abs_file)

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

  test_data_handler.load_from_files(file_names)
  list_trajs = test_data_handler.get_as_column_lists(include_terminal=True)

  # =========== INFERENCE ============
  prediction_results = []
  for epi in tqdm(list_trajs):
    list_states, list_actions, list_latents = epi

    if alg == "abs":
      np_pzx = smooth_inference_zx(list_states, list_actions, num_agents,
                                   tup_num_latents, num_abs, np_abs,
                                   list_np_policy, list_np_tx, list_bx)

      max_idx = np_pzx.reshape(np_pzx.shape[0], -1).argmax(1)
      max_coords = np.unravel_index(max_idx, np_pzx.shape[1:])
      max_coords = list(zip(*max_coords))
      np_result = np.zeros((num_agents, np_pzx.shape[0]))
      for a_idx in range(num_agents):
        for t, z_x in enumerate(max_coords):
          np_result[a_idx, t] = int(z_x[a_idx + 1] == list_latents[t][a_idx])

      # for t in range(np_pzx.shape[0]):
      #   np_zx_cur = np_pzx[t]
      #   max_zx = np.max(np_zx_cur)
      #   list_same_idx = np.argwhere(np_zx_cur.reshape(-1) == max_zx)
      #   xhat = random.choice(list_same_idx)[0]
      #   joint_idx = np.unravel_index(xhat, np_zx_cur.shape)
      #   for a_idx in range(num_agents):
      #     np_result[a_idx,
      #               t] = int(joint_idx[a_idx + 1] == list_latents[t][a_idx])

      epi_accuracy = np.array(np_result, dtype=np.int32).mean()
    elif alg == "maxz":
      list_zx = smooth_inference_max_z(list_states, list_actions, num_agents,
                                       tup_num_latents, num_abs, np_abs,
                                       list_np_policy, list_np_tx, list_bx)
      traj_len = len(list_zx[1])
      np_result = np.zeros((num_agents, traj_len))
      for a_idx in range(num_agents):
        for t in range(traj_len):
          np_px = list_zx[a_idx + 1][t]
          list_same_idx = np.argwhere(np_px == np.max(np_px))
          xhat = random.choice(list_same_idx)[0]
          np_result[a_idx, t] = int(xhat == list_latents[t][a_idx])

      epi_accuracy = np.array(np_result, dtype=np.int32).mean()
    else:
      list_np_px = smooth_inference_sa(list_states, list_actions, num_agents,
                                       tup_num_latents, list_np_policy,
                                       list_np_tx, list_bx)
      np_result = np.zeros((len(list_np_px), list_np_px[0].shape[0]))
      for a_idx, np_px in enumerate(list_np_px):
        for t in range(np_px.shape[0]):
          list_same_idx = np.argwhere(np_px[t] == np.max(np_px[t]))
          xhat = random.choice(list_same_idx)[0]
          np_result[a_idx, t] = int(xhat == list_latents[t][a_idx])
      epi_accuracy = np.array(np_result, dtype=np.int32).mean()

    prediction_results.append(epi_accuracy)

  print(np.array(prediction_results).mean())
  return prediction_results


if __name__ == "__main__":
  res = prediction_result()
