import os
import glob
import numpy as np
import random
import pandas as pd
from ai_coach_core.latent_inference.decoding import forward_inference


def prediction_result(domain_name: str,
                      num_train: int,
                      supervision: float,
                      gen_testset: bool = False,
                      num_testset: int = 100,
                      fix_illegal: bool = False):
  sup_txt = ("%.2f" % supervision).replace('.', ',')
  policy1_file = domain_name + f"_btil2_policy_synth_woTx_FTTT_{num_train}_{sup_txt}_a1.npy"
  policy2_file = domain_name + f"_btil2_policy_synth_woTx_FTTT_{num_train}_{sup_txt}_a2.npy"
  tx1_file = domain_name + f"_btil2_tx_synth_FTTT_{num_train}_{sup_txt}_a1.npy"
  tx2_file = domain_name + f"_btil2_tx_synth_FTTT_{num_train}_{sup_txt}_a2.npy"

  # =========== LOAD ENV ============
  if domain_name == "movers":
    from ai_coach_domain.box_push_v2.agent import BoxPushAIAgent_PO_Team
    from ai_coach_domain.box_push_v2.maps import MAP_MOVERS
    from ai_coach_domain.box_push_v2.policy import Policy_Movers
    from ai_coach_domain.box_push_v2.mdp import MDP_Movers_Agent
    from ai_coach_domain.box_push_v2.mdp import MDP_Movers_Task
    from ai_coach_domain.box_push_v2.simulator import BoxPushSimulatorV2
    from ai_coach_domain.box_push.utils import BoxPushTrajectories
    game_map = MAP_MOVERS
    temperature = 0.3

    MDP_Task = MDP_Movers_Task(**game_map)
    MDP_Agent = MDP_Movers_Agent(**game_map)

    init_state = ([0] * len(game_map["boxes"]), game_map["a1_init"],
                  game_map["a2_init"])

    policy1 = Policy_Movers(MDP_Task, MDP_Agent, temperature, 0)
    policy2 = Policy_Movers(MDP_Task, MDP_Agent, temperature, 1)
    agent_1 = BoxPushAIAgent_PO_Team(init_state, policy1, agent_idx=0)
    agent_2 = BoxPushAIAgent_PO_Team(init_state, policy2, agent_idx=1)

    game = BoxPushSimulatorV2(0)
    test_data_handler = BoxPushTrajectories(MDP_Task, MDP_Agent)

  elif domain_name == "cleanup_v2":
    from ai_coach_domain.box_push_v2.agent import BoxPushAIAgent_PO_Indv
    from ai_coach_domain.box_push_v2.maps import MAP_CLEANUP_V2
    from ai_coach_domain.box_push_v2.policy import Policy_Cleanup
    from ai_coach_domain.box_push_v2.mdp import MDP_Cleanup_Agent
    from ai_coach_domain.box_push_v2.mdp import MDP_Cleanup_Task
    from ai_coach_domain.box_push_v2.simulator import BoxPushSimulatorV2
    from ai_coach_domain.box_push.utils import BoxPushTrajectories
    game_map = MAP_CLEANUP_V2
    temperature = 0.3

    MDP_Task = MDP_Cleanup_Task(**game_map)
    MDP_Agent = MDP_Cleanup_Agent(**game_map)

    init_state = ([0] * len(game_map["boxes"]), game_map["a1_init"],
                  game_map["a2_init"])

    policy1 = Policy_Cleanup(MDP_Task, MDP_Agent, temperature, 0)
    policy2 = Policy_Cleanup(MDP_Task, MDP_Agent, temperature, 1)
    agent_1 = BoxPushAIAgent_PO_Indv(init_state, policy1, agent_idx=0)
    agent_2 = BoxPushAIAgent_PO_Indv(init_state, policy2, agent_idx=1)

    game = BoxPushSimulatorV2(0)
    test_data_handler = BoxPushTrajectories(MDP_Task, MDP_Agent)

  elif domain_name == "rescue_2":
    from ai_coach_domain.rescue.agent import AIAgent_Rescue_PartialObs
    from ai_coach_domain.rescue.maps import MAP_RESCUE
    from ai_coach_domain.rescue.policy import Policy_Rescue
    from ai_coach_domain.rescue.mdp import MDP_Rescue_Agent, MDP_Rescue_Task
    from ai_coach_domain.rescue.simulator import RescueSimulator
    from ai_coach_domain.rescue.utils import RescueTrajectories
    game_map = MAP_RESCUE
    temperature = 0.3

    MDP_Task = MDP_Rescue_Task(**game_map)
    MDP_Agent = MDP_Rescue_Agent(**game_map)

    init_states = ([1] * len(game_map["work_locations"]), game_map["a1_init"],
                   game_map["a2_init"])
    policy1 = Policy_Rescue(MDP_Task, MDP_Agent, temperature, 0)
    policy2 = Policy_Rescue(MDP_Task, MDP_Agent, temperature, 1)
    agent_1 = AIAgent_Rescue_PartialObs(init_states, 0, policy1)
    agent_2 = AIAgent_Rescue_PartialObs(init_states, 1, policy2)

    game = RescueSimulator()
    game.max_steps = 30

    def conv_latent_to_idx(agent_idx, latent):
      if agent_idx == 0:
        return agent_1.conv_latent_to_idx(latent)
      else:
        return agent_2.conv_latent_to_idx(latent)

    test_data_handler = RescueTrajectories(
        MDP_Task, (MDP_Agent.num_latents, MDP_Agent.num_latents),
        conv_latent_to_idx)

  game.init_game(**game_map)
  game.set_autonomous_agent(agent_1, agent_2)

  data_dir = os.path.join(os.path.dirname(__file__), "data/")
  model_dir = os.path.join(data_dir, "learned_models/")

  # =========== LOAD MODELS ============
  np_policy1 = np.load(model_dir + policy1_file)
  np_tx1 = np.load(model_dir + tx1_file)
  np_policy2 = np.load(model_dir + policy2_file)
  np_tx2 = np.load(model_dir + tx2_file)

  list_np_policy = [np_policy1, np_policy2]
  list_np_tx = [np_tx1, np_tx2]

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
    np_dist = list_np_tx[nidx][xidx, tuple_aidx[0], tuple_aidx[1], sidx_n]

    # NOTE: for illegal or unencountered states,
    #       we assume mental model was maintained.
    if fix_illegal:
      if np.all(np_dist == np_dist[0]):
        np_dist = np.zeros_like(np_dist)
        np_dist[xidx] = 1

    return np_dist[xidx_n]

  def init_latent_nxs(nidx, xidx, sidx):
    if nidx == 0:
      agent = agent_1
    else:
      agent = agent_2

    num_latents = agent.agent_model.policy_model.get_num_latent_states()
    return 1 / num_latents  # uniform

  prediction_results = []
  list_prev_np_x_dist = None
  for epi in list_trajs:
    list_states, list_actions, list_latents = epi

    episode_result = []
    for t in range(len(list_actions) - 1):
      _, list_np_x_dist = forward_inference(list_states[:t + 2],
                                            list_actions[:t + 1],
                                            game.get_num_agents(),
                                            MDP_Agent.num_latents, policy_nxsa,
                                            Tx_nxsasx, init_latent_nxs,
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
  domains = ["movers", "cleanup_v2", "rescue_2"]
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
  df.to_csv(data_dir + "eval_result.csv", index=False)
