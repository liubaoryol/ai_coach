import os
import numpy as np
import pickle
from tqdm import tqdm
from aic_core.intervention.feedback_strategy import (InterventionValueBased,
                                                     InterventionRuleBased,
                                                     E_CertaintyHandling)
import ai_coach_domain.intervention_simulator as intervention_simulator
import pandas as pd


def intervention_result(domain_name,
                        num_runs,
                        no_intervention,
                        selection_type: str,
                        certainty_type: str,
                        infer_thres,
                        interv_thres,
                        cost,
                        fix_illegal=True,
                        increase_step=False):
  data_dir = os.path.join(os.path.dirname(__file__), "data/")
  model_dir = os.path.join(data_dir, "learned_models/")

  e_certainty = E_CertaintyHandling[
      certainty_type] if certainty_type is not None else None
  # theta = 0.5
  # delta = 5
  # cost = 0
  theta = infer_thres
  delta = interv_thres

  num_train = 500
  supervision = 0.3
  sup_txt = ("%.2f" % supervision).replace('.', ',')

  if domain_name == "rescue_2":
    iteration = 30
  elif domain_name == "rescue_3":
    iteration = 15
  else:
    iteration = 500

  v_value_file_name = (
      domain_name +
      f"_{num_train}_{sup_txt}_{iteration}_merged_v_values_learned.pickle")
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

  if domain_name == "movers":
    from ai_coach_domain.box_push_v2.agent import BoxPushAIAgent_PO_Team
    from ai_coach_domain.box_push_v2.maps import MAP_MOVERS
    from ai_coach_domain.box_push_v2.policy import Policy_Movers
    from ai_coach_domain.box_push_v2.mdp import MDP_Movers_Agent
    from ai_coach_domain.box_push_v2.mdp import MDP_Movers_Task
    from ai_coach_domain.box_push_v2.simulator import BoxPushSimulatorV2
    from ai_coach_domain.box_push.agent_model import (
        get_holding_box_and_floor_boxes)
    game_map = MAP_MOVERS
    MDP_Task = MDP_Movers_Task(**game_map)
    MDP_Agent = MDP_Movers_Agent(**game_map)

    temperature = 0.3
    policy1 = Policy_Movers(MDP_Task, MDP_Agent, temperature, 0)
    policy2 = Policy_Movers(MDP_Task, MDP_Agent, temperature, 1)

    init_state = ([0] * len(game_map["boxes"]), game_map["a1_init"],
                  game_map["a2_init"])
    agent1 = BoxPushAIAgent_PO_Team(init_state, policy1, agent_idx=0)
    agent2 = BoxPushAIAgent_PO_Team(init_state, policy2, agent_idx=1)
    agents = [agent1, agent2]
    game = BoxPushSimulatorV2(0)

    def get_state_action(history):
      step, bstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat = history
      return (bstt, a1pos, a2pos), (a1act, a2act)

    def valid_latent1(obs_idx):
      num_latents = agent1.agent_model.policy_model.get_num_latent_states()
      return list(range(num_latents))

    def valid_latent2(obs_idx):
      box_states, _, _ = MDP_Task.conv_mdp_sidx_to_sim_states(obs_idx)
      num_drops = len(MDP_Task.drops)
      num_goals = len(MDP_Task.goals)
      a1_box, a2_box, valid_box = get_holding_box_and_floor_boxes(
          box_states, num_drops, num_goals)
      list_valid_lat = []
      if a1_box > -1 and a1_box == a2_box:
        lat = agent1.agent_model.policy_model.conv_latent_to_idx(("goal", 0))
        list_valid_lat.append(lat)
      else:
        for idx in valid_box:
          lat = agent1.agent_model.policy_model.conv_latent_to_idx(
              ("pickup", idx))
          list_valid_lat.append(lat)
      return list_valid_lat

    fn_valid_latent = valid_latent2

  elif domain_name == "cleanup_v2":
    from ai_coach_domain.box_push_v2.agent import BoxPushAIAgent_PO_Indv
    from ai_coach_domain.box_push_v2.maps import MAP_CLEANUP_V2
    from ai_coach_domain.box_push_v2.policy import Policy_Cleanup
    from ai_coach_domain.box_push_v2.mdp import MDP_Cleanup_Agent
    from ai_coach_domain.box_push_v2.mdp import MDP_Cleanup_Task
    from ai_coach_domain.box_push_v2.simulator import BoxPushSimulatorV2
    game_map = MAP_CLEANUP_V2
    MDP_Task = MDP_Cleanup_Task(**game_map)
    MDP_Agent = MDP_Cleanup_Agent(**game_map)

    temperature = 0.3
    policy1 = Policy_Cleanup(MDP_Task, MDP_Agent, temperature, 0)
    policy2 = Policy_Cleanup(MDP_Task, MDP_Agent, temperature, 1)

    init_state = ([0] * len(game_map["boxes"]), game_map["a1_init"],
                  game_map["a2_init"])
    agent1 = BoxPushAIAgent_PO_Indv(init_state, policy1, agent_idx=0)
    agent2 = BoxPushAIAgent_PO_Indv(init_state, policy2, agent_idx=1)
    agents = [agent1, agent2]
    game = BoxPushSimulatorV2(0)

    def get_state_action(history):
      step, bstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat = history
      return (bstt, a1pos, a2pos), (a1act, a2act)

  elif domain_name == "cleanup_v3":
    from ai_coach_domain.box_push_v2.agent import BoxPushAIAgent_PO_Indv
    from ai_coach_domain.box_push_v2.maps import MAP_CLEANUP_V3
    from ai_coach_domain.box_push_v2.policy import Policy_Cleanup
    from ai_coach_domain.box_push_v2.mdp import MDP_Cleanup_Agent
    from ai_coach_domain.box_push_v2.mdp import MDP_Cleanup_Task
    from ai_coach_domain.box_push_v2.simulator import BoxPushSimulatorV2
    game_map = MAP_CLEANUP_V3
    MDP_Task = MDP_Cleanup_Task(**game_map)
    MDP_Agent = MDP_Cleanup_Agent(**game_map)

    temperature = 0.3
    policy1 = Policy_Cleanup(MDP_Task, MDP_Agent, temperature, 0)
    policy2 = Policy_Cleanup(MDP_Task, MDP_Agent, temperature, 1)

    init_state = ([0] * len(game_map["boxes"]), game_map["a1_init"],
                  game_map["a2_init"])
    agent1 = BoxPushAIAgent_PO_Indv(init_state, policy1, agent_idx=0)
    agent2 = BoxPushAIAgent_PO_Indv(init_state, policy2, agent_idx=1)
    agents = [agent1, agent2]
    game = BoxPushSimulatorV2(0)

    def get_state_action(history):
      step, bstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat = history
      return (bstt, a1pos, a2pos), (a1act, a2act)
  elif domain_name == "rescue_2":
    from ai_coach_domain.rescue.agent import AIAgent_Rescue_PartialObs
    from ai_coach_domain.rescue.maps import MAP_RESCUE
    from ai_coach_domain.rescue.policy import Policy_Rescue
    from ai_coach_domain.rescue.mdp import MDP_Rescue_Agent, MDP_Rescue_Task
    from ai_coach_domain.rescue.simulator import RescueSimulator
    game_map = MAP_RESCUE
    temperature = 0.3

    MDP_Task = MDP_Rescue_Task(**game_map)
    MDP_Agent = MDP_Rescue_Agent(**game_map)

    init_states = ([1] * len(game_map["work_locations"]), game_map["a1_init"],
                   game_map["a2_init"])
    policy1 = Policy_Rescue(MDP_Task, MDP_Agent, temperature, 0)
    policy2 = Policy_Rescue(MDP_Task, MDP_Agent, temperature, 1)
    agent1 = AIAgent_Rescue_PartialObs(init_states, 0, policy1)
    agent2 = AIAgent_Rescue_PartialObs(init_states, 1, policy2)
    agents = [agent1, agent2]

    game = RescueSimulator()
    game.max_steps = 30

    def get_state_action(history):
      step, score, wstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat = history
      return (wstt, a1pos, a2pos), (a1act, a2act)
  elif domain_name == "rescue_3":
    from ai_coach_domain.rescue_v2.agent import AIAgent_Rescue_PartialObs
    from ai_coach_domain.rescue_v2.maps import MAP_RESCUE
    from ai_coach_domain.rescue_v2.policy import Policy_Rescue
    from ai_coach_domain.rescue_v2.mdp import MDP_Rescue_Agent, MDP_Rescue_Task
    from ai_coach_domain.rescue_v2.simulator import RescueSimulatorV2
    game_map = MAP_RESCUE
    temperature = 0.3

    MDP_Task = MDP_Rescue_Task(**game_map)
    MDP_Agent = MDP_Rescue_Agent(**game_map)

    init_states = ([1] * len(game_map["work_locations"]), game_map["a1_init"],
                   game_map["a2_init"], game_map["a3_init"])
    policy1 = Policy_Rescue(MDP_Task, MDP_Agent, temperature, 0)
    policy2 = Policy_Rescue(MDP_Task, MDP_Agent, temperature, 1)
    policy3 = Policy_Rescue(MDP_Task, MDP_Agent, temperature, 2)
    agent1 = AIAgent_Rescue_PartialObs(init_states, 0, policy1)
    agent2 = AIAgent_Rescue_PartialObs(init_states, 1, policy2)
    agent3 = AIAgent_Rescue_PartialObs(init_states, 2, policy3)
    agents = [agent1, agent2, agent3]

    game = RescueSimulatorV2()
    game.max_steps = 15

    def get_state_action(history):
      (step, score, wstt, a1pos, a2pos, a3pos, a1act, a2act, a3act, a1lat,
       a2lat, a3lat) = history
      return (wstt, a1pos, a2pos, a3pos), (a1act, a2act, a3act)
  else:
    raise NotImplementedError(domain_name)

  with open(data_dir + v_value_file_name, 'rb') as handle:
    np_v_values = pickle.load(handle)

  np_policy1 = np.load(model_dir + policy1_file)
  np_tx1 = np.load(model_dir + tx1_file)
  np_policy2 = np.load(model_dir + policy2_file)
  np_tx2 = np.load(model_dir + tx2_file)

  list_np_policy = [np_policy1, np_policy2]
  list_np_tx = [np_tx1, np_tx2]
  if len(agents) == 3:
    np_policy3 = np.load(model_dir + policy3_file)
    np_tx3 = np.load(model_dir + tx3_file)
    list_np_policy.append(np_policy3)
    list_np_tx.append(np_tx3)

  if no_intervention:
    intervention_strategy = None
  else:
    if selection_type == "Value":
      intervention_strategy = InterventionValueBased(
          np_v_values,
          e_certainty,
          inference_threshold=theta,
          intervention_threshold=delta,
          intervention_cost=cost)
    elif selection_type == "Rule":
      intervention_strategy = InterventionRuleBased(
          fn_valid_latent,
          len(agents),
          e_certainty,
          inference_threshold=theta,
          intervention_threshold=delta,
          intervention_cost=cost)
    else:
      raise NotImplementedError

  game.init_game(**game_map)
  game.set_autonomous_agent(*agents)
  sim = intervention_simulator.InterventionSimulator(
      game,
      list_np_policy,
      list_np_tx,
      intervention_strategy,
      get_state_action,
      fix_illegal,
      increase_step=increase_step)

  list_score, list_num_feedback = sim.run_game(num_runs)
  return list(zip(list_score, list_num_feedback))


if __name__ == "__main__":

  num_runs = 100

  VALUE = "Value"
  RULE = "Rule"
  AVERAGE = "Average"  # Average | Threshold
  THRESHOLD = "Threshold"
  NO_INTERVENTION = True
  INTERVENTION = False

  DO_TEST = False

  if DO_TEST:
    domain_name = "movers"
    list_res = intervention_result(domain_name, num_runs, INTERVENTION, VALUE,
                                   AVERAGE, 0, 0.2, 0)
    print(np.array(list_res).mean(axis=0))

    raise RuntimeError

  rows = []

  list_cost = [0, 0.2, 0.5, 1]
  # cost = list_cost[0]
  list_infer_thres = [0, 0.2, 0.3, 0.5, 0.7, 0.9]
  domains = ["movers", "cleanup_v3", "rescue_2", "rescue_3"]
  dict_interv_thres = {
      domains[0]: [0, 1, 3, 5, 10, 15, 20, 30, 50],
      domains[1]: [0, 0.3, 0.5, 1.0, 2.0, 5.0, 10, 15, 20],
      domains[2]: [0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
      domains[3]: [0, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0],
  }

  for cost in list_cost:
    for domain_name in domains:
      if cost == 1:
        list_increase_step = [False, True]
      else:
        list_increase_step = [False]
      for increase_step in list_increase_step:
        prefix = ""
        if increase_step:
          prefix = "_budget"

        print(domain_name)
        list_interv_thres = dict_interv_thres[domain_name]
        # ====== rule-based intervention
        if domain_name == "movers":
          num_total_setup = 2 + 2 * len(list_interv_thres) + 2 * len(
              list_infer_thres)
          progress_bar = tqdm(total=num_total_setup)
          for theta in list_infer_thres:
            list_res = intervention_result(domain_name,
                                           num_runs,
                                           INTERVENTION,
                                           RULE,
                                           THRESHOLD,
                                           theta,
                                           None,
                                           cost,
                                           increase_step=increase_step)

            rows = rows + [
                (domain_name, "Rule_thres" + prefix, cost, 0, theta, *item)
                for item in list_res
            ]
            progress_bar.update()

          list_res = intervention_result(domain_name,
                                         num_runs,
                                         INTERVENTION,
                                         RULE,
                                         AVERAGE,
                                         None,
                                         None,
                                         cost,
                                         increase_step=increase_step)
          progress_bar.update()

          rows = rows + [(domain_name, "Rule_avg" + prefix, cost, 0, 0, *item)
                         for item in list_res]
        else:
          num_total_setup = 1 + 2 * len(list_interv_thres) + len(
              list_infer_thres)
          progress_bar = tqdm(total=num_total_setup)
        # ===== Baseline: No intervention
        list_res = intervention_result(domain_name,
                                       num_runs,
                                       NO_INTERVENTION,
                                       None,
                                       None,
                                       None,
                                       None,
                                       cost,
                                       increase_step=increase_step)

        rows = rows + [
            (domain_name, "No_intervention" + prefix, cost, 0, 0, *item)
            for item in list_res
        ]
        progress_bar.update()

        # ===== Strategy 1: Stochastic
        for delta in list_interv_thres:
          list_res = intervention_result(domain_name,
                                         num_runs,
                                         INTERVENTION,
                                         VALUE,
                                         AVERAGE,
                                         None,
                                         delta,
                                         cost,
                                         increase_step=increase_step)

          rows = rows + [
              (domain_name, "Average" + prefix, cost, delta, 0, *item)
              for item in list_res
          ]
          progress_bar.update()

        # ===== Strategy 2: Deterministic
        for delta in list_interv_thres:
          list_res = intervention_result(domain_name,
                                         num_runs,
                                         INTERVENTION,
                                         VALUE,
                                         THRESHOLD,
                                         0,
                                         delta,
                                         cost,
                                         increase_step=increase_step)

          rows = rows + [(domain_name, "Argmax" + prefix, cost, delta, 0, *item)
                         for item in list_res]
          progress_bar.update()

        delta_s3 = list_interv_thres[3]
        # ===== Strategy 3: Deterministic with threshold
        for theta in list_infer_thres:
          list_res = intervention_result(domain_name,
                                         num_runs,
                                         INTERVENTION,
                                         VALUE,
                                         THRESHOLD,
                                         theta,
                                         delta_s3,
                                         cost,
                                         increase_step=increase_step)

          rows = rows + [(domain_name, "Argmax_thres" + prefix, cost, delta_s3,
                          theta, *item) for item in list_res]
          progress_bar.update()
        progress_bar.close()

  df = pd.DataFrame(rows,
                    columns=[
                        'domain', 'strategy', 'cost', 'interv_thres',
                        'infer_thres', 'score', 'num_feedback'
                    ])

  data_dir = os.path.join(os.path.dirname(__file__), "data/")
  df.to_csv(data_dir + "intervention_result8.csv", index=False)
