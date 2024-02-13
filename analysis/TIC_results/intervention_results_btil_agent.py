import os
import numpy as np
import pickle
from tqdm import tqdm
from aic_core.intervention.feedback_strategy import (
    InterventionValueBased, InterventionRuleBased,
    PartialInterventionValueBased, E_CertaintyHandling)
import aic_domain.intervention_simulator as intervention_simulator
import pandas as pd

VALUE = "Value"
VALUE_FIXROBOT = "ValueFixedRobot"
RULE = "Rule"
AVERAGE = "Average"  # Average | Threshold
THRESHOLD = "Threshold"


def intervention_result(domain_name,
                        num_runs,
                        no_intervention,
                        selection_type: str,
                        certainty_type: str,
                        infer_thres,
                        interv_thres,
                        cost,
                        fix_illegal=True,
                        increase_step=False,
                        humandata=True):
  cur_dir = os.path.dirname(__file__)
  if humandata:
    data_dir = os.path.join(cur_dir, "human_data/")
  else:
    data_dir = os.path.join(cur_dir, "data/")

  model_dir = os.path.join(data_dir, "learned_models/")

  e_certainty = E_CertaintyHandling[
      certainty_type] if certainty_type is not None else None
  # theta = 0.5
  # delta = 5
  # cost = 0
  theta = infer_thres
  delta = interv_thres

  num_train = 160 if humandata else 500
  supervision = 0.3
  sup_txt = ("%.2f" % supervision).replace('.', ',')

  if domain_name == "rescue_2":
    iteration = 30
  elif domain_name == "rescue_3":
    iteration = 15
  else:
    iteration = 150

  v_value_file_name = (
      domain_name +
      f"_{num_train}_{sup_txt}_{iteration}_merged_v_values_learned.pickle")
  if domain_name == "rescue_3":
    tx_dependency = "FTTTT"
  else:
    tx_dependency = "FTTT"

  datasource = "human" if humandata else "synth"
  algname = "btil_dec"
  suffix = f"{tx_dependency}_{num_train}_{sup_txt}"

  policy1_file = (domain_name +
                  f"_{algname}_policy_{datasource}_woTx_{suffix}_a1.npy")
  policy2_file = (domain_name +
                  f"_{algname}_policy_{datasource}_woTx_{suffix}_a2.npy")
  policy3_file = (domain_name +
                  f"_{algname}_policy_{datasource}_woTx_{suffix}_a3.npy")

  tx1_file = (domain_name + f"_{algname}_tx_{datasource}_{suffix}_a1.npy")
  tx2_file = (domain_name + f"_{algname}_tx_{datasource}_{suffix}_a2.npy")
  tx3_file = (domain_name + f"_{algname}_tx_{datasource}_{suffix}_a3.npy")

  if domain_name == "movers":
    from aic_domain.box_push_v2.agent import BoxPushAIAgent_BTIL
    from aic_domain.agent import BTILCachedPolicy
    from aic_domain.box_push_v2.maps import MAP_MOVERS
    from aic_domain.box_push_v2.mdp import MDP_Movers_Agent
    from aic_domain.box_push_v2.mdp import MDP_Movers_Task
    from aic_domain.box_push_v2.simulator import BoxPushSimulatorV2
    from aic_domain.box_push.agent_model import (get_holding_box_and_floor_boxes
                                                 )
    game_map = MAP_MOVERS
    MDP_Task = MDP_Movers_Task(**game_map)
    MDP_Agent = MDP_Movers_Agent(**game_map)

    np_policy1 = np.load(model_dir + policy1_file)
    np_tx1 = np.load(model_dir + tx1_file)
    np_policy2 = np.load(model_dir + policy2_file)
    np_tx2 = np.load(model_dir + tx2_file)
    mask = (False, True, True, True)

    policy1 = BTILCachedPolicy(np_policy1, MDP_Task, 0, MDP_Agent.latent_space)
    policy2 = BTILCachedPolicy(np_policy2, MDP_Task, 1, MDP_Agent.latent_space)

    agent1 = BoxPushAIAgent_BTIL(np_tx1, mask, policy1, 0)
    agent2 = BoxPushAIAgent_BTIL(np_tx2, mask, policy2, 1)
    agents = [agent1, agent2]
    game = BoxPushSimulatorV2(0)

    def get_state_action(history):
      step, bstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat = history
      return (bstt, a1pos, a2pos), (a1act, a2act)

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

      list_valid_combo_pairs = []
      for lat in list_valid_lat:
        list_valid_combo_pairs.append(tuple([lat] * 2))

      return list_valid_combo_pairs

    fn_valid_latent = valid_latent2
  elif domain_name == "rescue_2":
    from aic_domain.rescue.agent import AIAgent_Rescue_BTIL
    from aic_domain.agent import BTILCachedPolicy
    from aic_domain.rescue.maps import MAP_RESCUE
    from aic_domain.rescue.mdp import MDP_Rescue_Agent, MDP_Rescue_Task
    from aic_domain.rescue.simulator import RescueSimulator
    game_map = MAP_RESCUE
    MDP_Task = MDP_Rescue_Task(**game_map)
    MDP_Agent = MDP_Rescue_Agent(**game_map)

    np_policy1 = np.load(model_dir + policy1_file)
    np_tx1 = np.load(model_dir + tx1_file)
    np_policy2 = np.load(model_dir + policy2_file)
    np_tx2 = np.load(model_dir + tx2_file)
    mask = (False, True, True, True)

    policy1 = BTILCachedPolicy(np_policy1, MDP_Task, 0, MDP_Agent.latent_space)
    policy2 = BTILCachedPolicy(np_policy2, MDP_Task, 1, MDP_Agent.latent_space)

    agent1 = AIAgent_Rescue_BTIL(np_tx1, mask, policy1, 0)
    agent2 = AIAgent_Rescue_BTIL(np_tx2, mask, policy2, 1)
    agents = [agent1, agent2]

    game = RescueSimulator()
    game.max_steps = 30

    def get_state_action(history):
      step, score, wstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat = history
      return (wstt, a1pos, a2pos), (a1act, a2act)

  else:
    raise NotImplementedError(domain_name)

  with open(data_dir + v_value_file_name, 'rb') as handle:
    np_v_values = pickle.load(handle)

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
    if selection_type == VALUE:
      intervention_strategy = InterventionValueBased(
          np_v_values,
          e_certainty,
          inference_threshold=theta,
          intervention_threshold=delta,
          intervention_cost=cost)
    elif selection_type == RULE:
      intervention_strategy = InterventionRuleBased(
          fn_valid_latent,
          len(agents),
          e_certainty,
          inference_threshold=theta,
          intervention_threshold=delta,
          intervention_cost=cost)
    elif selection_type == VALUE_FIXROBOT:
      intervention_strategy = PartialInterventionValueBased(
          np_v_values,
          inference_threshold=theta,
          intervention_threshold=delta,
          intervention_cost=cost,
          fixed_agents=[game.AGENT2])
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

  NO_INTERVENTION = True
  INTERVENTION = False

  DO_TEST = False

  if DO_TEST:
    domain_name = "movers"
    list_res = intervention_result(domain_name, num_runs, INTERVENTION, VALUE,
                                   AVERAGE, 0, 3, 0)
    print(np.array(list_res).mean(axis=0))

    raise RuntimeError

  list_cost = [1]
  # cost = list_cost[0]
  domains = ["movers", "rescue_2"]
  LIST_INFER_THRES = [0, 0.2, 0.3, 0.5, 0.7, 0.9]
  DICT_INTERV_THRES = {
      "movers": [0, 1, 3, 5, 10, 15, 20, 30, 50],
      "cleanup_v3": [0, 0.3, 0.5, 1.0, 2.0, 5.0, 10, 15, 20],
      "rescue_2": [0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
      "rescue_3": [0, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0],
  }

  INTERV_THRES_INDICES_FOR_CONFIDENCE_METHOD = [0, 1, 2, 3]

  list_config = []
  for cost in list_cost:
    for dname in domains:
      if dname == "movers":
        # Expectation-Rule
        config = (dname, INTERVENTION, RULE, AVERAGE, 0, 0, cost, "Rule_avg")
        list_config.append(config)

      # No intervention
      config = (dname, NO_INTERVENTION, None, None, 0, 0, cost,
                "No_intervention")
      list_config.append(config)

      for theta in LIST_INFER_THRES:
        if dname == "movers":
          # Confidence-Rule
          config = (dname, INTERVENTION, RULE, THRESHOLD, theta, 0, cost,
                    "Rule_thres")
          list_config.append(config)

        for idx in INTERV_THRES_INDICES_FOR_CONFIDENCE_METHOD:
          delta = DICT_INTERV_THRES[dname][idx]
          # Confidence-Value
          config = (dname, INTERVENTION, VALUE, THRESHOLD, theta, delta, cost,
                    "Argmax_thres")
          list_config.append(config)

          config = (dname, INTERVENTION, VALUE_FIXROBOT, THRESHOLD, theta,
                    delta, cost, "Argmax_thres_robot_fix")
          list_config.append(config)

      for delta in DICT_INTERV_THRES[dname]:
        # Expectation-Value
        config = (dname, INTERVENTION, VALUE, AVERAGE, 0, delta, cost,
                  "Average")
        list_config.append(config)
        # Deterministic-Value
        config = (dname, INTERVENTION, VALUE, THRESHOLD, 0, delta, cost,
                  "Argmax")
        list_config.append(config)

        # Deterministic-Value
        config = (dname, INTERVENTION, VALUE_FIXROBOT, THRESHOLD, 0, delta,
                  cost, "Argmax_robot_fix")
        list_config.append(config)

  rows = []
  for config in tqdm(list_config):
    dname, no_int, sel_type, cer_type, theta, delta, cost, method_name = config
    list_res = intervention_result(dname,
                                   num_runs,
                                   no_int,
                                   sel_type,
                                   cer_type,
                                   theta,
                                   delta,
                                   cost,
                                   increase_step=False)
    rows = rows + [(dname, method_name, cost, delta, theta, *item)
                   for item in list_res]

  df = pd.DataFrame(rows,
                    columns=[
                        'domain', 'strategy', 'cost', 'interv_thres',
                        'infer_thres', 'score', 'num_feedback'
                    ])

  data_dir = os.path.join(os.path.dirname(__file__), "human_data/")
  df.to_csv(data_dir + "btil_intervention_result_20240213.csv", index=False)
