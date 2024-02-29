from typing import Sequence
import os
import numpy as np
import pickle
import click
from tqdm import tqdm
from aic_core.intervention.feedback_strategy import (
    InterventionValueBased, InterventionRuleBased,
    PartialInterventionValueBased, E_CertaintyHandling)
import aic_domain.intervention_simulator as intervention_simulator
import pandas as pd
from aic_domain.agent.agent import AIAgent_Abstract
from aic_domain.box_push.agent_model import get_holding_box_and_floor_boxes
import load_domain

VALUE = "Value"
VALUE_FIXROBOT = "ValueFixedRobot"
RULE = "Rule"
AVERAGE = "Average"  # Average | Threshold
THRESHOLD = "Threshold"

NO_INTERVENTION = True
INTERVENTION = False


def movers_compatible_latents_rule(obs_idx, agents: Sequence[AIAgent_Abstract]):
  agent1_model = agents[0].agent_model
  mdp_task = agent1_model.get_reference_mdp()
  box_states, _, _ = (mdp_task.conv_mdp_sidx_to_sim_states(obs_idx))
  num_drops = len(mdp_task.drops)
  num_goals = len(mdp_task.goals)
  a1_box, a2_box, valid_box = get_holding_box_and_floor_boxes(
      box_states, num_drops, num_goals)
  list_valid_lat = []
  if a1_box > -1 and a1_box == a2_box:
    lat = agent1_model.policy_model.conv_latent_to_idx(("goal", 0))
    list_valid_lat.append(lat)
  else:
    for idx in valid_box:
      lat = agent1_model.policy_model.conv_latent_to_idx(("pickup", idx))
      list_valid_lat.append(lat)

  list_valid_combo_pairs = []
  for lat in list_valid_lat:
    list_valid_combo_pairs.append(tuple([lat] * 2))

  return list_valid_combo_pairs


def intervention_result(domain_name,
                        num_runs,
                        no_intervention,
                        selection_type: str,
                        certainty_type: str,
                        infer_thres,
                        interv_thres,
                        cost,
                        data_dir,
                        policy1_file,
                        policy2_file,
                        policy3_file,
                        tx1_file,
                        tx2_file,
                        tx3_file,
                        bx1_file,
                        bx2_file,
                        bx3_file,
                        v_value_file,
                        fix_illegal=True,
                        increase_step=False,
                        is_btil_agent=False):
  model_dir = os.path.join(data_dir, "learned_models/")

  e_certainty = E_CertaintyHandling[
      certainty_type] if certainty_type is not None else None
  theta = infer_thres
  delta = interv_thres

  np_policy1 = np.load(model_dir + policy1_file)
  np_policy2 = np.load(model_dir + policy2_file)

  np_tx1 = np.load(model_dir + tx1_file)
  np_tx2 = np.load(model_dir + tx2_file)

  np_bx1 = np.load(model_dir + bx1_file) if bx1_file else None
  np_bx2 = np.load(model_dir + bx2_file) if bx2_file else None

  list_np_policy = [np_policy1, np_policy2]
  list_np_tx = [np_tx1, np_tx2]

  dict_btil_args = {}
  dict_btil_args['np_policy1'] = np_policy1
  dict_btil_args['np_policy2'] = np_policy2
  dict_btil_args['np_tx1'] = np_tx1
  dict_btil_args['np_tx2'] = np_tx2
  dict_btil_args['np_bx1'] = np_bx1
  dict_btil_args['np_bx2'] = np_bx2
  dict_btil_args['mask'] = (False, True, True, True)

  fixed_agents = [1]
  fn_valid_latent = None
  if domain_name == "movers":
    vec_domain_data = load_domain.load_movers(is_btil_agent, dict_btil_args)
    fn_valid_latent = lambda obs: movers_compatible_latents_rule(obs, agents)
  elif domain_name == "cleanup_v2":
    vec_domain_data = load_domain.load_cleanup_v2(is_btil_agent, dict_btil_args)
  elif domain_name == "cleanup_v3":
    vec_domain_data = load_domain.load_cleanup_v3(is_btil_agent, dict_btil_args)
  elif domain_name == "rescue_2":
    vec_domain_data = load_domain.load_rescue_2(is_btil_agent, dict_btil_args)
  elif domain_name == "rescue_3":

    np_policy3 = np.load(model_dir + policy3_file)
    np_tx3 = np.load(model_dir + tx3_file)
    np_bx3 = np.load(model_dir + bx3_file) if bx3_file else None
    dict_btil_args['np_policy3'] = np_policy3
    dict_btil_args['np_tx3'] = np_tx3
    dict_btil_args['np_bx3'] = np_bx3
    dict_btil_args['mask'] = (False, True, True, True, True)

    list_np_policy.append(np_policy3)
    list_np_tx.append(np_tx3)
    fixed_agents.append(2)

    vec_domain_data = load_domain.load_rescue_3(is_btil_agent, dict_btil_args)
  else:
    raise NotImplementedError(domain_name)

  game, agents, _, _, game_map = vec_domain_data

  with open(os.path.join(data_dir, v_value_file), 'rb') as handle:
    np_v_values = pickle.load(handle)

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
          fixed_agents=fixed_agents)
    else:
      raise NotImplementedError

  game.init_game(**game_map)
  game.set_autonomous_agent(*agents)
  sim = intervention_simulator.InterventionSimulator(
      game,
      list_np_policy,
      list_np_tx,
      intervention_strategy,
      fix_illegal,
      increase_step=increase_step)

  list_score, list_num_feedback = sim.run_game(num_runs)
  return list(zip(list_score, list_num_feedback))


# yapf: disable
@click.command()
@click.option("--domain", type=str, default="movers",
              help="movers|cleanup_v3|rescue_2|rescue_3")
@click.option("--costs", type=str, default="0;1",
              help="use semicolon(;) with no space to specify multiple costs")
@click.option("--num-runs", type=int, default=100, help="")
@click.option("--dir-name", type=str, default="data", help="data|human_data")
@click.option("--output-name", type=str, default="", help="")
@click.option("--policy1-file", type=str, default="", help="")
@click.option("--policy2-file", type=str, default="", help="")
@click.option("--policy3-file", type=str, default="", help="")
@click.option("--tx1-file", type=str, default="", help="")
@click.option("--tx2-file", type=str, default="", help="")
@click.option("--tx3-file", type=str, default="", help="")
@click.option("--bx1-file", type=str, default="", help="")
@click.option("--bx2-file", type=str, default="", help="")
@click.option("--bx3-file", type=str, default="", help="")
@click.option("--v-value-file", type=str, default="", help="")
@click.option("--is-btil-agent", type=bool, default=False, help="")
# yapf: enable
def main(domain: str, costs: str, num_runs, dir_name, output_name, policy1_file,
         policy2_file, policy3_file, tx1_file, tx2_file, tx3_file, bx1_file,
         bx2_file, bx3_file, v_value_file, is_btil_agent):
  dname = domain
  list_cost = [int(item) for item in costs.split(';')]

  data_dir = os.path.join(os.path.dirname(__file__), dir_name)

  LIST_INFER_THRES = [0, 0.2, 0.3, 0.5, 0.7, 0.9]
  DICT_INTERV_THRES = {
      "movers": [0, 1, 3, 5, 10, 15, 20, 30, 50],
      "cleanup_v3": [0, 0.3, 0.5, 1.0, 2.0, 5.0, 10, 15, 20],
      "rescue_2": [0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
      "rescue_3": [0, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0],
  }

  INTERV_THRES_INDICES_FOR_CONFIDENCE_METHOD = [0, 1, 2, 3, 4]

  list_config = []
  for cost in list_cost:
    if dname == "movers":
      # Expectation-Rule
      config = (dname, INTERVENTION, RULE, AVERAGE, 0, 0, cost, "Rule_avg")
      list_config.append(config)

    # No intervention
    config = (dname, NO_INTERVENTION, None, None, 0, 0, cost, "No_intervention")
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

        config = (dname, INTERVENTION, VALUE_FIXROBOT, THRESHOLD, theta, delta,
                  cost, "Argmax_thres_robot_fix")
        list_config.append(config)

    for delta in DICT_INTERV_THRES[dname]:
      # Expectation-Value
      config = (dname, INTERVENTION, VALUE, AVERAGE, 0, delta, cost, "Average")
      list_config.append(config)
      # Deterministic-Value
      config = (dname, INTERVENTION, VALUE, THRESHOLD, 0, delta, cost, "Argmax")
      list_config.append(config)

      # Deterministic-Value
      config = (dname, INTERVENTION, VALUE_FIXROBOT, THRESHOLD, 0, delta, cost,
                "Argmax_robot_fix")
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
                                   data_dir,
                                   policy1_file,
                                   policy2_file,
                                   policy3_file,
                                   tx1_file,
                                   tx2_file,
                                   tx3_file,
                                   bx1_file,
                                   bx2_file,
                                   bx3_file,
                                   v_value_file,
                                   increase_step=False,
                                   is_btil_agent=is_btil_agent)
    rows = rows + [(dname, method_name, cost, delta, theta, *item)
                   for item in list_res]

  df = pd.DataFrame(rows,
                    columns=[
                        'domain', 'strategy', 'cost', 'interv_thres',
                        'infer_thres', 'score', 'num_feedback'
                    ])
  save_dir = os.path.join(data_dir, "intervention_results")
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  output_name = os.path.splitext(output_name)[0]

  df.to_csv(os.path.join(save_dir, output_name + f"-{dname}.csv"), index=False)


if __name__ == "__main__":

  DO_TEST = False

  if DO_TEST:
    domain_name = "movers"
    list_res = intervention_result(domain_name, 100, INTERVENTION, VALUE,
                                   AVERAGE, 0, 3, 0)
    print(np.array(list_res).mean(axis=0))

    raise RuntimeError

  main()
