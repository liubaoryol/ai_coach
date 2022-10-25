import os
import numpy as np
import pickle
from tqdm import tqdm
from ai_coach_core.intervention.feedback_strategy import (
    InterventionValueBased, E_ComboSelection, E_CertaintyHandling)
import intervention_simulator
import pandas as pd

# class BoxPushSimulatorV2_Intervention(BoxPushSimulatorV2):

#   def __init__(self, list_np_policy, list_np_tx, intervention) -> None:
#     super().__init__(None)
#     self.list_np_policy = list_np_policy
#     self.list_np_tx = list_np_tx
#     self.list_prev_np_x_dist = None
#     self.intervention = intervention  # type: InterventionValueBased

#   def reset_game(self):
#     super().reset_game()
#     self.num_feedback = 0

#   def get_prev_state(self, agent_idx):
#     _, bstt, a1pos, a2pos, _, _, _, _ = self.history[-1]
#     return bstt, a1pos, a2pos

#   def get_prev_actions(self):
#     _, _, _, _, a1act, a2act, _, _ = self.history[-1]
#     return a1act, a2act

#   def take_a_step(self, map_agent_2_action) -> None:
#     super().take_a_step(map_agent_2_action)

#     # inference
#     if not isinstance(self.agent_1, BoxPushAIAgent_PartialObs):
#       raise RuntimeError("Invalid agent class")

#     if not isinstance(self.agent_2, BoxPushAIAgent_PartialObs):
#       raise RuntimeError("Invalid agent class")

#     task_mdp = self.agent_1.agent_model.get_reference_mdp()

#     a1act, a2act = self.get_prev_actions()

#     sidx = task_mdp.conv_sim_states_to_mdp_sidx(self.get_prev_state(0))
#     aidx1 = AGENT_ACTIONSPACE.action_to_idx[a1act]
#     aidx2 = AGENT_ACTIONSPACE.action_to_idx[a2act]

#     if self.intervention is None:
#       return

#     list_state = []
#     list_state.append(sidx)
#     list_action = []
#     list_action.append((aidx1, aidx2))

#     sidx_n = task_mdp.conv_sim_states_to_mdp_sidx(
#         tuple(self.get_state_for_each_agent(0)))
#     list_state.append(sidx_n)

#     num_latents = self.agent_1.agent_model.policy_model.get_num_latent_states()

#     def policy_nxsa(nidx, xidx, sidx, tuple_aidx):
#       return self.list_np_policy[nidx][xidx, sidx, tuple_aidx[nidx]]

#     def Tx_nxsasx(nidx, xidx, sidx, tuple_aidx, sidx_n, xidx_n):
#       # return self.list_np_tx[nidx][xidx, tuple_aidx[0], tuple_aidx[1], sidx_n,
#       #                              xidx_n]
#       np_dist = self.list_np_tx[nidx][xidx, tuple_aidx[0], tuple_aidx[1],
#                                       sidx_n]

#       # for illegal states or states that haven't appeared during the training,
#       # we assume mental model was maintained.
#       # if np.all(np_dist == np_dist[0]):
#       #   np_dist = np.zeros_like(np_dist)
#       #   np_dist[xidx] = 1

#       return np_dist[xidx_n]

#     def init_latent_nxs(nidx, xidx, sidx):
#       if nidx == 0:
#         return self.agent_1.agent_model.initial_mental_distribution(sidx)[xidx]
#       else:
#         return self.agent_2.agent_model.initial_mental_distribution(sidx)[xidx]

#     _, list_np_x_dist = forward_inference(list_state, list_action,
#                                           self.get_num_agents(), num_latents,
#                                           policy_nxsa, Tx_nxsasx,
#                                           init_latent_nxs,
#                                           self.list_prev_np_x_dist)
#     self.list_prev_np_x_dist = list_np_x_dist

#     # intervention
#     feedback = self.intervention.get_intervention(self.list_prev_np_x_dist,
#                                                   sidx_n)
#     if feedback is None:
#       return

#     self.num_feedback += 1

#     if 0 in feedback:
#       lat1 = feedback[0]
#       self.agent_1.set_latent(
#           self.agent_1.agent_model.policy_model.conv_idx_to_latent(lat1))
#       np_int_x_dist = np.zeros(len(self.list_prev_np_x_dist[0]))
#       np_int_x_dist[lat1] = 1.0
#       self.list_prev_np_x_dist[0] = np_int_x_dist

#     if 1 in feedback:
#       lat2 = feedback[1]
#       self.agent_2.set_latent(
#           self.agent_2.agent_model.policy_model.conv_idx_to_latent(lat2))
#       np_int_x_dist = np.zeros(len(self.list_prev_np_x_dist[1]))
#       np_int_x_dist[lat2] = 1.0
#       self.list_prev_np_x_dist[1] = np_int_x_dist


def intervention_result(domain_name,
                        num_runs,
                        no_intervention,
                        selection_type: str,
                        certainty_type: str,
                        infer_thres,
                        interv_thres,
                        cost,
                        fix_illegal=True):
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
  else:
    iteration = 500

  v_value_file_name = domain_name + f"_{num_train}_{sup_txt}_{iteration}_merged_v_values_learned.pickle"
  policy1_file = domain_name + f"_btil2_policy_synth_woTx_FTTT_{num_train}_{sup_txt}_a1.npy"
  policy2_file = domain_name + f"_btil2_policy_synth_woTx_FTTT_{num_train}_{sup_txt}_a2.npy"
  tx1_file = domain_name + f"_btil2_tx_synth_FTTT_{num_train}_{sup_txt}_a1.npy"
  tx2_file = domain_name + f"_btil2_tx_synth_FTTT_{num_train}_{sup_txt}_a2.npy"

  if domain_name == "movers":
    from ai_coach_domain.box_push_v2.agent import BoxPushAIAgent_PO_Team
    from ai_coach_domain.box_push_v2.maps import MAP_MOVERS
    from ai_coach_domain.box_push_v2.policy import Policy_Movers
    from ai_coach_domain.box_push_v2.mdp import MDP_Movers_Agent
    from ai_coach_domain.box_push_v2.mdp import MDP_Movers_Task
    from ai_coach_domain.box_push_v2.simulator import BoxPushSimulatorV2
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
    game = BoxPushSimulatorV2(0)

    def get_state_action(history):
      step, bstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat = history
      return (bstt, a1pos, a2pos), (a1act, a2act)

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

    game = RescueSimulator()
    game.max_steps = 30

    def get_state_action(history):
      step, score, wstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat = history
      return (wstt, a1pos, a2pos), (a1act, a2act)

  with open(data_dir + v_value_file_name, 'rb') as handle:
    np_v_values = pickle.load(handle)

  np_policy1 = np.load(model_dir + policy1_file)
  np_tx1 = np.load(model_dir + tx1_file)
  np_policy2 = np.load(model_dir + policy2_file)
  np_tx2 = np.load(model_dir + tx2_file)
  if no_intervention:
    intervention_strategy = None
  if selection_type == "Value":
    intervention_strategy = InterventionValueBased(np_v_values,
                                                   e_certainty,
                                                   inference_threshold=theta,
                                                   intervention_threshold=delta,
                                                   intervention_cost=cost)
  elif selection_type == "Rule":
    intervention_strategy = InterventionValueBased(np_v_values,
                                                   e_certainty,
                                                   inference_threshold=theta,
                                                   intervention_threshold=delta,
                                                   intervention_cost=cost)
  else:
    raise NotImplementedError

  game.init_game(**game_map)
  game.set_autonomous_agent(agent1, agent2)
  sim = intervention_simulator.InterventionSimulator(
      game, [np_policy1, np_policy2], [np_tx1, np_tx2], intervention_strategy,
      get_state_action, fix_illegal)

  list_score, list_num_feedback = sim.run_game(num_runs)
  return list(zip(list_score, list_num_feedback))


if __name__ == "__main__":

  domain_name = "cleanup_v2"
  num_runs = 100

  cost = 0

  VALUE = "Value"
  AVERAGE = "Average"  # Average | Threshold
  THRESHOLD = "Threshold"
  NO_INTERVENTION = True
  INTERVENTION = False

  DO_TEST = True

  if DO_TEST:
    list_res = intervention_result(domain_name, num_runs, INTERVENTION, VALUE,
                                   AVERAGE, None, 0.1, cost)
    print(np.array(list_res).mean(axis=0))

    raise RuntimeError

  rows = []

  domains = ["movers", "cleanup_v2", "rescue_2"]
  dict_interv_thres = {
      "movers": [0, 1, 3, 5, 10, 15, 20, 30, 50],
      "cleanup_v2": [0, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0],
      "rescue_2": [0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
  }

  list_infer_thres = [0, 0.2, 0.3, 0.5, 0.7, 0.9]
  for domain_name in domains:
    print(domain_name)
    list_interv_thres = dict_interv_thres[domain_name]
    num_total_setup = 1 + 2 * len(list_interv_thres) + len(list_infer_thres)
    progress_bar = tqdm(total=num_total_setup)
    # ===== Baseline: No intervention
    list_res = intervention_result(domain_name, num_runs, NO_INTERVENTION, None,
                                   None, None, None, cost)

    rows = rows + [(domain_name, "No_intervention", cost, 0, 0, *item)
                   for item in list_res]
    progress_bar.update()

    # ===== Strategy 1: Stochastic
    for delta in list_interv_thres:
      list_res = intervention_result(domain_name, num_runs, INTERVENTION, VALUE,
                                     AVERAGE, None, delta, cost)

      rows = rows + [(domain_name, "Average", cost, delta, 0, *item)
                     for item in list_res]
      progress_bar.update()

    # ===== Strategy 2: Deterministic
    for delta in list_interv_thres:
      list_res = intervention_result(domain_name, num_runs, INTERVENTION, VALUE,
                                     THRESHOLD, 0, delta, cost)

      rows = rows + [(domain_name, "Argmax", cost, delta, 0, *item)
                     for item in list_res]
      progress_bar.update()

    delta_s3 = list_interv_thres[3]
    # ===== Strategy 3: Deterministic with threshold
    for theta in list_infer_thres:
      list_res = intervention_result(domain_name, num_runs, INTERVENTION, VALUE,
                                     THRESHOLD, theta, delta_s3, cost)

      rows = rows + [(domain_name, "Argmax_thres", cost, delta_s3, theta, *item)
                     for item in list_res]
      progress_bar.update()
    progress_bar.close()

  df = pd.DataFrame(rows,
                    columns=[
                        'domain', 'strategy', 'cost', 'interv_thres',
                        'infer_thres', 'score', 'num_feedback'
                    ])

  data_dir = os.path.join(os.path.dirname(__file__), "data/")
  df.to_csv(data_dir + "intervention_result.csv", index=False)
