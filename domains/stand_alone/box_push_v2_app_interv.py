import numpy as np
from ai_coach_domain.box_push_v2.simulator import BoxPushSimulatorV2
from ai_coach_domain.box_push_v2.maps import MAP_MOVERS, MAP_CLEANUP_V2
from ai_coach_domain.box_push_v2.mdp import (MDP_Movers_Task, MDP_Movers_Agent,
                                             MDP_Cleanup_Task,
                                             MDP_Cleanup_Agent)
from ai_coach_domain.box_push_v2.policy import Policy_Movers, Policy_Cleanup
from ai_coach_domain.box_push_v2.agent import (BoxPushAIAgent_PO_Team,
                                               BoxPushAIAgent_PO_Indv,
                                               BoxPushAIAgent_BTIL,
                                               BoxPushAIAgent_Team,
                                               BoxPushAIAgent_Indv)
from ai_coach_domain.agent import BTILCachedPolicy
from stand_alone.box_push_app import BoxPushApp
import pickle
from ai_coach_core.intervention.feedback_strategy import (
    get_combos_sorted_by_simulated_values)
from stand_alone.intervention_simulator import InterventionSimulator
from ai_coach_core.intervention.feedback_strategy import (
    InterventionValueBased, E_CertaintyHandling)

TEST_BTIL_AGENT = False
TEST_BTIL_USE_TRUE_TX = False
IS_MOVERS = False
DATA_DIR = "misc/BTIL_feedback_results/data/"
if IS_MOVERS:
  GAME_MAP = MAP_MOVERS
  POLICY = Policy_Movers
  MDP_TASK = MDP_Movers_Task
  MDP_AGENT = MDP_Movers_Agent
  AGENT = BoxPushAIAgent_PO_Team
  TEST_AGENT = BoxPushAIAgent_Team
  V_VAL_FILE_NAME = "movers_500_0,30_500_merged_v_values_learned.pickle"
  NP_POLICY_A1 = "movers_btil2_policy_synth_woTx_FTTT_500_0,30_a1.npy"
  NP_POLICY_A2 = "movers_btil2_policy_synth_woTx_FTTT_500_0,30_a2.npy"
  NP_TX_A1 = "movers_btil2_tx_synth_FTTT_500_0,30_a1.npy"
  NP_TX_A2 = "movers_btil2_tx_synth_FTTT_500_0,30_a2.npy"
else:
  GAME_MAP = MAP_CLEANUP_V2
  POLICY = Policy_Cleanup
  MDP_TASK = MDP_Cleanup_Task
  MDP_AGENT = MDP_Cleanup_Agent
  AGENT = BoxPushAIAgent_PO_Indv
  TEST_AGENT = BoxPushAIAgent_Indv
  V_VAL_FILE_NAME = "cleanup_v2_500_0,30_500_merged_v_values_learned.pickle"
  NP_POLICY_A1 = "cleanup_v2_btil2_policy_synth_woTx_FTTT_500_0,30_a1.npy"
  NP_POLICY_A2 = "cleanup_v2_btil2_policy_synth_woTx_FTTT_500_0,30_a2.npy"
  NP_TX_A1 = "cleanup_v2_btil2_tx_synth_FTTT_500_0,30_a1.npy"
  NP_TX_A2 = "cleanup_v2_btil2_tx_synth_FTTT_500_0,30_a2.npy"


class BoxPushV2App(BoxPushApp):

  def __init__(self) -> None:
    super().__init__()

  def _init_game(self):
    'define game related variables and objects'
    # game_map["a2_init"] = (1, 2)
    self.x_grid = GAME_MAP["x_grid"]
    self.y_grid = GAME_MAP["y_grid"]
    self.game = BoxPushSimulatorV2(None)
    self.game.max_steps = 100

    self.game.init_game(**GAME_MAP)

    mdp_task = MDP_TASK(**GAME_MAP)
    mdp_agent = MDP_AGENT(**GAME_MAP)
    self.mdp = mdp_task

    if V_VAL_FILE_NAME is not None:
      with open(DATA_DIR + V_VAL_FILE_NAME, 'rb') as handle:
        self.np_v_values = pickle.load(handle)

    if not TEST_BTIL_AGENT:
      TEMPERATURE = 0.3
      init_states = ([0] * len(GAME_MAP["boxes"]), GAME_MAP["a1_init"],
                     GAME_MAP["a2_init"])

      # agent1 = InteractiveAgent()
      policy1 = POLICY(mdp_task, mdp_agent, TEMPERATURE, agent_idx=0)
      agent1 = AGENT(init_states, policy1, agent_idx=0)

      policy2 = POLICY(mdp_task, mdp_agent, TEMPERATURE, agent_idx=1)
      agent2 = AGENT(init_states, policy2, agent_idx=1)
    else:
      model_dir = DATA_DIR + "/learned_models/"  # noqa: E501
      np_policy_1 = np.load(model_dir + NP_POLICY_A1)
      test_policy_1 = BTILCachedPolicy(np_policy_1, mdp_task, 0,
                                       mdp_agent.latent_space)
      np_policy_2 = np.load(model_dir + NP_POLICY_A2)
      test_policy_2 = BTILCachedPolicy(np_policy_2, mdp_task, 1,
                                       mdp_agent.latent_space)

      if TEST_BTIL_USE_TRUE_TX:
        agent1 = TEST_AGENT(test_policy_1, agent_idx=0)
        agent2 = TEST_AGENT(test_policy_2, agent_idx=1)
      else:
        np_tx_1 = np.load(model_dir + NP_TX_A1)
        np_tx_2 = np.load(model_dir + NP_TX_A2)
        mask = (False, True, True, True)
        agent1 = BoxPushAIAgent_BTIL(np_tx_1, mask, test_policy_1, 0)
        agent2 = BoxPushAIAgent_BTIL(np_tx_2, mask, test_policy_2, 1)

    model_dir = DATA_DIR + "/learned_models/"  # noqa: E501
    np_policy_1 = np.load(model_dir + NP_POLICY_A1)
    np_policy_2 = np.load(model_dir + NP_POLICY_A2)
    np_tx_1 = np.load(model_dir + NP_TX_A1)
    np_tx_2 = np.load(model_dir + NP_TX_A2)

    intervention_strategy = InterventionValueBased(
        self.np_v_values,
        E_CertaintyHandling.Threshold,
        inference_threshold=0,
        intervention_threshold=0.02,
        intervention_cost=0)

    def get_state_action(history):
      step, bstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat = history
      return (bstt, a1pos, a2pos), (a1act, a2act)

    self.game.set_autonomous_agent(agent1, agent2)
    self.interv_sim = InterventionSimulator(self.game,
                                            [np_policy_1, np_policy_2],
                                            [np_tx_1, np_tx_2],
                                            intervention_strategy,
                                            get_state_action,
                                            fix_illegal=True)
    self.interv_sim.reset_game()

  def _update_canvas_scene(self):
    super()._update_canvas_scene()

    x_unit = int(self.canvas_width / self.x_grid)
    y_unit = int(self.canvas_height / self.y_grid)

    self.create_text((self.game.a1_pos[0] + 0.5) * x_unit,
                     (self.game.a1_pos[1] + 0.4) * y_unit,
                     str(self.game.agent_1.get_current_latent()))
    self.create_text((self.game.a2_pos[0] + 0.5) * x_unit,
                     (self.game.a2_pos[1] + 0.6) * y_unit,
                     str(self.game.agent_2.get_current_latent()))
    if V_VAL_FILE_NAME is not None:
      game = self.game  # type: BoxPushSimulatorV2
      tup_state = tuple(game.get_state_for_each_agent(0))
      oidx = self.mdp.conv_sim_states_to_mdp_sidx(tup_state)
      list_combos = get_combos_sorted_by_simulated_values(
          self.np_v_values, oidx)
      print(list_combos[:6])
      print(game.get_score())

  def _on_key_pressed(self, key_event):
    if not self._started:
      return

    agent, e_type, e_value = self._conv_key_to_agent_event(key_event.keysym)
    self.game.event_input(agent, e_type, e_value)
    if self._event_based:
      action_map = self.game.get_joint_action()
      self.game.take_a_step(action_map)
      print("====")
      print(action_map)
      self.interv_sim.intervene()

      if not self.game.is_finished():
        # update canvas
        self._update_canvas_scene()
        self._update_canvas_overlay()
        # pop-up for latent?
      else:
        self._on_game_end()
    else:
      pass


if __name__ == "__main__":
  app = BoxPushV2App()
  app.run()
