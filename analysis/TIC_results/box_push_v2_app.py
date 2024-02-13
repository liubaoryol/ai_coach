import numpy as np
import os
from aic_domain.box_push_v2.simulator import BoxPushSimulatorV2
from aic_domain.box_push_v2.maps import MAP_MOVERS
from aic_domain.box_push_v2.maps import MAP_CLEANUP_V3 as MAP_CLEANUP
from aic_domain.box_push_v2.mdp import (MDP_Movers_Task, MDP_Movers_Agent,
                                        MDP_Cleanup_Task, MDP_Cleanup_Agent)
from aic_domain.box_push_v2.policy import Policy_Movers, Policy_Cleanup
from aic_domain.box_push_v2.agent import (
    BoxPushAIAgent_PO_Team, BoxPushAIAgent_PO_Indv, BoxPushAIAgent_BTIL,
    BoxPushAIAgent_BTIL_ABS, BoxPushAIAgent_Team, BoxPushAIAgent_Indv)
from aic_domain.agent import BTILCachedPolicy
from stand_alone.box_push_app import BoxPushApp
import pickle
from aic_core.intervention.feedback_strategy import (get_sorted_x_combos)
from aic_core.utils.mdp_utils import StateSpace

TEST_BTIL_AGENT = False
TEST_BTIL_USE_TRUE_TX = False
IS_MOVERS = True
DATA_DIR = os.path.dirname(__file__) + "/data/"
V_VAL_FILE_NAME = None
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
  GAME_MAP = MAP_CLEANUP
  POLICY = Policy_Cleanup
  MDP_TASK = MDP_Cleanup_Task
  MDP_AGENT = MDP_Cleanup_Agent
  AGENT = BoxPushAIAgent_PO_Indv
  TEST_AGENT = BoxPushAIAgent_Indv
  # V_VAL_FILE_NAME = "cleanup_v3_500_0,30_500_merged_v_values_learned.pickle"
  NP_POLICY_A1 = "cleanup_v3_btil2_policy_synth_woTx_FTTT_500_0,30_a1.npy"
  NP_POLICY_A2 = "cleanup_v3_btil2_policy_synth_woTx_FTTT_500_0,30_a2.npy"
  NP_TX_A1 = "cleanup_v3_btil2_tx_synth_FTTT_500_0,30_a1.npy"
  NP_TX_A2 = "cleanup_v3_btil2_tx_synth_FTTT_500_0,30_a2.npy"


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
                                       StateSpace(np.arange(5)))
      np_policy_2 = np.load(model_dir + NP_POLICY_A2)
      test_policy_2 = BTILCachedPolicy(np_policy_2, mdp_task, 1,
                                       StateSpace(np.arange(5)))

      if TEST_BTIL_USE_TRUE_TX:
        agent1 = TEST_AGENT(test_policy_1, agent_idx=0)
        agent2 = TEST_AGENT(test_policy_2, agent_idx=1)
      else:
        np_tx_1 = np.load(model_dir + NP_TX_A1)
        np_tx_2 = np.load(model_dir + NP_TX_A2)

        mask = (False, True, True, True)
        agent1 = BoxPushAIAgent_BTIL(np_tx_1, mask, test_policy_1, 0)
        agent2 = BoxPushAIAgent_BTIL(np_tx_2, mask, test_policy_2, 1)

    self.game.set_autonomous_agent(agent1, agent2)

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
      list_combos = get_sorted_x_combos(self.np_v_values, oidx)
      print("=================================================")
      print(list_combos)


if __name__ == "__main__":
  app = BoxPushV2App()
  app.run()
