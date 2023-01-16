import numpy as np
from ai_coach_domain.box_push_v2.simulator import BoxPushSimulatorV2
from ai_coach_domain.box_push_v2.maps import MAP_MOVERS
from ai_coach_domain.box_push_v2.maps import MAP_CLEANUP_V3 as MAP_CLEANUP
from ai_coach_domain.box_push_v2.mdp import (MDP_Movers_Task, MDP_Movers_Agent,
                                             MDP_Cleanup_Task,
                                             MDP_Cleanup_Agent)
from ai_coach_domain.box_push_v2.policy import Policy_Movers, Policy_Cleanup
from ai_coach_domain.box_push_v2.agent import (
    BoxPushAIAgent_PO_Team, BoxPushAIAgent_PO_Indv, BoxPushAIAgent_BTIL,
    BoxPushAIAgent_BTIL_ABS, BoxPushAIAgent_Team, BoxPushAIAgent_Indv,
    AIAgent_NoMind)
from ai_coach_domain.agent.cached_agent import (BTILCachedPolicy,
                                                NoMindCachedPolicy)
from stand_alone.box_push_app import BoxPushApp
import pickle
from ai_coach_core.intervention.feedback_strategy import (
    get_combos_sorted_by_simulated_values)
from ai_coach_core.utils.mdp_utils import StateSpace

TEST_LEARNED_AGENT = True
IS_MOVERS = True
DATA_DIR = "misc/BTIL_abstract_results/data"
if IS_MOVERS:
  GAME_MAP = MAP_MOVERS
  POLICY = Policy_Movers
  MDP_TASK = MDP_Movers_Task
  MDP_AGENT = MDP_Movers_Agent
  AGENT = BoxPushAIAgent_Team
  TEST_AGENT = BoxPushAIAgent_Team
  # V_VAL_FILE_NAME = "movers_500_0,30_500_merged_v_values_learned.pickle"
  NP_POLICY_A1 = "movers_bc_1000_pi_a1_alldata.npy"
  NP_POLICY_A2 = "movers_bc_1000_pi_a2_alldata.npy"
else:
  GAME_MAP = MAP_CLEANUP
  POLICY = Policy_Cleanup
  MDP_TASK = MDP_Cleanup_Task
  MDP_AGENT = MDP_Cleanup_Agent
  AGENT = BoxPushAIAgent_Indv
  TEST_AGENT = BoxPushAIAgent_Indv
  # V_VAL_FILE_NAME = "cleanup_v3_500_0,30_500_merged_v_values_learned.pickle"
  NP_POLICY_A1 = "cleanup_v3_btil2_policy_synth_woTx_FTTT_500_0,30_a1.npy"
  NP_POLICY_A2 = "cleanup_v3_btil2_policy_synth_woTx_FTTT_500_0,30_a2.npy"


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

    model_dir = DATA_DIR + "/learned_models/"  # noqa: E501
    np_policy_1 = np.load(model_dir + NP_POLICY_A1)
    np_policy_1 = np_policy_1 / np.sum(np_policy_1, axis=1)[:, None]
    test_policy_1 = NoMindCachedPolicy(np_policy_1, mdp_task, 0)
    np_policy_2 = np.load(model_dir + NP_POLICY_A2)
    np_policy_2 = np_policy_2 / np.sum(np_policy_2, axis=1)[:, None]
    test_policy_2 = NoMindCachedPolicy(np_policy_2, mdp_task, 1)

    mask = (False, True, True, True)
    agent1 = AIAgent_NoMind(test_policy_1, 0)
    agent2 = AIAgent_NoMind(test_policy_2, 1)

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


if __name__ == "__main__":
  app = BoxPushV2App()
  app.run()
