import numpy as np
import os
from aic_domain.box_push_v3.simulator import BoxPushSimulatorV3
from aic_domain.box_push_v2.maps import MAP_MOVERS
from aic_domain.box_push_v2.maps import MAP_CLEANUP_V3 as MAP_CLEANUP
from aic_domain.box_push_v3.mdp import (MDP_MoversV3_Task, MDP_MoversV3_Agent,
                                        MDP_CleanupV3_Task, MDP_CleanupV3_Agent)
from aic_domain.box_push_v3.policy import Policy_MoversV3, Policy_CleanupV3
from aic_domain.box_push_v2.agent import (BoxPushAIAgent_PO_Team,
                                          BoxPushAIAgent_PO_Indv,
                                          BoxPushAIAgent_BTIL,
                                          BoxPushAIAgent_BTIL_ABS,
                                          BoxPushAIAgent_Team,
                                          BoxPushAIAgent_Indv, AIAgent_NoMind)
from aic_domain.agent.cached_agent import (BTILCachedPolicy, NoMindCachedPolicy)
from stand_alone.app_box_push import BoxPushApp

TEST_LEARNED_AGENT = True
IS_MOVERS = True
DATA_DIR = os.path.dirname(__file__) + "/data"
if IS_MOVERS:
  GAME_MAP = MAP_MOVERS
  POLICY = Policy_MoversV3
  MDP_TASK = MDP_MoversV3_Task
  MDP_AGENT = MDP_MoversV3_Agent
  AGENT = BoxPushAIAgent_Team
  TEST_AGENT = BoxPushAIAgent_Team
  # V_VAL_FILE_NAME = "movers_500_0,30_500_merged_v_values_learned.pickle"
  NP_POLICY_A1 = "movers_bc_bayes_abs_100_30_pi_a1.npy"
  NP_POLICY_A2 = "movers_bc_bayes_abs_100_30_pi_a2.npy"
  NP_ABS = "movers_bayes_abs_500_30_abs.npy"
else:
  GAME_MAP = MAP_CLEANUP
  POLICY = Policy_CleanupV3
  MDP_TASK = MDP_CleanupV3_Task
  MDP_AGENT = MDP_CleanupV3_Agent
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
    self.game = BoxPushSimulatorV3(False)
    self.game.max_steps = 200

    self.game.init_game(**GAME_MAP)

    mdp_task = MDP_TASK(**GAME_MAP)
    mdp_agent = MDP_AGENT(**GAME_MAP)
    self.mdp = mdp_task

    model_dir = DATA_DIR + "/learned_models/"  # noqa: E501
    if NP_ABS is not None:
      np_abs = np.load(model_dir + NP_ABS)
    else:
      np_abs = None

    np_policy_1 = np.load(model_dir + NP_POLICY_A1)
    np_policy_1 = np_policy_1 / np.sum(np_policy_1, axis=1)[:, None]
    test_policy_1 = NoMindCachedPolicy(np_policy_1, mdp_task, 0, np_abs)
    np_policy_2 = np.load(model_dir + NP_POLICY_A2)
    np_policy_2 = np_policy_2 / np.sum(np_policy_2, axis=1)[:, None]
    test_policy_2 = NoMindCachedPolicy(np_policy_2, mdp_task, 1, np_abs)

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
