import numpy as np
from ai_coach_domain.box_push_v2.simulator import BoxPushSimulatorV2
from ai_coach_domain.box_push_v2.maps import MAP_MOVERS, MAP_CLEANUP
from ai_coach_domain.box_push_v2.mdp import (MDP_Movers_Task, MDP_Movers_Agent,
                                             MDP_Cleanup_Task,
                                             MDP_Cleanup_Agent)
from ai_coach_domain.box_push_v2.policy import Policy_Movers, Policy_Cleanup
from ai_coach_domain.box_push_v2.agent import (BoxPushAIAgent_PO_Team,
                                               BoxPushAIAgent_PO_Indv,
                                               BoxPushAIAgent_BTIL,
                                               BoxPushAIAgent_Team)
from ai_coach_domain.agent import InteractiveAgent, BTILCachedPolicy
from stand_alone.box_push_app import BoxPushApp

IS_MOVERS = True
if IS_MOVERS:
  GAME_MAP = MAP_MOVERS
  POLICY = Policy_Movers
  MDP_TASK = MDP_Movers_Task
  MDP_AGENT = MDP_Movers_Agent
  AGENT = BoxPushAIAgent_PO_Team
else:
  GAME_MAP = MAP_CLEANUP
  POLICY = Policy_Cleanup
  MDP_TASK = MDP_Cleanup_Task
  MDP_AGENT = MDP_Cleanup_Agent
  AGENT = BoxPushAIAgent_PO_Indv


class StaticBoxPushApp(BoxPushApp):
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

    TEST_BTIL_AGENT = False
    TEST_BTIL_USE_TRUE_TX = False

    mdp_task = MDP_TASK(**GAME_MAP)
    mdp_agent = MDP_AGENT(**GAME_MAP)
    if not TEST_BTIL_AGENT:
      TEMPERATURE = 0.3
      policy2 = POLICY(mdp_task, mdp_agent, TEMPERATURE, agent_idx=1)
      agent1 = InteractiveAgent()
      init_states = ([0] * len(GAME_MAP["boxes"]), GAME_MAP["a1_init"],
                     GAME_MAP["a2_init"])
      agent2 = AGENT(init_states, policy2, agent_idx=BoxPushSimulatorV2.AGENT2)
    else:
      data_dir = "/home/sangwon/Projects/ai_coach/misc/BTIL_feedback_results/data/learned_models/"  # noqa: E501
      np_policy_1 = np.load(
          data_dir + "movers_v2_btil_policy_synth_woTx_FTTT_500_0.30_a1.npy")
      test_policy_1 = BTILCachedPolicy(np_policy_1, mdp_task, 0)
      np_policy_2 = np.load(
          data_dir + "movers_v2_btil_policy_synth_woTx_FTTT_500_0.30_a2.npy")
      test_policy_2 = BTILCachedPolicy(np_policy_2, mdp_task, 1)

      if TEST_BTIL_USE_TRUE_TX:
        agent1 = BoxPushAIAgent_Team(test_policy_1, agent_idx=0)
        agent2 = BoxPushAIAgent_Team(test_policy_2, agent_idx=1)
      else:
        np_tx_1 = np.load(data_dir +
                          "movers_v2_btil_tx_synth_FTTT_500_0.30_a1.npy")
        np_tx_2 = np.load(data_dir +
                          "movers_v2_btil_tx_synth_FTTT_500_0.30_a2.npy")
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


if __name__ == "__main__":
  app = StaticBoxPushApp()
  app.run()
