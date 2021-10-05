import numpy as np
import ai_coach_domain.box_push.maps as bp_maps
import ai_coach_domain.box_push_static.mdp as bps_mdp
from ai_coach_domain.box_push.simulator import BoxPushSimulator_AloneOrTogether
from stand_alone.box_push_app import BoxPushApp
from ai_coach_domain.box_push_static.policy import get_static_action

GAME_MAP = bp_maps.EXP1_MAP
MDP_AGENT = bps_mdp.StaticBoxPushMDP(**GAME_MAP)


class TEST_StaticBoxPushSimulator(BoxPushSimulator_AloneOrTogether):
  def take_a_step(self, map_agent_2_action):
    a1_action = None
    if BoxPushSimulator_AloneOrTogether.AGENT1 in map_agent_2_action:
      a1_action = map_agent_2_action[BoxPushSimulator_AloneOrTogether.AGENT1]

    a2_action = None
    if BoxPushSimulator_AloneOrTogether.AGENT2 in map_agent_2_action:
      a2_action = map_agent_2_action[BoxPushSimulator_AloneOrTogether.AGENT2]

    if a1_action is None and a2_action is None:
      return

    if a1_action is None:
      a1_action = bps_mdp.EventType.STAY

    if a2_action is None:
      a2_action = bps_mdp.EventType.STAY

    sidx = MDP_AGENT.conv_sim_states_to_mdp_sidx(self.a1_pos, self.a2_pos,
                                                 self.box_states)
    aidx1 = MDP_AGENT.a1_a_space.action_to_idx[a1_action]
    aidx2 = MDP_AGENT.a2_a_space.action_to_idx[a2_action]
    joint_aidx = MDP_AGENT.np_action_to_idx[int(aidx1), int(aidx2)]
    xidx = MDP_AGENT.latent_space.state_to_idx[self.a1_latent]
    print(self.a1_latent[0] + ", " + a1_action.name)
    print(self.a2_latent[0] + ", " + a2_action.name)
    print(MDP_AGENT.reward(xidx, sidx, joint_aidx))

    super().take_a_step(map_agent_2_action)


class StaticBoxPushApp(BoxPushApp):
  def __init__(self) -> None:
    super().__init__()

  def _init_game(self):
    'define game related variables and objects'
    GAME_ENV_ID = 0
    # game_map["a2_init"] = (1, 2)
    self.x_grid = GAME_MAP["x_grid"]
    self.y_grid = GAME_MAP["y_grid"]
    self.game = TEST_StaticBoxPushSimulator(GAME_ENV_ID)
    self.mdp_agent = MDP_AGENT
    self.game.max_steps = 100

    self.game.init_game(**GAME_MAP)
    temperature = 1

    def get_a1_action(**kwargs):
      return get_static_action(self.mdp_agent,
                               BoxPushSimulator_AloneOrTogether.AGENT1,
                               temperature, **kwargs)

    def get_a2_action(**kwargs):
      return get_static_action(self.mdp_agent,
                               BoxPushSimulator_AloneOrTogether.AGENT2,
                               temperature, **kwargs)

    def get_init_x(box_states, a1_pos, a2_pos):
      a1_latent = ("together", 0) if np.random.randint(2) == 0 else ("alone", 0)
      a2_latent = ("together", 0) if np.random.randint(2) == 0 else ("alone", 0)
      return a1_latent, a2_latent

    self.game.set_autonomous_agent(cb_get_A1_action=get_a1_action,
                                   cb_get_A2_action=get_a2_action,
                                   cb_get_A1_mental_state=None,
                                   cb_get_A2_mental_state=None,
                                   cb_get_init_mental_state=get_init_x)


if __name__ == "__main__":
  app = StaticBoxPushApp()
  app.run()
