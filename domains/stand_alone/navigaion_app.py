import numpy as np
import ai_coach_domain.navigation.maps as nv_maps
import ai_coach_domain.navigation.mdp as nv_mdp
from ai_coach_domain.navigation.simulator import NavigationSimulator
import ai_coach_domain.navigation.policy as nv_pol
from stand_alone.box_push_app import BoxPushApp

GAME_MAP = nv_maps.NAVI_MAP
MDP_AGENT = nv_mdp.NavigationMDP(**GAME_MAP)

TEMPER = 0.3


class NavigationApp(BoxPushApp):
  def __init__(self) -> None:
    super().__init__()

  def _init_game(self):
    'define game related variables and objects'
    GAME_ENV_ID = 0
    # game_map["a2_init"] = (1, 2)
    self.x_grid = GAME_MAP["x_grid"]
    self.y_grid = GAME_MAP["y_grid"]
    self.game = NavigationSimulator(GAME_ENV_ID, nv_mdp.transition_navi)
    self.game.max_steps = 500

    self.game.init_game(**GAME_MAP)

    def get_a1_action(**kwargs):
      return nv_pol.get_static_action(MDP_AGENT, NavigationSimulator.AGENT1,
                                      TEMPER, **kwargs)

    def get_a2_action(**kwargs):
      return nv_pol.get_static_action(MDP_AGENT, NavigationSimulator.AGENT2,
                                      TEMPER, **kwargs)

    def get_init_x(box_states, a1_pos, a2_pos):
      a1 = "GH1" if np.random.randint(2) == 0 else "GH2"
      a2 = 1 if np.random.randint(2) == 0 else 2
      a3 = "GH1" if np.random.randint(2) == 0 else "GH2"
      a4 = 1 if np.random.randint(2) == 0 else 2
      return (a1, a2), (a3, a4)
      # return ("GH1", 1), ("GH1", 1)

    self.game.set_autonomous_agent(cb_get_A1_action=get_a1_action,
                                   cb_get_A2_action=get_a2_action,
                                   cb_get_init_mental_state=get_init_x)


if __name__ == "__main__":
  app = NavigationApp()
  app.run()
