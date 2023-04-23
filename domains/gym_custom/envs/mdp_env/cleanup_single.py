from ai_coach_core.gym.envs.env_from_mdp import EnvFromMDP
from ai_coach_domain.cleanup_single.mdp import MDPCleanupSingle
from ai_coach_domain.cleanup_single.maps import MAP_SINGLE_V1


class CleanupSingleEnv_v0(EnvFromMDP):

  def __init__(self):
    game_map = MAP_SINGLE_V1
    mdp = MDPCleanupSingle(**game_map)
    init_bstate = [0] * len(game_map["boxes"])
    init_pos = game_map["init_pos"]
    init_sidx = mdp.conv_sim_states_to_mdp_sidx((init_bstate, init_pos))
    possible_init_states = [init_sidx]

    super().__init__(mdp, possible_init_states, use_central_action=True)
