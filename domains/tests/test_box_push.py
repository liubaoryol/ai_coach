import glob
import os
import random
from ai_coach_domain.box_push.simulator import (BoxPushSimulator_AlwaysAlone)
from ai_coach_domain.box_push.mdp import (BoxPushAgentMDP_AlwaysAlone)
# from ai_coach_domain.box_push.policy import get_test_indv_action
from ai_coach_domain.box_push.policy import get_indv_action
from ai_coach_domain.box_push.maps import EXP1_MAP
from ai_coach_domain.box_push.transition_x import (get_a1_latent_indv,
                                                   get_a2_latent_indv,
                                                   get_valid_box_to_pickup)

BoxPushSimulator = BoxPushSimulator_AlwaysAlone
BoxPushMDP = BoxPushAgentMDP_AlwaysAlone

if __name__ == "__main__":
  game_map = EXP1_MAP
  DATA_DIR = os.path.join(os.path.dirname(__file__), "data/vi_box_push/")

  sim = BoxPushSimulator(0)
  sim.init_game(**game_map)
  sim.max_steps = 200

  mdp = BoxPushMDP(**game_map)

  def get_a1_action(**kwargs):
    return get_indv_action(mdp, BoxPushSimulator.AGENT1, 0.3, **kwargs)

  def get_a2_action(**kwargs):
    return get_indv_action(mdp, BoxPushSimulator.AGENT2, 0.3, **kwargs)

  def get_a1_latent(cur_state, a1_action, a2_action, a1_latent, next_state):
    num_drops = len(sim.drops)
    num_goals = len(sim.goals)
    return get_a1_latent_indv(cur_state, a1_action, a2_action, a1_latent,
                              next_state, num_drops, num_goals)

  def get_a2_latent(cur_state, a1_action, a2_action, a2_latent, next_state):
    num_drops = len(sim.drops)
    num_goals = len(sim.goals)
    return get_a2_latent_indv(cur_state, a1_action, a2_action, a2_latent,
                              next_state, num_drops, num_goals)

  def get_init_x():
    a1_latent = None
    a2_latent = None
    valid_boxes = get_valid_box_to_pickup(sim.box_states, len(sim.drops),
                                          len(sim.goals))

    if len(valid_boxes) > 0:
      box_idx = random.choice(valid_boxes)
      a1_latent = ("pickup", box_idx)
      box_idx2 = random.choice(valid_boxes)
      a2_latent = ("pickup", box_idx2)
    return a1_latent, a2_latent

  sim.set_autonomous_agent(cb_get_A1_action=get_a1_action,
                           cb_get_A2_action=get_a2_action,
                           cb_get_A1_mental_state=get_a1_latent,
                           cb_get_A2_mental_state=get_a2_latent,
                           cb_get_init_mental_state=get_init_x)

  GEN_TRAIN_SET = False
  file_prefix = "test_"
  if GEN_TRAIN_SET:
    file_names = glob.glob(os.path.join(DATA_DIR, file_prefix + '*.txt'))
    for fmn in file_names:
      os.remove(fmn)

    sim.run_simulation(100, os.path.join(DATA_DIR, file_prefix), "header")

  TEST_VI = True
  if TEST_VI:
    from ai_coach_core.model_inference.var_infer.var_infer_dynamic_x import (
        VarInferDuo)
    from ai_coach_domain.box_push.helper import transition_always_alone
    file_names = glob.glob(os.path.join(DATA_DIR, file_prefix + '*.txt'))

    # file_names = [file_names[0]]
    trajectories = []
    for idx, file_nm in enumerate(file_names):
      trj = BoxPushSimulator.read_file(file_nm)
      trj_mdp = []
      for bstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat in trj:
        sidx = mdp.conv_sim_states_to_mdp_sidx(a1pos, a2pos, bstt)
        aidx1 = mdp.conv_sim_action_to_mdp_aidx(a1act)
        aidx2 = mdp.conv_sim_action_to_mdp_aidx(a2act)
        if idx < 50:
          xidx1 = mdp.latent_space.state_to_idx[a1lat]
          xidx2 = mdp.latent_space.state_to_idx[a2lat]
        else:
          xidx1 = None
          xidx2 = None

        trj_mdp.append([sidx, (aidx1, aidx2), (xidx1, xidx2)])

      trajectories.append(trj_mdp)
    # print(len(trajectories))
    # print(trajectories)

    def transition_s(sidx, aidx1, aidx2, sidx_n):
      a1_pos, a2_pos, box_states = mdp.conv_mdp_sidx_to_sim_states(sidx)
      act1 = mdp.conv_mdp_aidx_to_sim_action(aidx1)
      act2 = mdp.conv_mdp_aidx_to_sim_action(aidx2)
      list_p_next_env = transition_always_alone(box_states, a1_pos, a2_pos,
                                                act1, act2, sim.boxes,
                                                sim.goals, sim.walls, sim.drops,
                                                sim.x_grid, sim.y_grid)
      for p, box_states_n, a1_pos_n, a2_pos_n in list_p_next_env:
        sidx_new = mdp.conv_sim_states_to_mdp_sidx(a1_pos_n, a2_pos_n,
                                                   box_states_n)
        if sidx_new == sidx_n:
          return p
      return 0

    var_inf = VarInferDuo(trajectories, mdp.num_states, mdp.num_latents,
                          (mdp.num_actions, mdp.num_actions), transition_s)

    var_inf.set_dirichlet_prior(1.5, 1.5, 1.5)
    var_inf.do_inference()

    var_inf.list_np_policy[0].shape
