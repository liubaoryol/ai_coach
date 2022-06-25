import numpy as np
from ai_coach_core.latent_inference.decoding import (most_probable_sequence)
from ai_coach_domain.box_push.agent_model import (
    assumed_initial_mental_distribution)

if __name__ == "__main__":
  import glob
  from ai_coach_domain.box_push.utils import BoxPushTrajectories

  import ai_coach_domain.box_push.maps as bp_maps
  import ai_coach_domain.box_push.simulator as bp_sim
  import ai_coach_domain.box_push.mdp as bp_mdp

  # Set the domain
  ############################################################################
  GAME_MAP = bp_maps.EXP1_MAP

  BoxPushSimulator = bp_sim.BoxPushSimulator_AlwaysTogether
  sim = BoxPushSimulator(0)
  sim.init_game(**GAME_MAP)

  TEMPERATURE = 0.3

  MDP_AGENT = bp_mdp.BoxPushTeamMDP_AlwaysTogether(**GAME_MAP)
  MDP_TASK = MDP_AGENT

  # load test trajectories
  ############################################################################
  traj_dir = "misc/BTIL_results/data/exp1_team_box_push_test/"
  test_file_names = glob.glob(traj_dir + '*.txt')

  test_data = BoxPushTrajectories(sim, MDP_TASK, MDP_AGENT)
  test_data.load_from_files(test_file_names)
  test_traj = test_data.get_as_column_lists(include_terminal=False)

  idx = 0
  sample = test_traj[idx]

  # load models
  ############################################################################
  model_dir = "misc/BTIL_results/data/learned_models/"

  policy_file = model_dir + "exp1_team_btil_policy_human_woTx_66_1.00_a1.npy"
  policy = np.load(policy_file)

  tx_file = model_dir + "exp1_team_btil_tx_human_66_1.00_a1.npy"
  tx = np.load(tx_file)

  # human mental state inference
  ############################################################################
  def policy_nxsa(nidx, xidx, sidx, tuple_aidx):
    return policy[xidx, sidx, tuple_aidx[0]]

  def Tx_nxsasx(nidx, xidx, sidx, tuple_aidx, sidx_n, xidx_n):
    return tx[xidx, sidx, tuple_aidx[0], tuple_aidx[1], xidx_n]

  def init_latent_nxs(nidx, xidx, sidx):
    return assumed_initial_mental_distribution(0, sidx, MDP_AGENT)[xidx]

  inferred_x = most_probable_sequence(sample[0], sample[1], 1,
                                      MDP_AGENT.num_latents, policy_nxsa,
                                      Tx_nxsasx, init_latent_nxs)

  true_x = list(zip(*sample[2]))[0]
  print(inferred_x)
  print(true_x)
