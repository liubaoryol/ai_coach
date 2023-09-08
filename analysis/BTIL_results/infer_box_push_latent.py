import numpy as np
import os
from aic_core.utils.decoding import (most_probable_sequence, forward_inference)
from aic_domain.box_push.agent_model import (assumed_initial_mental_distribution
                                             )

if __name__ == "__main__":
  import glob
  from aic_domain.box_push.utils import BoxPushTrajectories

  import aic_domain.box_push.maps as bp_maps
  import aic_domain.box_push.simulator as bp_sim
  import aic_domain.box_push.mdp as bp_mdp

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
  traj_dir = os.path.dirname(__file__) + "/data/exp1_team_box_push_test/"
  test_file_names = glob.glob(traj_dir + '*.txt')

  trajories = BoxPushTrajectories(sim, MDP_TASK, MDP_AGENT)
  trajories.load_from_files(test_file_names)
  list_sax_columns = trajories.get_as_column_lists(include_terminal=False)
  list_sax_columns_w_terminal = trajories.get_as_column_lists(
      include_terminal=True)

  # load models
  ############################################################################
  model_dir = os.path.dirname(__file__) + "/data/learned_models/"

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

  idx = 0
  sax_sample = list_sax_columns[idx]
  inferred_x_seq = most_probable_sequence(sax_sample[0], sax_sample[1], 1,
                                          MDP_AGENT.num_latents, policy_nxsa,
                                          Tx_nxsasx, init_latent_nxs)

  true_x_seq = list(zip(*sax_sample[2]))[0]

  sax_sample_w_term = list_sax_columns_w_terminal[idx]

  list_inferred_x = []
  list_x_dist = []

  prev_dist = None
  for step in range(1, len(sax_sample_w_term[0])):
    inferred_x, x_dist = forward_inference(sax_sample_w_term[0][:step],
                                           sax_sample_w_term[1][:step - 1], 1,
                                           MDP_AGENT.num_latents, policy_nxsa,
                                           Tx_nxsasx, init_latent_nxs)
    list_inferred_x.append(inferred_x[0])
    list_x_dist.append(x_dist[0])

    inferred_x_, x_dist_ = forward_inference(sax_sample_w_term[0][:step],
                                             sax_sample_w_term[1][:step - 1],
                                             1,
                                             MDP_AGENT.num_latents,
                                             policy_nxsa,
                                             Tx_nxsasx,
                                             init_latent_nxs,
                                             list_np_prev_px=prev_dist)
    prev_dist = x_dist_

  np_decoding = np.vstack([inferred_x_seq[0], list_inferred_x,
                           true_x_seq]).transpose()
  print(np_decoding)
