from typing import Sequence
import os
import glob
import click
import random
from ai_coach_core.latent_inference.decoding import smooth_inference
import numpy as np
# rl algorithm


def downstream_imitation_learning(trajectories_w_prob: Sequence[np.ndarray]):
  pass


@click.command()
@click.option("--domain", type=str, default="movers", help="")
def main(domain):

  # define the domain where trajectories were generated
  ##################################################
  if domain == "movers":
    from ai_coach_domain.box_push.utils import BoxPushTrajectories
    from ai_coach_domain.box_push_v2.agent import BoxPushAIAgent_Team
    from ai_coach_domain.box_push_v2.simulator import BoxPushSimulatorV2
    from ai_coach_domain.box_push_v2.maps import MAP_MOVERS
    from ai_coach_domain.box_push_v2.policy import Policy_Movers
    from ai_coach_domain.box_push_v2.mdp import (MDP_Movers_Agent,
                                                 MDP_Movers_Task)
    sim = BoxPushSimulatorV2(0)
    TEMPERATURE = 0.3
    GAME_MAP = MAP_MOVERS
    SAVE_PREFIX = GAME_MAP["name"]
    MDP_TASK = MDP_Movers_Task(**GAME_MAP)
    MDP_AGENT = MDP_Movers_Agent(**GAME_MAP)
    POLICY_1 = Policy_Movers(MDP_TASK, MDP_AGENT, TEMPERATURE, 0)
    POLICY_2 = Policy_Movers(MDP_TASK, MDP_AGENT, TEMPERATURE, 1)
    # init_states = ([0] * len(GAME_MAP["boxes"]), GAME_MAP["a1_init"],
    #                GAME_MAP["a2_init"])
    AGENT_1 = BoxPushAIAgent_Team(POLICY_1, agent_idx=sim.AGENT1)
    AGENT_2 = BoxPushAIAgent_Team(POLICY_2, agent_idx=sim.AGENT2)
    AGENTS = [AGENT_1, AGENT_2]
    train_data = BoxPushTrajectories(MDP_TASK, MDP_AGENT)
  elif domain == "cleanup_v3":
    from ai_coach_domain.box_push.utils import BoxPushTrajectories
    from ai_coach_domain.box_push_v2.agent import BoxPushAIAgent_Indv
    from ai_coach_domain.box_push_v2.simulator import BoxPushSimulatorV2
    from ai_coach_domain.box_push_v2.maps import MAP_CLEANUP_V3
    from ai_coach_domain.box_push_v2.policy import Policy_Cleanup
    from ai_coach_domain.box_push_v2.mdp import (MDP_Cleanup_Agent,
                                                 MDP_Cleanup_Task)
    sim = BoxPushSimulatorV2(0)
    TEMPERATURE = 0.3
    GAME_MAP = MAP_CLEANUP_V3
    SAVE_PREFIX = GAME_MAP["name"]
    MDP_TASK = MDP_Cleanup_Task(**GAME_MAP)
    MDP_AGENT = MDP_Cleanup_Agent(**GAME_MAP)
    POLICY_1 = Policy_Cleanup(MDP_TASK, MDP_AGENT, TEMPERATURE, 0)
    POLICY_2 = Policy_Cleanup(MDP_TASK, MDP_AGENT, TEMPERATURE, 1)
    AGENT_1 = BoxPushAIAgent_Indv(POLICY_1, agent_idx=sim.AGENT1)
    AGENT_2 = BoxPushAIAgent_Indv(POLICY_2, agent_idx=sim.AGENT2)
    AGENTS = [AGENT_1, AGENT_2]
    train_data = BoxPushTrajectories(MDP_TASK, MDP_AGENT)

  # load files
  ##################################################
  DATA_DIR = os.path.join(os.path.dirname(__file__), "data/")
  TRAIN_DIR = os.path.join(DATA_DIR, SAVE_PREFIX + '_train')

  file_names = glob.glob(os.path.join(TRAIN_DIR, '*.txt'))
  random.shuffle(file_names)

  train_data.load_from_files(file_names)
  traj_labeled_ver = train_data.get_as_row_lists(no_latent_label=False,
                                                 include_terminal=True)
  traj_unlabel_ver = train_data.get_as_row_lists(no_latent_label=True,
                                                 include_terminal=True)
  num_traj = len(traj_labeled_ver)

  # stats
  ##################################################
  np_traj_len = np.array([len(traj) for traj in traj_labeled_ver])
  print(len(np_traj_len[np_traj_len <= 60]))

  opt_idx = np.arange(num_traj)[np_traj_len <= 60]
  opt_labeled_trajs = [traj_labeled_ver[idx] for idx in opt_idx]

  counts = np.zeros((60, 4))
  for traj in opt_labeled_trajs:
    for t, (s, a, x) in enumerate(traj):
      if x is not None:
        counts[t][x[0]] += 1

  # optimal trajectories
  ##################################################
  opt_unlabeled_trajs = [traj_unlabel_ver[idx] for idx in opt_idx]

  # load models
  ##################################################
  np_abs = None
  list_np_pi = []
  list_np_tx = []
  list_np_bx = []

  if domain == "rescue_3":
    tx_dependency = "FTTTT"
  else:
    tx_dependency = "FTTT"

  num_train = 1000

  model_dir = os.path.join(DATA_DIR, "learned_models/")
  file_name = (f"{domain}_btil_abs_{tx_dependency}_{num_train}_abs.npy")
  np_abs = np.load(model_dir + file_name)

  for idx_a in range(len(AGENTS)):
    file_name = f"{domain}_btil_abs_{tx_dependency}_{num_train}_pi_a{idx_a}.npy"
    list_np_pi.append(np.load(model_dir + file_name))

    file_name = f"{domain}_btil_abs_{tx_dependency}_{num_train}_tx_a{idx_a}.npy"
    list_np_tx.append(np.load(model_dir + file_name))

    file_name = f"{domain}_btil_abs_{tx_dependency}_{num_train}_bx_a{idx_a}.npy"
    list_np_bx.append(np.load(model_dir + file_name))

  # prediction
  ##################################################
  num_latent = 5
  num_abstate = 30
  num_agents = len(AGENTS)
  tup_num_latent = (num_latent, ) * num_agents

  traj_sa_joint = []
  for traj in opt_unlabeled_trajs:
    traj_sa = []
    for s, a, x in traj:
      traj_sa.append((s, a))

    traj_sa_joint.append(traj_sa)

  list_list_predictions = []
  for idx_a in range(len(AGENTS)):
    list_predict = []
    for sa_traj in traj_sa_joint:
      list_predict.append(
          smooth_inference(sa_traj, len(AGENTS), tup_num_latent, num_abstate,
                           np_abs, list_np_pi, list_np_tx, list_np_bx))
    list_list_predictions.append(list_predict)


if __name__ == "__main__":
  pass
