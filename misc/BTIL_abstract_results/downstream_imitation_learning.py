from typing import Sequence
import os
import glob
import click
from tqdm import tqdm
import random
from aic_core.utils.decoding import (smooth_inference_max_z,
                                     smooth_inference_zx)
from aic_ml.baselines.sb3_algorithms import behavior_cloning_sb3
import numpy as np
from datetime import datetime
# rl algorithm


def downstream_imitation_learning(trajectories_w_prob: Sequence[np.ndarray]):
  pass


@click.command()
@click.option("--domain", type=str, default="movers", help="")
def main(domain):

  # define the domain where trajectories were generated
  ##################################################
  if domain == "movers":
    from aic_domain.box_push.utils import BoxPushTrajectories
    from aic_domain.box_push_v2.agent import BoxPushAIAgent_Team
    from aic_domain.box_push_v2.maps import MAP_MOVERS
    from aic_domain.box_push_v3.simulator import BoxPushSimulatorV3
    from aic_domain.box_push_v3.policy import Policy_MoversV3
    from aic_domain.box_push_v3.mdp import (MDP_MoversV3_Agent,
                                            MDP_MoversV3_Task)
    sim = BoxPushSimulatorV3(False)
    TEMPERATURE = 0.3
    GAME_MAP = MAP_MOVERS
    SAVE_PREFIX = GAME_MAP["name"]
    MDP_TASK = MDP_MoversV3_Task(**GAME_MAP)
    MDP_AGENT = MDP_MoversV3_Agent(**GAME_MAP)
    POLICY_1 = Policy_MoversV3(MDP_TASK, MDP_AGENT, TEMPERATURE, 0)
    POLICY_2 = Policy_MoversV3(MDP_TASK, MDP_AGENT, TEMPERATURE, 1)
    # init_states = ([0] * len(GAME_MAP["boxes"]), GAME_MAP["a1_init"],
    #                GAME_MAP["a2_init"])
    AGENT_1 = BoxPushAIAgent_Team(POLICY_1, agent_idx=sim.AGENT1)
    AGENT_2 = BoxPushAIAgent_Team(POLICY_2, agent_idx=sim.AGENT2)
    AGENTS = [AGENT_1, AGENT_2]
    train_data = BoxPushTrajectories(MDP_TASK, MDP_AGENT)
  elif domain == "cleanup_v3":
    from aic_domain.box_push.utils import BoxPushTrajectories
    from aic_domain.box_push_v2.agent import BoxPushAIAgent_Indv
    from aic_domain.box_push_v2.simulator import BoxPushSimulatorV2
    from aic_domain.box_push_v2.maps import MAP_CLEANUP_V3
    from aic_domain.box_push_v2.policy import Policy_Cleanup
    from aic_domain.box_push_v2.mdp import (MDP_Cleanup_Agent, MDP_Cleanup_Task)
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
  TRAIN_DIR = os.path.join(DATA_DIR, SAVE_PREFIX + '_train_opt_sameinit')

  file_names = glob.glob(os.path.join(TRAIN_DIR, '*.txt'))
  random.shuffle(file_names)

  train_data.load_from_files(file_names)
  # traj_labeled_ver = train_data.get_as_row_lists(no_latent_label=False,
  #                                                include_terminal=True)
  # traj_unlabel_ver = train_data.get_as_row_lists(no_latent_label=True,
  #                                                include_terminal=True)
  list_trajs = train_data.get_as_column_lists(include_terminal=True)
  num_traj = len(list_trajs)

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
  model_dir = os.path.join(DATA_DIR, "learned_models/")

  num_train = 1000
  num_x = 4
  num_abs = 30

  num_agents = len(AGENTS)
  model_dir = os.path.join(DATA_DIR, "learned_models/")
  filename_pre = (
      f"{domain}_btil_abs_{tx_dependency}_{num_train}_{num_x}_{num_abs}")
  file_name = filename_pre + ("_abs.npy")
  np_abs = np.load(model_dir + file_name)

  for idx_a in range(num_agents):
    file_name = filename_pre + f"_pi_a{idx_a+1}.npy"
    list_np_pi.append(np.load(model_dir + file_name))

    file_name = filename_pre + f"_tx_a{idx_a+1}.npy"
    list_np_tx.append(np.load(model_dir + file_name))

    file_name = filename_pre + f"_bx_a{idx_a+1}.npy"
    list_np_bx.append(np.load(model_dir + file_name))

  # prediction
  ##################################################
  tup_num_latent = (num_x, ) * num_agents

  # traj_sa_joint = []
  # for traj in traj_labeled_ver:
  #   traj_sa = []
  #   for s, a, x in traj:
  #     traj_sa.append((s, a))

  #   traj_sa_joint.append(traj_sa)
  from gym_custom.envs.mdp_env.env_aicoaching import EnvFromLearnedModels
  env = EnvFromLearnedModels(MDP_TASK,
                             np_abs,
                             list_np_pi,
                             list_np_tx,
                             list_np_bx, [0],
                             use_central_action=True)
  list_predict = []
  INFERENCE_MAX_Z = False
  if INFERENCE_MAX_Z:
    for i, traj in tqdm(enumerate(list_trajs)):
      list_states, list_actions, list_latents = traj
      list_zx = smooth_inference_max_z(list_states, list_actions, num_agents,
                                       tup_num_latent, num_abs, np_abs,
                                       list_np_pi, list_np_tx, list_np_bx)
      predictions = []
      traj_len = len(list_zx[1])
      for t in range(traj_len):
        x_tup = []
        for aidx in range(1, len(list_zx)):
          x_tup.append(np.argmax(list_zx[aidx][t]))
        x_tup = tuple(x_tup)
        predictions.append((list_zx[0][t], env.each_2_joint[x_tup]))

      if len(list_zx[0]) > traj_len:
        predictions.append((list_zx[0][traj_len], None))

      list_predict.append(predictions)
  else:
    for i, traj in tqdm(enumerate(list_trajs)):
      list_states, list_actions, list_latents = traj
      tmp_infer_dist = smooth_inference_zx(list_states, list_actions,
                                           num_agents, tup_num_latent, num_abs,
                                           np_abs, list_np_pi, list_np_tx,
                                           list_np_bx)
      max_idx = tmp_infer_dist.reshape(tmp_infer_dist.shape[0], -1).argmax(1)
      max_coords = np.unravel_index(max_idx, tmp_infer_dist.shape[1:])
      max_coords = list(zip(*max_coords))
      # store z, and x's separately.
      list_predict.append([(elem[0], env.each_2_joint[tuple(elem[1:])])
                           for elem in max_coords])
      if len(list_states) > len(list_actions):
        list_predict[-1].append((np_abs[list_states[-1]].argmax(), None))

  LOG_DIR = os.path.join(os.path.dirname(__file__), "logs/")
  if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
  logpath = LOG_DIR + str(datetime.today())

  BC = True
  num_iterations = 1000
  save_prefix = SAVE_PREFIX
  policy = None
  if BC:
    save_prefix += "_bc_abs_"
    policy = behavior_cloning_sb3(list_predict, num_abs,
                                  np.prod(tup_num_latent), logpath + "/bc_z",
                                  num_iterations)
    policy = policy.reshape(num_abs, *tup_num_latent)

  temp_dir = DATA_DIR + "learned_models/"
  if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
  save_prefix += ("%d" % (len(list_predict), ))

  # save models
  save_prefix = os.path.join(temp_dir, save_prefix)
  if policy is not None:
    np.save(save_prefix + "_pi_z", policy)


if __name__ == "__main__":
  main()
