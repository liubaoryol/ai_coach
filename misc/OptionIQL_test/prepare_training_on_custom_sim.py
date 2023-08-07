import os
import glob
import random
from ai_coach_core.model_learning.OptionIQL.helper.utils import (
    conv_trajectories_2_iql_format)
import ai_coach_core.gym  # noqa: F401


def prepare_training_on_custom_sim(domain):
  save_prefix = ""
  if domain == "cleanup":
    from ai_coach_domain.cleanup_single.maps import MAP_SINGLE_V1
    from ai_coach_domain.cleanup_single.mdp import MDPCleanupSingle
    from ai_coach_domain.cleanup_single.utils import CleanupSingleTrajectories

    GAME_MAP = MAP_SINGLE_V1
    save_prefix += GAME_MAP["name"]
    mdp_task = MDPCleanupSingle(**GAME_MAP)
    train_data = CleanupSingleTrajectories(mdp_task)

    init_bstate = [0] * len(GAME_MAP["boxes"])
    init_pos = GAME_MAP["init_pos"]
    init_sidx = mdp_task.conv_sim_states_to_mdp_sidx((init_bstate, init_pos))
    possible_init_states = [init_sidx]

  # load files
  ##################################################
  DATA_DIR = os.path.join(os.path.dirname(__file__), "data/")
  dir_suffix = "_train"

  TRAIN_DIR = os.path.join(DATA_DIR, save_prefix + dir_suffix)

  file_names = glob.glob(os.path.join(TRAIN_DIR, '*.txt'))
  random.shuffle(file_names)

  train_data.load_from_files(file_names)
  traj_labeled_ver = train_data.get_as_row_lists(no_latent_label=False,
                                                 include_terminal=True)
  num_traj = len(traj_labeled_ver)

  # convert trajectories
  dir_iq_data = os.path.join(DATA_DIR, "iq_data", f"{domain}_{num_traj}")
  if not os.path.exists(dir_iq_data):
    os.makedirs(dir_iq_data)
  path_iq_data = os.path.join(dir_iq_data, f"{domain}_{num_traj}.pkl")

  conv_trajectories_2_iql_format(traj_labeled_ver, lambda a: a, lambda s, a: -1,
                                 path_iq_data)

  ##################################################

  env_kwargs = {
      'mdp': mdp_task,
      'possible_init_states': possible_init_states,
      'use_central_action': True
  }

  return env_kwargs
