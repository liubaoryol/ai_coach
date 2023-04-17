import os
import glob
import random
import click
from ai_coach_core.model_learning.LatentIQL.train_mental_iql import (
    train_mental_iql)
from ai_coach_core.model_learning.LatentIQL.helper.utils import (
    conv_trajectories_2_iql_format)
import ai_coach_core.gym  # noqa: F401


# yapf: disable
@click.command()
@click.option("--domain", type=str, default="cleanup", help="cleanup")  # noqa: E501
@click.option("--num-data", type=int, default=500, help="")
# yapf: enable
def main(domain, num_data):
  save_prefix = ""
  if domain == "cleanup":
    from ai_coach_domain.cleanup_single.simulator import CleanupSingleSimulator
    from ai_coach_domain.cleanup_single.maps import MAP_SINGLE_V1
    from ai_coach_domain.cleanup_single.policy import Policy_CleanupSingle
    from ai_coach_domain.cleanup_single.mdp import MDPCleanupSingle
    from ai_coach_domain.cleanup_single.agent import Agent_CleanupSingle
    from ai_coach_domain.cleanup_single.utils import CleanupSingleTrajectories

    sim = CleanupSingleSimulator()
    TEMPERATURE = 0.3
    GAME_MAP = MAP_SINGLE_V1
    save_prefix += GAME_MAP["name"]
    mdp_task = MDPCleanupSingle(**GAME_MAP)
    policy = Policy_CleanupSingle(mdp_task, TEMPERATURE)
    agent = Agent_CleanupSingle(policy)
    train_data = CleanupSingleTrajectories(mdp_task)

    init_bstate = [0] * len(GAME_MAP["boxes"])
    init_pos = GAME_MAP["init_pos"]
    init_sidx = mdp_task.conv_sim_states_to_mdp_sidx((init_bstate, init_pos))
    possible_init_states = [init_sidx]

  sim.init_game(**GAME_MAP)

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
  LOG_DIR = os.path.join(os.path.dirname(__file__), "logs/")
  if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
  output_dir = os.path.join(os.path.dirname(__file__), "output/")
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  ##################################################

  env_kwargs = {
      'mdp': mdp_task,
      'possible_init_states': possible_init_states,
      'use_central_action': True
  }

  num_iterations = 100000
  seed = 0
  batch_size = 200
  learn_alpha = True
  clip_grad_val = 0.5
  learning_rate = 0.0003

  MIQL = True
  if MIQL:
    train_mental_iql('envfrommdp-v0',
                     env_kwargs,
                     seed,
                     batch_size,
                     mdp_task.num_latents,
                     path_iq_data,
                     num_traj,
                     LOG_DIR,
                     output_dir,
                     replay_mem=1000,
                     eps_steps=200,
                     eps_window=10,
                     num_learn_steps=num_iterations,
                     log_interval=100,
                     eval_interval=1000,
                     list_hidden_dims=[128, 128],
                     clip_grad_val=clip_grad_val,
                     learn_alpha=learn_alpha,
                     learning_rate=learning_rate,
                     gumbel_temperature=1.0)


if __name__ == "__main__":
  main()
