import os
import glob
import click


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

    sim = CleanupSingleSimulator()
    TEMPERATURE = 0.3
    GAME_MAP = MAP_SINGLE_V1
    save_prefix += GAME_MAP["name"]
    mdp_task = MDPCleanupSingle(**GAME_MAP)
    policy = Policy_CleanupSingle(mdp_task, TEMPERATURE)
    agent = Agent_CleanupSingle(policy)

  sim.init_game(**GAME_MAP)
  sim.set_autonomous_agent(agent)

  # generate data
  ############################################################################
  DATA_DIR = os.path.join(os.path.dirname(__file__), "data/")
  train_dir = os.path.join(DATA_DIR, save_prefix + '_train')

  train_prefix = "train_"
  file_names = glob.glob(os.path.join(train_dir, train_prefix + '*.txt'))
  for fmn in file_names:
    os.remove(fmn)
  sim.run_simulation(num_data, os.path.join(train_dir, train_prefix), "header")


if __name__ == "__main__":
  main()
