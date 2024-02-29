import logging
import os
import click

@click.command()
@click.option("--traj-path", type=str, default="tw2020_trajectory", help="") 
# yapf: enable
def main(traj_path):    
    DATA_DIR = os.path.join(os.path.dirname(__file__), f"{traj_path}/")
    TRAIN_DIR_PREFIX = os.path.join(os.path.dirname(__file__), "data/")
    TRAIN_DIR_MOVER = os.path.join(TRAIN_DIR_PREFIX, "movers" + "_train")
    TRAIN_DIR_RESCUE2 = os.path.join(TRAIN_DIR_PREFIX, "rescue_2" + "_train")

    # Create train folders if they are not there
    if not os.path.exists(TRAIN_DIR_MOVER):
        os.mkdir(TRAIN_DIR_MOVER)
    
    if not os.path.exists(TRAIN_DIR_RESCUE2):
        os.mkdir(TRAIN_DIR_RESCUE2)

    MOVER_DCOL_PREFIX = "dcol_session_a"
    RESCUE_2_DCOL_PREFIX = "dcol_session_c"
    print(f"TRAIN_DIR_MOVER: {TRAIN_DIR_MOVER}")
    print(f"TRAIN_DIR_RESCUE2: {TRAIN_DIR_RESCUE2}")

    # Iterate through all users, add dcol to corresponding training data folder
    for item in os.listdir(DATA_DIR):
        experiment_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(experiment_path):
            for session_name in os.listdir(experiment_path):
                print(session_name)
                source_path = os.path.join(experiment_path, session_name)
                if session_name.startswith(MOVER_DCOL_PREFIX):
                    target_path = os.path.join(TRAIN_DIR_MOVER, session_name)
                    print(f"Source path: {source_path}")
                    print(f"Target path: {target_path}")
                    os.rename(source_path, target_path)
                if session_name.startswith(RESCUE_2_DCOL_PREFIX):
                    
                    target_path = os.path.join(TRAIN_DIR_RESCUE2, session_name)
                    print(f"Source path: {source_path}")
                    print(f"Target path: {target_path}")
                    os.rename(source_path, target_path)


if __name__ == "__main__":
  logging.basicConfig(
      level=logging.INFO,
      format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
      handlers=[logging.StreamHandler()],
      force=True)
  main()