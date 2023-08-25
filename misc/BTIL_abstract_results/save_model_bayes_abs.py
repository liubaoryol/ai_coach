import os
import glob
import click
import logging
import random
import numpy as np
from aic_ml.BTIL.bayesian_abstraction import (Bayes_Abstraction)


# yapf: disable
@click.command()
@click.option("--domain", type=str, default="movers", help="movers / cleanup_v3 / rescue_2 /rescue_3")  # noqa: E501
@click.option("--num-training-data", type=int, default=500, help="")
@click.option("--pi-prior", type=float, default=3, help="")
@click.option("--abs-prior", type=float, default=3, help="")
@click.option("--num-iteration", type=int, default=500, help="")
@click.option("--num-abstates", type=int, default=30, help="")
@click.option("--batch-size", type=int, default=500, help="")
# yapf: enable
def main(domain, num_training_data, pi_prior, abs_prior, batch_size,
         num_abstates, num_iteration):
  logging.info("domain: %s" % (domain, ))
  logging.info("num training data: %s" % (num_training_data, ))
  logging.info("pi prior: %s" % (pi_prior, ))
  logging.info("abs prior: %s" % (abs_prior, ))
  logging.info("batch size: %s" % (batch_size, ))
  logging.info("num abstates: %s" % (num_abstates, ))

  # define the domain where trajectories were generated
  ##################################################
  if domain == "movers":
    from ai_coach_domain.box_push.utils import BoxPushTrajectories
    from ai_coach_domain.box_push_v2.agent import BoxPushAIAgent_Team
    from ai_coach_domain.box_push_v3.simulator import BoxPushSimulatorV3
    from ai_coach_domain.box_push_v2.maps import MAP_MOVERS
    from ai_coach_domain.box_push_v3.policy import Policy_MoversV3
    from ai_coach_domain.box_push_v3.mdp import (MDP_MoversV3_Agent,
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
  else:
    raise NotImplementedError

  sim.init_game(**GAME_MAP)
  sim.set_autonomous_agent(*AGENTS)

  # generate data
  ############################################################################
  DATA_DIR = os.path.join(os.path.dirname(__file__), "data/")
  TRAIN_DIR = os.path.join(DATA_DIR, SAVE_PREFIX + '_train')
  # load train set
  ##################################################
  file_names = glob.glob(os.path.join(TRAIN_DIR, '*.txt'))
  random.shuffle(file_names)

  num_train = min(num_training_data, len(file_names))
  logging.info(num_train)

  train_files = file_names[:num_train]

  train_data.load_from_files(train_files)
  traj_labeled_ver = train_data.get_as_row_lists(no_latent_label=False,
                                                 include_terminal=False)

  list_traj_sa = []
  for traj in traj_labeled_ver:
    traj_sa = []
    for s, a, _ in traj:
      traj_sa.append((s, a))
    list_traj_sa.append(traj_sa)

  logging.info(len(list_traj_sa))

  # learn policy and transition
  ##################################################
  logging.info("pi, abs:  %f, %f" % (pi_prior, abs_prior))

  joint_action_num = tuple([MDP_AGENT.num_actions] * len(AGENTS))

  logging.info("#########")
  logging.info("Bayes Abstraction (%d)" % (num_train, ))
  logging.info("#########")

  # learning models
  btil_models = Bayes_Abstraction(list_traj_sa,
                                  MDP_TASK.num_states,
                                  joint_action_num,
                                  epsilon=0.1,
                                  max_iteration=num_iteration,
                                  num_abstates=num_abstates,
                                  lr=1,
                                  decay=0)
  btil_models.set_prior(pi_prior, abs_prior)

  btil_models.initialize_param()
  btil_models.do_inference(batch_size)

  # save models
  save_dir = DATA_DIR + "learned_models/"
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)

  save_pre = SAVE_PREFIX + "_bayes_abs_"
  policy_file_name = save_pre
  policy_file_name += f"{num_train}_{num_abstates}"
  policy_file_name = os.path.join(save_dir, policy_file_name)
  for idx in range(len(btil_models.list_np_policy)):
    np.save(policy_file_name + f"_pi_a{idx + 1}",
            btil_models.list_np_policy[idx])

  abs_file_name = save_pre
  abs_file_name += f"{num_train}_{num_abstates}"
  abs_file_name = os.path.join(save_dir, abs_file_name)
  np.save(abs_file_name + "_abs", btil_models.np_prob_abstate)


if __name__ == "__main__":
  logging.basicConfig(
      level=logging.INFO,
      format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
      handlers=[logging.StreamHandler()],
      force=True)
  main()
