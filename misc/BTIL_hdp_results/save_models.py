import os
import glob
import click
import logging
import random
import numpy as np
from aic_ml.BTIL.btil_svi import BTIL_SVI


# yapf: disable
@click.command()
@click.option("--domain", type=str, default="movers", help="movers / cleanup_v3 / rescue_2 /rescue_3")  # noqa: E501
@click.option("--num-training-data", type=int, default=500, help="")
@click.option("--gem-prior", type=float, default=3, help="")
@click.option("--tx-prior", type=float, default=3, help="")
@click.option("--pi-prior", type=float, default=3, help="")
@click.option("--no-gem", type=bool, default=False, help="")
@click.option("--num-x", type=int, default=4, help="")
@click.option("--num-iteration", type=int, default=300, help="")
@click.option("--batch-size", type=int, default=500, help="")
@click.option("--lr", type=float, default=1.0, help="")
@click.option("--decay", type=float, default=0.01, help="")
@click.option("--lr-beta", type=float, default=0.001, help="")
@click.option("--tx-dependency", type=str, default="FTTT",
              help="sequence of T or F indicating dependency on cur_state, actions, and next_state")  # noqa: E501
# yapf: enable
def main(domain, num_training_data, gem_prior, tx_prior, pi_prior, num_x,
         no_gem, batch_size, tx_dependency, num_iteration, lr, lr_beta, decay):
  logging.info("domain: %s" % (domain, ))
  logging.info("num training data: %s" % (num_training_data, ))
  logging.info("GEM gamma: %s" % (gem_prior, ))
  logging.info("Tx alpha: %s" % (tx_prior, ))
  logging.info("pi rho: %s" % (pi_prior, ))
  logging.info("num x: %s" % (num_x, ))
  logging.info("batch size: %s" % (batch_size, ))
  logging.info("no gem : %s" % (no_gem, ))
  logging.info("Tx dependency: %s" % (tx_dependency, ))
  logging.info("lr, lr_beta, decay: (%s, %s, %s)" % (lr, lr_beta, decay))

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

  tuple_tx_dependency = []
  for cha in tx_dependency:
    if cha == "T":
      tuple_tx_dependency.append(True)
    else:
      tuple_tx_dependency.append(False)

  tuple_tx_dependency = tuple(tuple_tx_dependency)

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
  traj_unlabel_ver = train_data.get_as_row_lists(no_latent_label=True,
                                                 include_terminal=False)

  logging.info(len(traj_unlabel_ver))

  # learn policy and transition
  ##################################################
  logging.info("gamma, tx,  pi: %f, %f, %f" % (gem_prior, tx_prior, pi_prior))

  joint_action_num = tuple([MDP_AGENT.num_actions] * len(AGENTS))

  logging.info("#########")
  logging.info("BTIL (Unlabeled: %d)" % (num_train, ))
  logging.info("#########")

  # learning models
  btil_models = BTIL_SVI(traj_unlabel_ver,
                         MDP_TASK.num_states,
                         tuple([num_x] * len(AGENTS)),
                         joint_action_num,
                         trans_x_dependency=tuple_tx_dependency,
                         epsilon=0.1,
                         max_iteration=num_iteration,
                         lr=lr,
                         lr_beta=lr_beta,
                         decay=decay,
                         no_gem=no_gem)
  btil_models.set_prior(gem_prior, tx_prior, pi_prior)

  btil_models.initialize_param()
  btil_models.do_inference(batch_size)

  # save models
  save_dir = DATA_DIR + "learned_models/"
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)

  save_prefix = SAVE_PREFIX + "_btil_hdp_"
  save_prefix += tx_dependency + "_"
  save_prefix += ("%d_" % (num_train, ))
  save_prefix += ("%d" % (num_x, ))

  # save models
  save_prefix = os.path.join(DATA_DIR + "learned_models/", save_prefix)

  for idx in range(btil_models.num_agents):
    np.save(save_prefix + "_pi" + f"_a{idx + 1}",
            btil_models.list_np_policy[idx])
    np.save(save_prefix + "_tx" + f"_a{idx + 1}",
            btil_models.list_Tx[idx].np_Tx)
    np.save(save_prefix + "_bx" + f"_a{idx + 1}", btil_models.list_bx[idx])


if __name__ == "__main__":
  logging.basicConfig(
      level=logging.INFO,
      format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
      handlers=[logging.StreamHandler()],
      force=True)
  main()
