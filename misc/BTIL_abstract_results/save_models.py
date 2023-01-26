import os
import glob
import click
import logging
import random
import numpy as np
from ai_coach_core.model_learning.BTIL.btil_abstraction import BTIL_Abstraction


# yapf: disable
@click.command()
@click.option("--domain", type=str, default="movers", help="movers / cleanup_v3 / rescue_2 /rescue_3")  # noqa: E501
@click.option("--num-training-data", type=int, default=1000, help="")
@click.option("--supervision", type=float, default=0.2, help="")
@click.option("--gem-prior", type=float, default=3, help="")
@click.option("--tx-prior", type=float, default=3, help="")
@click.option("--pi-prior", type=float, default=3, help="")
@click.option("--abs-prior", type=float, default=3, help="")
@click.option("--num-x", type=int, default=4, help="")
@click.option("--num-abstract", type=int, default=30, help="")
@click.option("--num-iteration", type=int, default=1000, help="")
@click.option("--batch-size", type=int, default=100, help="")
@click.option("--load-param", type=bool, default=False, help="")
@click.option("--tx-dependency", type=str, default="FTTT",
              help="sequence of T or F indicating dependency on cur_state, actions, and next_state")  # noqa: E501
# yapf: enable
def main(domain, num_training_data, gem_prior, tx_prior, pi_prior, abs_prior,
         num_x, num_abstract, tx_dependency, num_iteration, batch_size,
         load_param, supervision):
  logging.info("domain: %s" % (domain, ))
  logging.info("num training data: %s" % (num_training_data, ))
  logging.info("GEM prior: %s" % (gem_prior, ))
  logging.info("Tx prior: %s" % (tx_prior, ))
  logging.info("pi prior: %s" % (pi_prior, ))
  logging.info("abs prior: %s" % (abs_prior, ))
  logging.info("num x: %s" % (num_x, ))
  logging.info("num abstract: %s" % (num_abstract, ))
  logging.info("num iteration: %s" % (num_iteration, ))
  logging.info("batch size: %s" % (batch_size, ))
  logging.info("load param: %s" % (load_param, ))
  logging.info("Tx dependency: %s" % (tx_dependency, ))

  # define the domain where trajectories were generated
  ##################################################
  if domain == "movers":
    from ai_coach_domain.box_push.utils import BoxPushTrajectories
    from ai_coach_domain.box_push_v2.agent import BoxPushAIAgent_PO_Team
    from ai_coach_domain.box_push_v2.maps import MAP_MOVERS
    from ai_coach_domain.box_push_v3.simulator import BoxPushSimulatorV3
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
    init_states = ([0] * len(GAME_MAP["boxes"]), GAME_MAP["a1_init"],
                   GAME_MAP["a2_init"])
    AGENT_1 = BoxPushAIAgent_PO_Team(init_states,
                                     POLICY_1,
                                     agent_idx=sim.AGENT1)
    AGENT_2 = BoxPushAIAgent_PO_Team(init_states,
                                     POLICY_2,
                                     agent_idx=sim.AGENT2)
    AGENTS = [AGENT_1, AGENT_2]
    train_data = BoxPushTrajectories(MDP_TASK, MDP_AGENT)
  elif domain == "cleanup_v3":
    from ai_coach_domain.box_push.utils import BoxPushTrajectories
    from ai_coach_domain.box_push_v2.agent import BoxPushAIAgent_PO_Indv
    from ai_coach_domain.box_push_v2.maps import MAP_CLEANUP_V3
    from ai_coach_domain.box_push_v3.simulator import BoxPushSimulatorV3
    from ai_coach_domain.box_push_v3.policy import Policy_CleanupV3
    from ai_coach_domain.box_push_v3.mdp import (MDP_CleanupV3_Agent,
                                                 MDP_CleanupV3_Task)
    sim = BoxPushSimulatorV3(False)
    TEMPERATURE = 0.3
    GAME_MAP = MAP_CLEANUP_V3
    SAVE_PREFIX = GAME_MAP["name"]
    MDP_TASK = MDP_CleanupV3_Task(**GAME_MAP)
    MDP_AGENT = MDP_CleanupV3_Agent(**GAME_MAP)
    POLICY_1 = Policy_CleanupV3(MDP_TASK, MDP_AGENT, TEMPERATURE, 0)
    POLICY_2 = Policy_CleanupV3(MDP_TASK, MDP_AGENT, TEMPERATURE, 1)
    init_states = ([0] * len(GAME_MAP["boxes"]), GAME_MAP["a1_init"],
                   GAME_MAP["a2_init"])
    AGENT_1 = BoxPushAIAgent_PO_Indv(init_states,
                                     POLICY_1,
                                     agent_idx=sim.AGENT1)
    AGENT_2 = BoxPushAIAgent_PO_Indv(init_states,
                                     POLICY_2,
                                     agent_idx=sim.AGENT2)
    AGENTS = [AGENT_1, AGENT_2]
    train_data = BoxPushTrajectories(MDP_TASK, MDP_AGENT)
  elif domain == "rescue_2":
    from ai_coach_domain.rescue.agent import AIAgent_Rescue_PartialObs
    from ai_coach_domain.rescue.simulator import RescueSimulator
    from ai_coach_domain.rescue.maps import MAP_RESCUE
    from ai_coach_domain.rescue.policy import Policy_Rescue
    from ai_coach_domain.rescue.mdp import MDP_Rescue_Agent, MDP_Rescue_Task
    from ai_coach_domain.rescue.utils import RescueTrajectories
    sim = RescueSimulator()
    sim.max_steps = 30
    TEMPERATURE = 0.3

    GAME_MAP = MAP_RESCUE
    SAVE_PREFIX = GAME_MAP["name"]
    MDP_TASK = MDP_Rescue_Task(**GAME_MAP)
    MDP_AGENT = MDP_Rescue_Agent(**GAME_MAP)
    POLICY_1 = Policy_Rescue(MDP_TASK, MDP_AGENT, TEMPERATURE, 0)
    POLICY_2 = Policy_Rescue(MDP_TASK, MDP_AGENT, TEMPERATURE, 1)

    init_states = ([1] * len(GAME_MAP["work_locations"]), GAME_MAP["a1_init"],
                   GAME_MAP["a2_init"])
    AGENT_1 = AIAgent_Rescue_PartialObs(init_states, 0, POLICY_1)
    AGENT_2 = AIAgent_Rescue_PartialObs(init_states, 1, POLICY_2)

    AGENTS = [AGENT_1, AGENT_2]

    def conv_latent_to_idx(agent_idx, latent):
      if agent_idx == 0:
        return AGENT_1.conv_latent_to_idx(latent)
      else:
        return AGENT_2.conv_latent_to_idx(latent)

    train_data = RescueTrajectories(
        MDP_TASK, (MDP_AGENT.num_latents, MDP_AGENT.num_latents),
        conv_latent_to_idx)
  elif domain == "rescue_3":
    from ai_coach_domain.rescue_v2.agent import AIAgent_Rescue_PartialObs
    from ai_coach_domain.rescue_v2.simulator import RescueSimulatorV2
    from ai_coach_domain.rescue_v2.maps import MAP_RESCUE
    from ai_coach_domain.rescue_v2.policy import Policy_Rescue
    from ai_coach_domain.rescue_v2.mdp import MDP_Rescue_Agent, MDP_Rescue_Task
    from ai_coach_domain.rescue_v2.utils import RescueV2Trajectories
    sim = RescueSimulatorV2()
    sim.max_steps = 15

    TEMPERATURE = 0.3
    GAME_MAP = MAP_RESCUE
    SAVE_PREFIX = GAME_MAP["name"]
    MDP_TASK = MDP_Rescue_Task(**GAME_MAP)
    MDP_AGENT = MDP_Rescue_Agent(**GAME_MAP)
    POLICY_1 = Policy_Rescue(MDP_TASK, MDP_AGENT, TEMPERATURE, 0)
    POLICY_2 = Policy_Rescue(MDP_TASK, MDP_AGENT, TEMPERATURE, 1)
    POLICY_3 = Policy_Rescue(MDP_TASK, MDP_AGENT, TEMPERATURE, 2)

    init_states = ([1] * len(GAME_MAP["work_locations"]), GAME_MAP["a1_init"],
                   GAME_MAP["a2_init"], GAME_MAP["a3_init"])
    AGENT_1 = AIAgent_Rescue_PartialObs(init_states, 0, POLICY_1)
    AGENT_2 = AIAgent_Rescue_PartialObs(init_states, 1, POLICY_2)
    AGENT_3 = AIAgent_Rescue_PartialObs(init_states, 2, POLICY_3)

    AGENTS = [AGENT_1, AGENT_2, AGENT_3]

    def conv_latent_to_idx(agent_idx, latent):
      return AGENTS[agent_idx].conv_latent_to_idx(latent)

    train_data = RescueV2Trajectories(
        MDP_TASK,
        (MDP_AGENT.num_latents, MDP_AGENT.num_latents, MDP_AGENT.num_latents),
        conv_latent_to_idx)
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
  traj_labeled_ver = train_data.get_as_row_lists(no_latent_label=False,
                                                 include_terminal=False)
  traj_unlabel_ver = train_data.get_as_row_lists(no_latent_label=True,
                                                 include_terminal=False)

  logging.info(len(traj_labeled_ver))
  labeled_idx = int(num_train * supervision)

  # learn policy and transition
  ##################################################
  logging.info("gem prior, tx prior, pi prior, abs prior: %f, %f, %f, %f" %
               (gem_prior, tx_prior, pi_prior, abs_prior))

  joint_action_num = tuple([MDP_AGENT.num_actions] * len(AGENTS))

  logging.info("#########")
  logging.info("BTIL (Labeled: %d, Unlabeled: %d)" %
               (labeled_idx, num_train - labeled_idx))
  logging.info("#########")

  # to backup params
  temp_dir = DATA_DIR + "learned_models/temp/"
  if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
  save_prefix = SAVE_PREFIX + "_btil_abs_"
  save_prefix += tx_dependency + "_"
  save_prefix += ("%d_" % (num_train, ))
  save_prefix += ("%d_%d" % (num_x, num_abstract))
  file_prefix = os.path.join(temp_dir, save_prefix)

  # learning models
  btil_models = BTIL_Abstraction(traj_labeled_ver[:labeled_idx] +
                                 traj_unlabel_ver[labeled_idx:],
                                 MDP_TASK.num_states,
                                 tuple([num_x] * len(AGENTS)),
                                 joint_action_num,
                                 trans_x_dependency=tuple_tx_dependency,
                                 max_iteration=num_iteration,
                                 epsilon_g=0.1,
                                 epsilon_l=0.05,
                                 lr=0.1,
                                 decay=0.01,
                                 num_abstates=num_abstract,
                                 save_file_prefix=file_prefix,
                                 no_gem=True)
  btil_models.set_prior(gem_prior, tx_prior, pi_prior, abs_prior)

  if load_param:
    btil_models.load_params()
  else:
    btil_models.initialize_param()

  btil_models.do_inference(batch_size=batch_size)

  # save models
  save_prefix = os.path.join(DATA_DIR + "learned_models/", save_prefix)

  for idx in range(btil_models.num_agents):
    np.save(save_prefix + "_pi" + f"_a{idx + 1}",
            btil_models.list_np_policy[idx])
    np.save(save_prefix + "_tx" + f"_a{idx + 1}",
            btil_models.list_Tx[idx].np_Tx)
    np.save(save_prefix + "_bx" + f"_a{idx + 1}", btil_models.list_bx[idx])

  np.save(save_prefix + "_abs", btil_models.np_prob_abstate)


if __name__ == "__main__":
  logging.basicConfig(
      level=logging.INFO,
      format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
      handlers=[logging.StreamHandler()],
      force=True)
  main()
