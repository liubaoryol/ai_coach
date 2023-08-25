import os
import numpy as np
import click
from aic_core.utils.mdp_utils import StateSpace


# yapf: disable
@click.command()
@click.option("--domain", type=str, default="movers", help="movers / cleanup_v3 / rescue_2 /rescue_3")  # noqa: E501
@click.option("--num-runs", type=int, default=100, help="")
@click.option("--learner", type=str, default="coach", help="coach / hdp / bc")
# yapf: enable
def main(domain, num_runs, learner):

  DATA_DIR = os.path.join(os.path.dirname(__file__), "data/")
  # define the domain where trajectories were generated
  ##################################################
  if domain == "movers":
    from ai_coach_domain.box_push_v2.maps import MAP_MOVERS
    from ai_coach_domain.agent.cached_agent import (BTILCachedPolicy,
                                                    NoMindCachedPolicy)
    from ai_coach_domain.box_push_v2.agent import (BoxPushAIAgent_BTIL,
                                                   AIAgent_NoMind)
    from ai_coach_domain.box_push_v3.simulator import BoxPushSimulatorV3
    from ai_coach_domain.box_push_v3.mdp import (MDP_MoversV3_Agent,
                                                 MDP_MoversV3_Task)

    sim = BoxPushSimulatorV3(False)
    sim.max_steps = 200
    GAME_MAP = MAP_MOVERS
    MDP_TASK = MDP_MoversV3_Task(**GAME_MAP)
    MDP_AGENT = MDP_MoversV3_Agent(**GAME_MAP)
    model_dir = DATA_DIR + "/learned_models/"  # noqa: E501

    num_train = 500
    num_opt_train = 100
    if learner == "coach" or learner == "hdp":
      num_x = 4
      file_name = (f"{domain}_btil_hdp_FTTT_{num_train}_{num_x}")
      NP_POLICY_A1 = file_name + "_pi_a1.npy"
      NP_POLICY_A2 = file_name + "_pi_a2.npy"
      NP_TX_A1 = file_name + "_tx_a1.npy"
      NP_TX_A2 = file_name + "_tx_a2.npy"
      NP_BX_A1 = file_name + "_bx_a1.npy"
      NP_BX_A2 = file_name + "_bx_a2.npy"

      np_policy_1 = np.load(model_dir + NP_POLICY_A1)
      test_policy_1 = BTILCachedPolicy(np_policy_1, MDP_TASK, 0,
                                       StateSpace(np.arange(num_x)))
      np_policy_2 = np.load(model_dir + NP_POLICY_A2)
      test_policy_2 = BTILCachedPolicy(np_policy_2, MDP_TASK, 1,
                                       StateSpace(np.arange(num_x)))
      np_tx_1 = np.load(model_dir + NP_TX_A1)
      np_tx_2 = np.load(model_dir + NP_TX_A2)

      np_bx_1 = np.load(model_dir + NP_BX_A1)
      np_bx_2 = np.load(model_dir + NP_BX_A2)

      if learner == "coach":
        NP_COACH_POLICY = f"movers_bc_hdp_{num_opt_train}_pi_x.npy"
        np_coach = np.load(model_dir + NP_COACH_POLICY)
      else:
        np_coach = None

      mask = (False, True, True, True)

      AGENT_1 = BoxPushAIAgent_BTIL(np_tx_1,
                                    mask,
                                    test_policy_1,
                                    0,
                                    np_bx=np_bx_1,
                                    np_coach=np_coach)
      AGENT_2 = BoxPushAIAgent_BTIL(np_tx_2,
                                    mask,
                                    test_policy_2,
                                    1,
                                    np_bx=np_bx_2,
                                    np_coach=np_coach)
    elif learner == "bc":
      file_name = (f"{domain}_bc_{num_opt_train}")
      NP_POLICY_A1 = file_name + "_pi_a1.npy"
      NP_POLICY_A2 = file_name + "_pi_a2.npy"

      np_policy_1 = np.load(model_dir + NP_POLICY_A1)
      np_policy_1 = np_policy_1 / np.sum(np_policy_1, axis=1)[:, None]
      test_policy_1 = NoMindCachedPolicy(np_policy_1, MDP_TASK, 0)
      np_policy_2 = np.load(model_dir + NP_POLICY_A2)
      np_policy_2 = np_policy_2 / np.sum(np_policy_2, axis=1)[:, None]
      test_policy_2 = NoMindCachedPolicy(np_policy_2, MDP_TASK, 1)

      AGENT_1 = AIAgent_NoMind(test_policy_1, 0)
      AGENT_2 = AIAgent_NoMind(test_policy_2, 1)

    AGENTS = [AGENT_1, AGENT_2]

  sim.init_game(**GAME_MAP)
  sim.set_autonomous_agent(*AGENTS)

  list_score = []
  for _ in range(num_runs):
    sim.reset_game()
    while not sim.is_finished():
      map_agent_2_action = sim.get_joint_action()
      sim.take_a_step(map_agent_2_action)

    list_score.append(sim.get_score())
  print(np.mean(list_score))


if __name__ == "__main__":
  main()