from itertools import count
import click
import torch
import numpy as np
import os
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

from aic_ml.IQLearn.utils.utils import make_env, evaluate
from aic_ml.IQLearn.agent import make_agent
from aic_ml.IQLearn.agent.softq_models import (SimpleQNetwork)


@click.command()
@click.option("--domain", type=str, default="movers", help="")
@click.option("--opt", type=bool, default=True, help="")
def main(domain, opt):

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

  # gym env
  ##################################################
  sim = BoxPushSimulatorV3(False)
  sim.init_game(**GAME_MAP)
  possible_init_states = []
  init_bstate = [0] * len(GAME_MAP["boxes"])
  for pos1 in sim.possible_positions:
    for pos2 in sim.possible_positions:
      init_states = (init_bstate, pos1, pos2)
      init_sidx = MDP_TASK.conv_sim_states_to_mdp_sidx(init_states)
      possible_init_states.append(init_sidx)

  env_kwargs = {
      'mdp': MDP_TASK,
      'possible_init_states': possible_init_states,
      'use_central_action': True
  }
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  env = make_env('envfrommdp-v0', True, env_make_kwargs=env_kwargs)

  q_net_base = SimpleQNetwork
  agent = make_agent(env, 64, device, q_net_base)

  output_dir = os.path.join(os.path.dirname(__file__), "output/")
  output_file = output_dir + "softq_iq_envfrommdp-v0_opt100_best"
  print(f'Loading policy from: {output_file}')

  agent.load(output_file)

  eval_returns, eval_timesteps = evaluate(agent, env, num_episodes=100)
  print(f'Avg. eval returns: {np.mean(eval_returns)},' +
        f' timesteps: {np.mean(eval_timesteps)}')


if __name__ == '__main__':
  main()
