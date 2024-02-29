from aic_domain.box_push.utils import BoxPushTrajectories
from aic_domain.box_push_v2.agent import (BoxPushAIAgent_PO_Team,
                                          BoxPushAIAgent_PO_Indv)
from aic_domain.box_push_v2.simulator import BoxPushSimulatorV2
from aic_domain.box_push_v2.maps import (MAP_MOVERS, MAP_CLEANUP_V2,
                                         MAP_CLEANUP_V3)
from aic_domain.box_push_v2.policy import Policy_Movers, Policy_Cleanup
from aic_domain.box_push_v2.mdp import (MDP_Movers_Agent, MDP_Movers_Task)
from aic_domain.box_push_v2.mdp import (MDP_Cleanup_Agent, MDP_Cleanup_Task)
from aic_domain.rescue.agent import AIAgent_Rescue_PartialObs
from aic_domain.rescue.simulator import RescueSimulator
from aic_domain.rescue.maps import MAP_RESCUE
from aic_domain.rescue.policy import Policy_Rescue
from aic_domain.rescue.mdp import MDP_Rescue_Agent, MDP_Rescue_Task
from aic_domain.rescue.utils import RescueTrajectories
from aic_domain.rescue_v2.agent import AIAgent_Rescue_PartialObs as \
                                                    AIAgent_Rescue_PartialObs_V2
from aic_domain.rescue_v2.simulator import RescueSimulatorV2
from aic_domain.rescue_v2.maps import MAP_RESCUE as MAP_RESCUE_V2
from aic_domain.rescue_v2.policy import Policy_Rescue as Policy_Rescue_V2
from aic_domain.rescue_v2.mdp import MDP_Rescue_Agent as MDP_Rescue_Agent_V2
from aic_domain.rescue_v2.mdp import MDP_Rescue_Task as MDP_Rescue_Task_V2
from aic_domain.rescue_v2.utils import RescueV2Trajectories
from aic_domain.agent import BTILCachedPolicy
from aic_domain.box_push_v2.agent import BoxPushAIAgent_BTIL
from aic_domain.rescue.agent import AIAgent_Rescue_BTIL


def _create_btil_agents(dict_btil_args, mdp_task, latent_space,
                        agent_class_base):
  np_policy1 = dict_btil_args.get('np_policy1', None)
  np_policy2 = dict_btil_args.get('np_policy2', None)
  np_tx1 = dict_btil_args.get('np_tx1', None)
  np_tx2 = dict_btil_args.get('np_tx2', None)
  np_bx1 = dict_btil_args.get('np_bx1', None)
  np_bx2 = dict_btil_args.get('np_bx2', None)
  mask = dict_btil_args.get('mask', None)

  list_np_policy = [np_policy1, np_policy2]
  list_np_tx = [np_tx1, np_tx2]
  list_np_bx = [np_bx1, np_bx2]

  agents = []
  for idx, np_policy in enumerate(list_np_policy):
    policy = BTILCachedPolicy(np_policy, mdp_task, idx, latent_space)
    agent = agent_class_base(list_np_tx[idx],
                             mask,
                             policy,
                             idx,
                             np_bx=list_np_bx[idx])
    agents.append(agent)

  return agents


def load_movers(is_btil_agent=False, dict_btil_args=None):
  sim = BoxPushSimulatorV2(0)
  GAME_MAP = MAP_MOVERS
  save_prefix = GAME_MAP["name"]
  mdp_task = MDP_Movers_Task(**GAME_MAP)
  mdp_agent = MDP_Movers_Agent(**GAME_MAP)

  if is_btil_agent:
    agents = _create_btil_agents(dict_btil_args, mdp_task,
                                 mdp_agent.latent_space, BoxPushAIAgent_BTIL)
  else:
    TEMPERATURE = 0.3
    policy_1 = Policy_Movers(mdp_task, mdp_agent, TEMPERATURE, 0)
    policy_2 = Policy_Movers(mdp_task, mdp_agent, TEMPERATURE, 1)
    init_states = ([0] * len(GAME_MAP["boxes"]), GAME_MAP["a1_init"],
                   GAME_MAP["a2_init"])
    agent_1 = BoxPushAIAgent_PO_Team(init_states,
                                     policy_1,
                                     agent_idx=sim.AGENT1)
    agent_2 = BoxPushAIAgent_PO_Team(init_states,
                                     policy_2,
                                     agent_idx=sim.AGENT2)

    agents = [agent_1, agent_2]

  train_data = BoxPushTrajectories(mdp_task, mdp_agent)

  return sim, agents, save_prefix, train_data, GAME_MAP


def load_cleanup_v2(is_btil_agent=False, dict_btil_args=None):
  sim = BoxPushSimulatorV2(0)
  GAME_MAP = MAP_CLEANUP_V2
  save_prefix = GAME_MAP["name"]
  mdp_task = MDP_Cleanup_Task(**GAME_MAP)
  mdp_agent = MDP_Cleanup_Agent(**GAME_MAP)

  if is_btil_agent:
    agents = _create_btil_agents(dict_btil_args, mdp_task,
                                 mdp_agent.latent_space, BoxPushAIAgent_BTIL)
  else:
    TEMPERATURE = 0.3
    policy_1 = Policy_Cleanup(mdp_task, mdp_agent, TEMPERATURE, 0)
    policy_2 = Policy_Cleanup(mdp_task, mdp_agent, TEMPERATURE, 1)
    init_states = ([0] * len(GAME_MAP["boxes"]), GAME_MAP["a1_init"],
                   GAME_MAP["a2_init"])
    agent_1 = BoxPushAIAgent_PO_Indv(init_states,
                                     policy_1,
                                     agent_idx=sim.AGENT1)
    agent_2 = BoxPushAIAgent_PO_Indv(init_states,
                                     policy_2,
                                     agent_idx=sim.AGENT2)
    agents = [agent_1, agent_2]

  train_data = BoxPushTrajectories(mdp_task, mdp_agent)

  return sim, agents, save_prefix, train_data, GAME_MAP


def load_cleanup_v3(is_btil_agent=False, dict_btil_args=None):
  sim = BoxPushSimulatorV2(0)
  GAME_MAP = MAP_CLEANUP_V3
  save_prefix = GAME_MAP["name"]
  mdp_task = MDP_Cleanup_Task(**GAME_MAP)
  mdp_agent = MDP_Cleanup_Agent(**GAME_MAP)

  if is_btil_agent:
    agents = _create_btil_agents(dict_btil_args, mdp_task,
                                 mdp_agent.latent_space, BoxPushAIAgent_BTIL)
  else:
    TEMPERATURE = 0.3
    policy_1 = Policy_Cleanup(mdp_task, mdp_agent, TEMPERATURE, 0)
    policy_2 = Policy_Cleanup(mdp_task, mdp_agent, TEMPERATURE, 1)
    init_states = ([0] * len(GAME_MAP["boxes"]), GAME_MAP["a1_init"],
                   GAME_MAP["a2_init"])
    agent_1 = BoxPushAIAgent_PO_Indv(init_states,
                                     policy_1,
                                     agent_idx=sim.AGENT1)
    agent_2 = BoxPushAIAgent_PO_Indv(init_states,
                                     policy_2,
                                     agent_idx=sim.AGENT2)
    agents = [agent_1, agent_2]

  train_data = BoxPushTrajectories(mdp_task, mdp_agent)

  return sim, agents, save_prefix, train_data, GAME_MAP


def load_rescue_2(is_btil_agent=False, dict_btil_args=None):
  sim = RescueSimulator()
  sim.max_steps = 30

  GAME_MAP = MAP_RESCUE
  save_prefix = GAME_MAP["name"]
  mdp_task = MDP_Rescue_Task(**GAME_MAP)
  mdp_agent = MDP_Rescue_Agent(**GAME_MAP)

  if is_btil_agent:
    agents = _create_btil_agents(dict_btil_args, mdp_task,
                                 mdp_agent.latent_space, AIAgent_Rescue_BTIL)
  else:
    TEMPERATURE = 0.3
    policy_1 = Policy_Rescue(mdp_task, mdp_agent, TEMPERATURE, 0)
    policy_2 = Policy_Rescue(mdp_task, mdp_agent, TEMPERATURE, 1)

    init_states = ([1] * len(GAME_MAP["work_locations"]), GAME_MAP["a1_init"],
                   GAME_MAP["a2_init"])
    agent_1 = AIAgent_Rescue_PartialObs(init_states, 0, policy_1)
    agent_2 = AIAgent_Rescue_PartialObs(init_states, 1, policy_2)

    agents = [agent_1, agent_2]

  def conv_latent_to_idx(agent_idx, latent):
    return agents[agent_idx].conv_latent_to_idx(latent)

  train_data = RescueTrajectories(
      mdp_task, (mdp_agent.num_latents, mdp_agent.num_latents),
      conv_latent_to_idx)

  return sim, agents, save_prefix, train_data, GAME_MAP


def load_rescue_3(is_btil_agent=False, dict_btil_args=None):
  assert is_btil_agent is False, "BTIL agent is not implemented for rescue_3"

  sim = RescueSimulatorV2()
  sim.max_steps = 15

  TEMPERATURE = 0.3
  GAME_MAP = MAP_RESCUE_V2
  save_prefix = GAME_MAP["name"]
  mdp_task = MDP_Rescue_Task_V2(**GAME_MAP)
  mdp_agent = MDP_Rescue_Agent_V2(**GAME_MAP)
  policy_1 = Policy_Rescue_V2(mdp_task, mdp_agent, TEMPERATURE, 0)
  policy_2 = Policy_Rescue_V2(mdp_task, mdp_agent, TEMPERATURE, 1)
  policy_3 = Policy_Rescue_V2(mdp_task, mdp_agent, TEMPERATURE, 2)

  init_states = ([1] * len(GAME_MAP["work_locations"]), GAME_MAP["a1_init"],
                 GAME_MAP["a2_init"], GAME_MAP["a3_init"])
  agent_1 = AIAgent_Rescue_PartialObs_V2(init_states, 0, policy_1)
  agent_2 = AIAgent_Rescue_PartialObs_V2(init_states, 1, policy_2)
  agent_3 = AIAgent_Rescue_PartialObs_V2(init_states, 2, policy_3)

  agents = [agent_1, agent_2, agent_3]

  def conv_latent_to_idx(agent_idx, latent):
    return agents[agent_idx].conv_latent_to_idx(latent)

  train_data = RescueV2Trajectories(
      mdp_task,
      (mdp_agent.num_latents, mdp_agent.num_latents, mdp_agent.num_latents),
      conv_latent_to_idx)

  return sim, agents, save_prefix, train_data, GAME_MAP
