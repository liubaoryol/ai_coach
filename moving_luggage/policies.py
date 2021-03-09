from stable_baselines import DQN
from policy.impl_utils import conv_to_np_env
from moving_luggage.constants import (
    KEY_BAGS, KEY_AGENTS, AgentActions, LATENT_LIGHT_BAGS)

DQN_HEAVY = DQN.load("deepq_movingluggage_heavy")
DQN_LIGHT = DQN.load("deepq_movingluggage_light")

def get_dqn_policy(env, agent_idx, mental_model):
    obs = conv_to_np_env(env[KEY_BAGS], env[KEY_AGENTS][0], env[KEY_AGENTS][1])
    action = None
    if mental_model == LATENT_LIGHT_BAGS:
        action, _ = DQN_LIGHT.predict(obs, deterministic=True)
    else:
        action, _ = DQN_HEAVY.predict(obs, deterministic=True)

    action1 = action % len(AgentActions)
    action2 = int(action / len(AgentActions))

    if agent_idx == 0:
        return AgentActions(action1)
    else:
        return AgentActions(action2)
