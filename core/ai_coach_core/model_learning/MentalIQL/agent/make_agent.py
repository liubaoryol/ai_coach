from typing import Type
import gym
from gym.spaces import Discrete, Box
from aicoach_baselines.option_gail.utils.config import Config


def make_miql_agent(config: Config, env: gym.Env):
  q_net_base = SimpleOptionQNetwork
