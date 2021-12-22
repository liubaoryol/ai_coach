import numpy as np
import gym
from gym import spaces
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from imitation.algorithms.adversarial import gail
from imitation.algorithms import bc
from imitation.data.types import Trajectory
from imitation.util import logger
from imitation.data import rollout


class GymEnvFromMDP(gym.Env):
  # uncomment below line if you need to render the environment
  # metadata = {'render.modes': ['console']}

  def __init__(self, num_states, num_actions, cb_transition, cb_is_terminal,
               init_state):
    super().__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(num_actions)
    # Example for using image as input (channel-first; channel-last also works):
    self.observation_space = spaces.Discrete(num_states)

    # new members for custom env
    self.num_states = num_states
    self.num_actions = num_actions
    self.cb_transition = cb_transition
    self.cb_is_terminal = cb_is_terminal
    self.init_state = init_state
    self.cur_state = init_state

  def step(self, action):
    self.cur_state = self.cb_transition(self.cur_state, action)
    # reward = self.mdp.reward(self.cur_state, action)
    reward = -1
    done = self.cb_is_terminal(self.cur_state)
    info = {}

    return self.cur_state, reward, done, info

  def reset(self):
    self.cur_state = self.init_state

    return self.cur_state  # reward, done, info can't be included

  # implement render function if need to be
  # def render(self, mode='human'):
  #   ...

  # implement close function if need to be
  # def close(self):
  #   ...


def get_action_distribution(sb3_policy: ActorCriticPolicy, sidx: int):
  stensor, _ = sb3_policy.obs_to_tensor(sidx)
  act_dist = sb3_policy.get_distribution(stensor)
  probs = act_dist.distribution.probs

  return probs.cpu().detach().numpy()


def gail_on_agent(num_states, num_actions, cb_transition, cb_is_terminal,
                  init_state, sa_trajectories, terminal_value, logpath):
  if len(sa_trajectories) == 0:
    return np.zeros((num_states, num_actions)) / num_actions

  # create vec env
  gymenv = GymEnvFromMDP(num_states, num_actions, cb_transition, cb_is_terminal,
                         init_state)
  venv = DummyVecEnv([lambda: gymenv])

  # transition
  list_trajectories = []
  for traj in sa_trajectories:
    list_state, list_action = list(zip(*traj))
    list_trajectories.append(
        Trajectory(obs=np.array(list_state),
                   acts=np.array(list_action[:-1]),
                   terminal=list_action[-1] == terminal_value,
                   infos=None))

  transitions = rollout.flatten_trajectories(list_trajectories)

  print(f"All Tensorboards and logging are being written inside {logpath}/.")
  gail_logger = logger.configure(logpath, format_strs=("log", "csv"))

  gail_trainer = gail.GAIL(venv=venv,
                           demonstrations=transitions,
                           demo_batch_size=32,
                           gen_algo=sb3.PPO("MlpPolicy",
                                            venv,
                                            verbose=0,
                                            n_steps=64),
                           custom_logger=gail_logger,
                           normalize_obs=False,
                           allow_variable_horizon=True)

  gail_trainer.train(total_timesteps=32000)
  np_policy = np.zeros((num_states, num_actions))
  for sidx in range(num_states):
    np_policy[sidx, :] = get_action_distribution(gail_trainer.policy, sidx)

  return np_policy


def behavior_cloning_sb3(sa_trajectories, num_states, num_actions):
  if len(sa_trajectories) == 0:
    return np.zeros((num_states, num_actions)) / num_actions

  observation_space = spaces.Discrete(num_states)
  action_space = spaces.Discrete(num_actions)

  # transition
  list_trajectories = []
  for traj in sa_trajectories:
    list_state, list_action = list(zip(*traj))
    list_trajectories.append(
        Trajectory(obs=np.array(list_state),
                   acts=np.array(list_action[:-1]),
                   terminal=list_action[-1] == -1,
                   infos=None))

  transitions = rollout.flatten_trajectories(list_trajectories)
  bc_trainer = bc.BC(observation_space=observation_space,
                     action_space=action_space,
                     demonstrations=transitions)
  bc_trainer.train(n_epochs=1)

  np_policy = np.zeros((num_states, num_actions))
  for sidx in range(num_states):
    np_policy[sidx, :] = get_action_distribution(bc_trainer.policy, sidx)

  return np_policy
