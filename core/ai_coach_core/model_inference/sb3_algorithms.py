import tempfile
import pathlib
import numpy as np
from gym import spaces
import stable_baselines3 as sb3
# import gym
# from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from imitation.algorithms.adversarial import gail
from imitation.algorithms import bc
from imitation.data.types import Trajectory
from imitation.util import logger
from imitation.data import rollout
from imitation.util.util import make_vec_env
import gym_aicoach  # noqa: F401


def get_sb3_policy(sb3_policy: ActorCriticPolicy, num_states: int,
                   num_actions: int):
  np_policy = np.zeros((num_states, num_actions))
  for sidx in range(num_states):
    stensor, _ = sb3_policy.obs_to_tensor(sidx)
    act_dist = sb3_policy.get_distribution(stensor)
    probs = act_dist.distribution.probs
    np_policy[sidx, :] = probs.cpu().detach().numpy()

  return np_policy


def gail_sb3(num_states,
             num_actions,
             cb_transition,
             cb_is_terminal,
             cb_legal_actions,
             init_state,
             sa_trajectories,
             terminal_value,
             logpath=None,
             demo_batch_size=32,
             ppo_batch_size=64,
             n_steps=64,
             total_timesteps=32000,
             callback_policy=None):
  if len(sa_trajectories) == 0:
    return np.zeros((num_states, num_actions)) / num_actions

  # create vec env
  # gymenv = gym.make('envfrommdp-v0',
  #                   num_states=num_states,
  #                   num_actions=num_actions,
  #                   cb_transition=cb_transition,
  #                   cb_is_terminal=cb_is_terminal,
  #                   cb_legal_actions=cb_legal_actions,
  #                   init_state=init_state)
  # venv = DummyVecEnv([lambda: gymenv])

  env_kwargs = dict(num_states=num_states,
                    num_actions=num_actions,
                    cb_transition=cb_transition,
                    cb_is_terminal=cb_is_terminal,
                    cb_legal_actions=cb_legal_actions,
                    init_state=init_state)

  venv = make_vec_env('envfrommdp-v0', n_envs=1, env_make_kwargs=env_kwargs)

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

  if logpath is None:
    tempdir = tempfile.TemporaryDirectory(prefix="gail")
    tempdir_path = pathlib.Path(tempdir.name)
    gail_logger = logger.configure(tempdir_path, format_strs=())
  else:
    print(f"All Tensorboards and logging are being written inside {logpath}/.")
    gail_logger = logger.configure(logpath, format_strs=("log", "csv"))

  gail_trainer = gail.GAIL(venv=venv,
                           demonstrations=transitions,
                           demo_batch_size=demo_batch_size,
                           gen_algo=sb3.PPO("MlpPolicy",
                                            venv,
                                            verbose=0,
                                            batch_size=ppo_batch_size,
                                            n_steps=n_steps),
                           custom_logger=gail_logger,
                           normalize_obs=False,
                           allow_variable_horizon=True)

  def each_round(r):
    if callback_policy is not None:
      np_policy = get_sb3_policy(gail_trainer.policy, num_states, num_actions)
      callback_policy(np_policy)

  gail_trainer.train(total_timesteps=total_timesteps, callback=each_round)
  # np_policy = np.zeros((num_states, num_actions))
  # for sidx in range(num_states):
  #   np_policy[sidx, :] = get_action_distribution(gail_trainer.policy, sidx)

  return get_sb3_policy(gail_trainer.policy, num_states, num_actions)


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

  return get_sb3_policy(bc_trainer.policy, num_states, num_actions)
