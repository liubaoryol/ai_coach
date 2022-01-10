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
from ai_coach_core.models.mdp import MDP
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


def gail_w_ppo(mdp: MDP,
               possible_init_states,
               sa_trajectories,
               terminal_value,
               logpath=None,
               demo_batch_size=64,
               ppo_batch_size=32,
               n_steps=64,
               total_timesteps=32000,
               do_pretrain=True,
               only_pretrain=False):

  if len(sa_trajectories) == 0:
    return np.ones((mdp.num_states, mdp.num_actions)) / mdp.num_actions

  # create vec env
  # gymenv = gym.make('envfrommdp-v0',
  #                   num_states=num_states,
  #                   num_actions=num_actions,
  #                   cb_transition=cb_transition,
  #                   cb_is_terminal=cb_is_terminal,
  #                   cb_legal_actions=cb_legal_actions,
  #                   init_state=init_state)
  # venv = DummyVecEnv([lambda: gymenv])

  env_kwargs = dict(mdp=mdp, possible_init_states=possible_init_states)

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

  # Pretrain
  if do_pretrain:
    bc_trainer = bc.BC(observation_space=venv.observation_space,
                       action_space=venv.action_space,
                       demonstrations=transitions,
                       batch_size=demo_batch_size,
                       policy=gail_trainer.gen_algo.policy,
                       custom_logger=gail_logger)
    bc_trainer.train(n_epochs=1)

    if only_pretrain:
      return get_sb3_policy(gail_trainer.policy, mdp.num_states,
                            mdp.num_actions)

  gail_trainer.train(total_timesteps=total_timesteps)
  # np_policy = np.zeros((num_states, num_actions))
  # for sidx in range(num_states):
  #   np_policy[sidx, :] = get_action_distribution(gail_trainer.policy, sidx)

  return get_sb3_policy(gail_trainer.policy, mdp.num_states, mdp.num_actions)


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
