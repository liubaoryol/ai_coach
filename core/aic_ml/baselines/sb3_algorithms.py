import tempfile
import pathlib
import numpy as np
from gym import spaces
import stable_baselines3 as sb3
# import gym
# from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from imitation.algorithms.adversarial import gail
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.algorithms import bc
from imitation.data.types import Trajectory
from imitation.util import logger
from imitation.data import rollout
from imitation.util.util import make_vec_env
from aic_core.models.mdp import MDP
import torch as th


def get_sb3_policy(sb3_policy: ActorCriticPolicy, num_states: int,
                   tuple_num_actions: list):
  list_np_policy = []
  for idx in range(len(tuple_num_actions)):
    list_np_policy.append(np.zeros((num_states, tuple_num_actions[idx])))

  for sidx in range(num_states):
    stensor, _ = sb3_policy.obs_to_tensor(sidx)
    act_dist = sb3_policy.get_distribution(stensor)
    if len(list_np_policy) == 1:
      probs = act_dist.distribution.probs
      list_np_policy[0][sidx, :] = probs.cpu().detach().numpy()
    else:
      for idx in range(len(list_np_policy)):
        probs = act_dist.distribution[idx].probs
        list_np_policy[idx][sidx, :] = probs.cpu().detach().numpy()

  if len(list_np_policy) == 1:
    return list_np_policy[0]
  else:
    return list_np_policy


def gail_w_ppo(
    mdp: MDP,
    possible_init_states,
    sa_trajectories,
    n_envs=1,
    logpath=None,
    demo_batch_size=64,
    #  ppo_batch_size=64,
    n_steps=32,
    total_timesteps=100000):

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

  rng = np.random.default_rng(0)
  venv = make_vec_env('envfrommdp-v0',
                      n_envs=n_envs,
                      rng=rng,
                      env_make_kwargs=env_kwargs)

  # transition
  list_trajectories = []
  for traj in sa_trajectories:
    list_state, list_action = list(zip(*traj))
    list_trajectories.append(
        Trajectory(obs=np.array(list_state),
                   acts=np.array(list_action[:-1]),
                   terminal=list_action[-1] is None,
                   infos=None))

  transitions = rollout.flatten_trajectories(list_trajectories)

  if logpath is None:
    tempdir = tempfile.TemporaryDirectory(prefix="gail")
    tempdir_path = pathlib.Path(tempdir.name)
    gail_logger = logger.configure(tempdir_path, format_strs=())
  else:
    print(f"All Tensorboards and logging are being written inside {logpath}/.")
    gail_logger = logger.configure(logpath,
                                   format_strs=("log", "csv", "tensorboard"))

  learner = sb3.PPO("MlpPolicy",
                    venv,
                    batch_size=n_steps * n_envs,
                    n_steps=n_steps,
                    policy_kwargs={'net_arch': [100, 100]})
  reward_net = BasicRewardNet(venv.observation_space,
                              venv.action_space,
                              hid_sizes=(100, 100))

  gail_trainer = gail.GAIL(venv=venv,
                           demonstrations=transitions,
                           demo_batch_size=demo_batch_size,
                           gen_algo=learner,
                           reward_net=reward_net,
                           n_disc_updates_per_round=5,
                           custom_logger=gail_logger)

  gail_trainer.train(total_timesteps=total_timesteps)
  # np_policy = np.zeros((num_states, num_actions))
  # for sidx in range(num_states):
  #   np_policy[sidx, :] = get_action_distribution(gail_trainer.policy, sidx)

  return get_sb3_policy(gail_trainer.policy, mdp.num_states,
                        mdp.list_num_actions)


def behavior_cloning_sb3(sa_trajectories, num_states, num_actions, logpath,
                         n_epochs):
  if len(sa_trajectories) == 0:
    return np.zeros((num_states, num_actions)) / num_actions

  rng = np.random.default_rng(0)

  observation_space = spaces.Discrete(num_states)
  action_space = spaces.Discrete(num_actions)
  bc_logger = logger.configure(logpath, format_strs=("tensorboard", ))

  # transition
  list_trajectories = []
  for traj in sa_trajectories:
    list_state, list_action = list(zip(*traj))
    list_trajectories.append(
        Trajectory(obs=np.array(list_state),
                   acts=np.array(list_action[:-1]),
                   terminal=list_action[-1] is None,
                   infos=None))

  policy = ActorCriticPolicy(observation_space=observation_space,
                             action_space=action_space,
                             lr_schedule=lambda _: th.finfo(th.float32).max,
                             net_arch=[100, 100])

  transitions = rollout.flatten_trajectories(list_trajectories)
  bc_trainer = bc.BC(observation_space=observation_space,
                     action_space=action_space,
                     demonstrations=transitions,
                     rng=rng,
                     policy=policy,
                     custom_logger=bc_logger)
  bc_trainer.train(n_epochs=n_epochs)

  return get_sb3_policy(bc_trainer.policy, num_states, (num_actions, ))
