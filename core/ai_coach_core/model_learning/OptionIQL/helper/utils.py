from typing import Callable, Any, Sequence
from collections import defaultdict
import os
import torch
import pickle
import numpy as np
from gym import Env
from ai_coach_core.latent_inference.decoding import most_probable_sequence_v2
from ai_coach_core.model_learning.IQLearn.utils.utils import eval_mode
from ..agent.option_sac import OptionSAC


def conv_trajectories_2_iql_format(sax_trajectories: Sequence,
                                   cb_conv_action_to_idx: Callable[[Any], int],
                                   cb_get_reward: Callable[[Any, Any],
                                                           float], path: str):
  'sa_trajectories: okay to include the terminal state'
  expert_trajs = defaultdict(list)

  for trajectory in sax_trajectories:
    traj = []
    for t in range(len(trajectory) - 1):
      cur_tup = trajectory[t]
      next_tup = trajectory[t + 1]

      state, action, latent = cur_tup[0], cur_tup[1], cur_tup[2]
      nxt_state, nxt_action, _ = next_tup[0], next_tup[1], next_tup[2]

      aidx = cb_conv_action_to_idx(action)
      reward = cb_get_reward(state, action)

      done = nxt_action is None
      traj.append((state, nxt_state, aidx, latent, reward, done))

    len_traj = len(traj)

    unzipped_traj = zip(*traj)
    states, next_states, actions, latents, rewards, dones = map(
        np.array, unzipped_traj)

    expert_trajs["states"].append(states.reshape(len_traj, -1))
    expert_trajs["next_states"].append(next_states.reshape(len_traj, -1))
    expert_trajs["actions"].append(actions.reshape(len_traj, -1))
    expert_trajs["rewards"].append(rewards)
    expert_trajs["dones"].append(dones)
    expert_trajs["lengths"].append(len(traj))
    expert_trajs["latents"].append(latents.reshape(len_traj, -1))

  print('Final size of Replay Buffer: {}'.format(sum(expert_trajs["lengths"])))
  with open(path, 'wb') as f:
    pickle.dump(expert_trajs, f)


def save(agent: OptionSAC,
         epoch,
         save_interval,
         env_name,
         agent_name,
         alg_type: str,
         output_dir='results',
         suffix=""):
  if epoch % save_interval == 0:
    name = f'{alg_type}_{env_name}'

    if not os.path.exists(output_dir):
      os.mkdir(output_dir)
    file_path = os.path.join(output_dir, f'{agent_name}_{name}' + suffix)
    agent.save(file_path)


def get_concat_samples(policy_batch, expert_batch, is_sqil: bool):
  (online_batch_state, online_batch_prev_lat, online_batch_prev_act,
   online_batch_next_state, online_batch_latent, online_batch_action,
   online_batch_reward, online_batch_done) = policy_batch

  (expert_batch_state, expert_batch_prev_lat, expert_batch_prev_act,
   expert_batch_next_state, expert_batch_latent, expert_batch_action,
   expert_batch_reward, expert_batch_done) = expert_batch

  if is_sqil:
    # convert policy reward to 0
    online_batch_reward = torch.zeros_like(online_batch_reward)
    # convert expert reward to 1
    expert_batch_reward = torch.ones_like(expert_batch_reward)

  batch_state = torch.cat([online_batch_state, expert_batch_state], dim=0)
  batch_prev_lat = torch.cat([online_batch_prev_lat, expert_batch_prev_lat],
                             dim=0)
  batch_prev_act = torch.cat([online_batch_prev_act, expert_batch_prev_act],
                             dim=0)
  batch_next_state = torch.cat(
      [online_batch_next_state, expert_batch_next_state], dim=0)
  batch_latent = torch.cat([online_batch_latent, expert_batch_latent], dim=0)
  batch_action = torch.cat([online_batch_action, expert_batch_action], dim=0)
  batch_reward = torch.cat([online_batch_reward, expert_batch_reward], dim=0)
  batch_done = torch.cat([online_batch_done, expert_batch_done], dim=0)
  is_expert = torch.cat([
      torch.zeros_like(online_batch_reward, dtype=torch.bool),
      torch.ones_like(expert_batch_reward, dtype=torch.bool)
  ],
                        dim=0)

  return (batch_state, batch_prev_lat, batch_prev_act, batch_next_state,
          batch_latent, batch_action, batch_reward, batch_done, is_expert)


def evaluate(agent: OptionSAC, env: Env, num_episodes=10, vis=True):
  """Evaluates the policy.
    Args:
      actor: A policy to evaluate.
      env: Environment to evaluate the policy on.
      num_episodes: A number of episodes to average the policy on.
    Returns:
      Averaged reward and a total number of steps.
    """
  total_timesteps = []
  total_returns = []

  while len(total_returns) < num_episodes:
    state = env.reset()
    prev_latent, prev_act = agent.prev_latent, agent.prev_action
    done = False

    with eval_mode(agent):
      while not done:
        latent, action = agent.choose_action(state,
                                             prev_latent,
                                             prev_act,
                                             sample=False)
        next_state, reward, done, info = env.step(action)
        state = next_state
        prev_latent = latent
        prev_act = action

        if 'episode' in info.keys():
          total_returns.append(info['episode']['r'])
          total_timesteps.append(info['episode']['l'])

  return total_returns, total_timesteps


def infer_mental_states(agent: OptionSAC, expert_traj, num_latent):
  num_samples = len(expert_traj["states"])

  def fit_shape_2_latent(val):
    if len(np.array(val).shape) == 0:
      val = np.repeat(val, num_latent)
    else:
      val = np.vstack([val] * num_latent)
    return val

  def policy_action_prob(agent_idx, obs, act):
    lat = np.arange(num_latent)

    obs = fit_shape_2_latent(obs)
    act = fit_shape_2_latent(act)
    return agent.evaluate_action(obs, lat, act).reshape(-1)

  def gather_trans_x(agent_idx, obs, act, next_obs):
    lat = np.arange(num_latent)
    obs = fit_shape_2_latent(obs)
    act = fit_shape_2_latent(act)
    next_obs = fit_shape_2_latent(next_obs)

    _, log_probs = agent.gather_mental_probs(next_obs, lat, act)
    return log_probs

  def x_prior(agent_idx, obs):
    probs = np.ones(num_latent) / num_latent
    return np.log(probs)

  mental_states = []
  for i_e in range(num_samples):
    expert_states = expert_traj["states"][i_e]
    expert_actions = expert_traj["actions"][i_e]
    x_sequence = most_probable_sequence_v2(expert_states, expert_actions, 1,
                                           num_latent, policy_action_prob,
                                           gather_trans_x, x_prior)
    mental_states.append(x_sequence[0])

  return mental_states


def get_expert_batch(agent: OptionSAC, expert_traj, num_latent, device):
  mental_states = infer_mental_states(agent, expert_traj, num_latent)
  num_samples = len(expert_traj["states"])

  prev_latent, prev_action = agent.prev_latent, agent.prev_action
  batch_obs = []
  batch_prev_lat = []
  batch_prev_act = []
  batch_next_obs = []
  batch_latent = []
  batch_action = []
  batch_reward = []
  batch_done = []

  for i_e in range(num_samples):
    batch_obs.append(np.array(expert_traj["states"][i_e]))

    batch_prev_lat.append(np.array(prev_latent).reshape(-1))
    batch_prev_lat.append(np.array(mental_states[i_e][:-1]).reshape(-1, 1))

    batch_prev_act.append(np.array(prev_action).reshape(-1))
    batch_prev_act.append(np.array(expert_traj["actions"][i_e][:-1]))

    batch_next_obs.append(np.array(expert_traj["next_states"][i_e]))
    batch_latent.append(np.array(mental_states[i_e]).reshape(-1, 1))
    batch_action.append(np.array(expert_traj["actions"][i_e]))
    batch_reward.append(np.array(expert_traj["rewards"][i_e]).reshape(-1, 1))
    batch_done.append(np.array(expert_traj["dones"][i_e]).reshape(-1, 1))

  batch_obs = np.vstack(batch_obs)
  batch_prev_lat = np.vstack(batch_prev_lat)
  batch_prev_act = np.vstack(batch_prev_act)
  batch_next_obs = np.vstack(batch_next_obs)
  batch_latent = np.vstack(batch_latent)
  batch_action = np.vstack(batch_action)
  batch_reward = np.vstack(batch_reward)
  batch_done = np.vstack(batch_done)

  batch_obs = torch.as_tensor(batch_obs, dtype=torch.float, device=device)
  batch_prev_lat = torch.as_tensor(batch_prev_lat,
                                   dtype=torch.float,
                                   device=device)
  batch_prev_act = torch.as_tensor(batch_prev_act,
                                   dtype=torch.float,
                                   device=device)
  batch_next_obs = torch.as_tensor(batch_next_obs,
                                   dtype=torch.float,
                                   device=device)
  batch_latent = torch.as_tensor(batch_latent, dtype=torch.float, device=device)
  batch_action = torch.as_tensor(batch_action, dtype=torch.float, device=device)
  batch_reward = torch.as_tensor(batch_reward, dtype=torch.float, device=device)
  batch_done = torch.as_tensor(batch_done, dtype=torch.float, device=device)

  return (batch_obs, batch_prev_lat, batch_prev_act, batch_next_obs,
          batch_latent, batch_action, batch_reward, batch_done)


def get_samples(batch_size, dataset):
  indexes = np.random.choice(np.arange(len(dataset[0])),
                             size=batch_size,
                             replace=False)

  eo, epl, epa, eno, el, ea, er, ed = (dataset[0][indexes], dataset[1][indexes],
                                       dataset[2][indexes], dataset[3][indexes],
                                       dataset[4][indexes], dataset[5][indexes],
                                       dataset[6][indexes], dataset[7][indexes])
  expert_batch = (eo, epl, epa, eno, el, ea, er, ed)
  return expert_batch
