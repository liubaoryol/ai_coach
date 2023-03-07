import os
import torch
import numpy as np
from gym import Env
from ai_coach_core.latent_inference.decoding import most_probable_sequence_v2
from ai_coach_core.model_learning.IQLearn.utils.utils import eval_mode
from ..agent.mental_sac import MentalSAC

NAN = float("nan")


def save(agent: MentalSAC,
         epoch,
         save_interval,
         env_name,
         agent_name,
         is_sqil: bool,
         output_dir='results',
         suffix=""):
  if epoch % save_interval == 0:
    if is_sqil:
      name = f'sqil_{env_name}'
    else:
      name = f'iq_{env_name}'

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


def evaluate(agent: MentalSAC, env: Env, num_latent, num_episodes=10, vis=True):
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
    prev_latent = NAN
    prev_action = NAN
    done = False

    with eval_mode(agent):
      while not done:
        latent, action = agent.choose_action(state,
                                             prev_latent,
                                             prev_action,
                                             sample=False)
        next_state, reward, done, info = env.step(action)
        state = next_state
        prev_latent = latent
        prev_action = action

        if 'episode' in info.keys():
          total_returns.append(info['episode']['r'])
          total_timesteps.append(info['episode']['l'])

  return total_returns, total_timesteps


def infer_mental_states(agent: MentalSAC, expert_traj, num_latent):
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
    return agent.evaluate_action(obs, lat, act)

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


def get_expert_batch(agent: MentalSAC, expert_traj, num_latent, device):
  mental_states = infer_mental_states(agent, expert_traj, num_latent)
  num_samples = len(expert_traj["states"])

  batch_obs = []
  batch_prev_lat = []
  batch_prev_act = []
  batch_next_obs = []
  batch_latent = []
  batch_action = []
  batch_reward = []
  batch_done = []
  for i_e in range(num_samples):
    batch_obs = batch_obs + expert_traj["states"][i_e]
    batch_prev_lat = batch_prev_lat + [NAN] + mental_states[i_e][:-1]
    batch_prev_act = batch_prev_act + [NAN] + expert_traj["actions"][i_e][:-1]
    batch_next_obs = batch_next_obs + expert_traj["next_states"][i_e]
    batch_latent = batch_latent + mental_states[i_e]
    batch_action = batch_action + expert_traj["actions"][i_e]
    batch_reward = batch_reward + expert_traj["rewards"][i_e]
    batch_done = batch_done + expert_traj["dones"][i_e]

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
  batch_reward = torch.as_tensor(batch_reward, dtype=torch.float,
                                 device=device).unsqueeze(1)
  batch_done = torch.as_tensor(batch_done, dtype=torch.float,
                               device=device).unsqueeze(1)

  return (batch_obs, batch_prev_lat, batch_prev_act, batch_next_obs,
          batch_latent, batch_action, batch_reward, batch_done)
