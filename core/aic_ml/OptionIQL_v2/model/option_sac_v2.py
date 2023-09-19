import torch.nn as nn
import copy
import torch
from torch.functional import F
import numpy as np
from torch.optim import Adam
from aic_ml.baselines.IQLearn.utils.utils import (soft_update, one_hot_w_nan)
from .option_policy import OptionPolicy
from .option_critic import OptionCritic
from omegaconf import DictConfig as Config

one_hot = one_hot_w_nan  # alias


class OptionSAC_V2(nn.Module):

  def __init__(self, config: Config, obs_dim, action_dim, lat_dim,
               critic: OptionCritic, policy: OptionPolicy) -> None:
    super().__init__()

    self.action_dim = action_dim
    self.obs_dim = obs_dim
    self.lat_dim = lat_dim
    self.batch_size = config.mini_batch_size

    self.device = config.device

    self.gamma = config.gamma

    self.init_temp = config.init_temp
    self.critic_tau = 0.005
    self.clip_grad_val = config.clip_grad_val
    self.learn_temp = config.learn_temp
    self.policy_update_frequency = 1
    self.critic_target_update_frequency = 1
    self.num_critic_update = config.num_critic_update
    self.num_actor_update = config.num_actor_update

    self.policy = policy.to(self.device)
    self._critic = critic.to(self.device)

    self.critic_target = copy.deepcopy(self._critic).to(self.device)
    self.critic_target.load_state_dict(self._critic.state_dict())

    self.log_alpha = torch.tensor(np.log(self.init_temp)).to(self.device)
    self.log_alpha.requires_grad = True

    self.target_entropy = -action_dim

    # optimizers
    self.reset_optimizers(config)

    self.to(self.device)
    self.critic_target.train()
    self.train()

  def train(self, training=True):
    self.training = training
    self.policy.train(training)
    self._critic.train(training)

  def reset_optimizers(self, config):
    critic_betas = alpha_betas = policy_betas = [0.9, 0.999]
    self.policy_optimizer = Adam(self.policy.parameters(),
                                 lr=config.optimizer_lr_policy,
                                 betas=policy_betas)
    self.critic_optimizer = Adam(self._critic.parameters(),
                                 lr=config.optimizer_lr_critic,
                                 betas=critic_betas)
    self.log_alpha_optimizer = Adam([self.log_alpha],
                                    lr=config.optimizer_lr_alpha,
                                    betas=alpha_betas)

  @property
  def alpha(self):
    return self.log_alpha.exp()

  @property
  def critic_net(self):
    return self._critic

  @property
  def critic_target_net(self):
    return self.critic_target

  def _conv_input(self, input, is_discrete, dimension):
    if is_discrete:
      input = np.array(input).reshape(-1)
      input = torch.FloatTensor(input).to(self.device)
      input = one_hot(input, dimension)
      # input = input.view(-1, dimension)
    else:
      input = torch.FloatTensor(input).to(self.device)
      if input.ndim < 2:
        input = input.unsqueeze(0)

    return input

  def choose_action(self, state, prev_latent, prev_action, sample=False):
    # --- convert inputs
    state = self._conv_input(state, False, self.obs_dim)
    prev_latent = self._conv_input(prev_latent, False, self.lat_dim)
    prev_action = self._conv_input(prev_action, False, self.action_dim)

    with torch.no_grad():
      latent = self.policy.sample_option(state, prev_latent, fixed=False)
      action = self.policy.sample_action(state, latent, fixed=False)

      latent_item = latent.detach().cpu().numpy()[0]
      action_item = action.detach().cpu().numpy()[0]

    return latent_item, action_item

  def critic(self, obs, prev_latent, prev_action, latent, action, both=False):
    return self._critic(obs, prev_latent, latent, action, both)

  def getV(self, obs, prev_latent, prev_action):

    latent, lat_log_prob = self.policy.sample_option(obs,
                                                     prev_latent,
                                                     fixed=False)

    action, act_log_prob = self.policy.sample_action(obs, latent, fixed=False)

    current_Q = self._critic(obs, prev_latent, latent, action)
    current_V = current_Q - self.alpha.detach() * (act_log_prob + lat_log_prob)
    return current_V

  def get_targetV(self, obs, prev_latent, prev_action):

    latent, lat_log_prob = self.policy.sample_option(obs,
                                                     prev_latent,
                                                     fixed=False)

    action, act_log_prob = self.policy.sample_action(obs, latent, fixed=False)

    current_Q = self.critic_target(obs, prev_latent, latent, action)
    current_V = current_Q - self.alpha.detach() * (act_log_prob + lat_log_prob)
    return current_V

  def update(self, replay_buffer, logger, step):
    (obs, prev_lat, prev_act, next_obs, latent, action, reward,
     done) = replay_buffer.get_samples(self.batch_size, self.device)

    losses = self.update_critic(obs, prev_lat, prev_act, next_obs, latent,
                                action, reward, done, logger, step)
    if step % self.policy_update_frequency == 0:
      actor_alpha_losses = self.update_actor_and_alpha(obs, prev_lat, prev_act,
                                                       logger, step)
      losses.update(actor_alpha_losses)

    # NOTE: ----
    if step % self.critic_target_update_frequency == 0:
      soft_update(self._critic, self.critic_target, self.critic_tau)

    return losses

  def update_critic(self, obs, prev_lat, prev_act, next_obs, latent, action,
                    reward, done, logger, step):

    with torch.no_grad():
      next_latent, lat_log_prob = self.policy.sample_option(next_obs, latent)

      next_action, act_log_prob = self.policy.sample_action(
          next_obs, next_latent)

      target_Q = self.critic_target(next_obs, latent, next_latent, next_action)
      target_V = target_Q - self.alpha.detach() * (act_log_prob + lat_log_prob)
      target_Q = reward + (1 - done) * self.gamma * target_V

    # get current Q estimates
    current_Q = self._critic(obs, prev_lat, latent, action, both=True)
    if isinstance(current_Q, tuple):
      q1_loss = F.mse_loss(current_Q[0], target_Q)
      q2_loss = F.mse_loss(current_Q[1], target_Q)
      critic_loss = q1_loss + q2_loss
    else:
      critic_loss = F.mse_loss(current_Q, target_Q)

    # logger.log_train('critic_loss', critic_loss, step)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    if self.clip_grad_val:
      nn.utils.clip_grad_norm_(self._critic.parameters(), self.clip_grad_val)
    self.critic_optimizer.step()

    return {'loss/critic': critic_loss.item()}

  def update_actor_and_alpha(self, obs, prev_lat, prev_act, logger, step):

    latent, lat_log_prob = self.policy.rsample_option(obs, prev_lat)

    action, act_log_prob = self.policy.rsample_action(obs, latent)
    actor_Q = self._critic(obs, prev_lat, latent, action)

    actor_loss = (self.alpha.detach() * (act_log_prob + lat_log_prob) -
                  actor_Q).mean()

    # logger.log_train('actor_loss', actor_loss, step)
    # logger.log_train('actor_entropy', -act_log_prob.mean(), step)
    # logger.log_train('thinker_entropy', -lat_log_prob.mean(), step)

    # optimize the actor
    self.policy_optimizer.zero_grad()
    actor_loss.backward()
    if self.clip_grad_val:
      nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad_val)
    self.policy_optimizer.step()

    losses = {
        'loss/actor': actor_loss.item(),
        'actor_loss/actor_entropy': -act_log_prob.mean().item(),
        'actor_loss/thinker_entropy': -lat_log_prob.mean().item()
    }

    return losses
