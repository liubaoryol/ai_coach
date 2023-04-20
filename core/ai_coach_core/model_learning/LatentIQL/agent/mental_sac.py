import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from torch.optim import Adam
from ai_coach_core.model_learning.IQLearn.utils.utils import (soft_update,
                                                              one_hot_w_nan)
from .mental_models import AbstractMentalActor, AbstractMentalThinker

one_hot = one_hot_w_nan  # alias


class MentalSAC(object):

  def __init__(self, obs_dim, action_dim, lat_dim, batch_size, discrete_obs,
               device, gamma, critic_tau, critic_lr,
               critic_target_update_frequency, init_temp, critic_betas,
               critic: nn.Module, actor: AbstractMentalActor,
               thinker: AbstractMentalThinker, learn_temp,
               actor_update_frequency, actor_lr, actor_betas, thinker_lr,
               thinker_betas, alpha_lr, alpha_betas, clip_grad_val):
    self.gamma = gamma
    self.batch_size = batch_size
    self.discrete_obs = discrete_obs
    self.obs_dim = obs_dim
    self.action_dim = action_dim
    self.lat_dim = lat_dim

    self.device = torch.device(device)

    self.clip_grad_val = clip_grad_val
    self.critic_tau = critic_tau
    self.learn_temp = learn_temp
    self.actor_update_frequency = actor_update_frequency
    self.critic_target_update_frequency = critic_target_update_frequency

    self._critic = critic.to(self.device)

    self.critic_target = copy.deepcopy(self._critic).to(self.device)
    self.critic_target.load_state_dict(self._critic.state_dict())

    self.actor = actor.to(self.device)
    self.thinker = thinker.to(self.device)

    self.log_alpha = torch.tensor(np.log(init_temp)).to(self.device)
    self.log_alpha.requires_grad = True
    # Target Entropy = −dim(A)
    self.target_entropy = -action_dim

    # optimizers
    self.actor_optimizer = Adam(self.actor.parameters(),
                                lr=actor_lr,
                                betas=actor_betas)
    self.thinker_optimizer = Adam(self.thinker.parameters(),
                                  lr=thinker_lr,
                                  betas=thinker_betas)
    self.critic_optimizer = Adam(self._critic.parameters(),
                                 lr=critic_lr,
                                 betas=critic_betas)
    self.log_alpha_optimizer = Adam([self.log_alpha],
                                    lr=alpha_lr,
                                    betas=alpha_betas)
    self.train()
    self.critic_target.train()

  def train(self, training=True):
    self.training = training
    self.actor.train(training)
    self.thinker.train(training)
    self._critic.train(training)

  @property
  def alpha(self):
    return self.log_alpha.exp()

  @property
  def critic_net(self):
    return self._critic

  @property
  def critic_target_net(self):
    return self.critic_target

  def gather_mental_probs(self, state, prev_latent, prev_action):
    # --- convert state
    if self.discrete_obs:
      state = np.array(state).reshape(-1)
      state = torch.FloatTensor(state).to(self.device)
      state = one_hot(state, self.obs_dim)
      state = state.view(-1, self.obs_dim)
    # ------
    else:
      state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

    if self.thinker.is_discrete():
      prev_latent = np.array(prev_latent).reshape(-1)
      prev_latent = torch.FloatTensor(prev_latent).to(self.device)
      prev_latent = one_hot(prev_latent, self.lat_dim)
      prev_latent = prev_latent.view(-1, self.lat_dim)
    else:
      prev_latent = torch.FloatTensor(prev_latent).to(self.device).unsqueeze(0)

    if self.actor.is_discrete():
      prev_action = np.array(prev_action).reshape(-1)
      prev_action = torch.FloatTensor(prev_action).to(self.device)
      prev_action = one_hot(prev_action, self.action_dim)
      prev_action = prev_action.view(-1, self.action_dim)
    else:
      prev_action = prev_action.view(-1, self.action_dim)

    with torch.no_grad():
      probs, log_probs = self.thinker.mental_probs(state, prev_latent,
                                                   prev_action)

    return probs.cpu().detach().numpy(), log_probs.cpu().detach().numpy()

  def evaluate_action(self, state, latent, action):
    # --- convert state
    if self.discrete_obs:
      state = np.array(state).reshape(-1)
      state = torch.FloatTensor(state).to(self.device)
      state = one_hot(state, self.obs_dim)
      state = state.view(-1, self.obs_dim)
    # ------
    else:
      state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

    if self.thinker.is_discrete():
      latent = np.array(latent).reshape(-1)
      latent = torch.FloatTensor(latent).to(self.device)
      latent = one_hot(latent, self.lat_dim)
      latent = latent.view(-1, self.lat_dim)
    else:
      latent = torch.FloatTensor(latent).to(self.device).unsqueeze(0)

    if self.actor.is_discrete():
      action = np.array(action).reshape(-1)
      action = torch.FloatTensor(action).to(self.device)
      action = one_hot(action, self.action_dim)
      action = action.view(-1, self.action_dim)
    else:
      action = action.view(-1, self.action_dim)

    with torch.no_grad():
      log_prob = self.actor.evaluate_action(state, latent, action)
    return log_prob.cpu().detach().numpy()

  def choose_action(self, state, prev_latent, prev_action, sample=False):
    # --- convert state
    if self.discrete_obs:
      state = np.array(state).reshape(-1)
      state = torch.FloatTensor(state).to(self.device)
      state = one_hot(state, self.obs_dim)
      state = state.view(-1, self.obs_dim)
    # ------
    else:
      state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

    if self.thinker.is_discrete():
      prev_latent = np.array(prev_latent).reshape(-1)
      prev_latent = torch.FloatTensor(prev_latent).to(self.device)
      prev_latent = one_hot(prev_latent, self.lat_dim)
      prev_latent = prev_latent.view(-1, self.lat_dim)
    else:
      prev_latent = torch.FloatTensor(prev_latent).to(self.device).unsqueeze(0)

    if self.actor.is_discrete():
      prev_action = np.array(prev_action).reshape(-1)
      prev_action = torch.FloatTensor(prev_action).to(self.device)
      prev_action = one_hot(prev_action, self.action_dim)
      prev_action = prev_action.view(-1, self.action_dim)
    else:
      prev_action = prev_action.view(-1, self.action_dim)

    with torch.no_grad():
      if sample:
        latent, _ = self.thinker.sample(state, prev_latent, prev_action)
        latent_item = latent.detach().cpu().numpy()[0]
        if self.thinker.is_discrete():
          latent = one_hot(latent, self.lat_dim)

        action, _ = self.actor.sample(state, latent)
        action_item = action.detach().cpu().numpy()[0]
      else:
        latent = self.thinker.exploit(state, prev_latent, prev_action)
        latent_item = latent.detach().cpu().numpy()[0]
        if self.thinker.is_discrete():
          latent = one_hot(latent, self.lat_dim)

        action = self.actor.exploit(state, latent)
        action_item = action.detach().cpu().numpy()[0]

    return latent_item, action_item

  def critic(self, obs, prev_latent, prev_action, latent, action, both=False):

    # --- convert state
    if self.discrete_obs:
      obs = one_hot(obs, self.obs_dim)
    # ------

    # --- convert latent
    if self.thinker.is_discrete():
      prev_latent = one_hot(prev_latent, self.lat_dim)
      latent = one_hot(latent, self.lat_dim)
    # ------

    # --- convert discrete action
    if self.actor.is_discrete():
      prev_action = one_hot(prev_action, self.action_dim)
      action = one_hot(action, self.action_dim)
    # ------

    return self._critic(obs, prev_latent, prev_action, latent, action, both)

  def getV(self, obs, prev_latent, prev_action):

    # --- convert state
    if self.discrete_obs:
      obs = one_hot(obs, self.obs_dim)
    # ------

    # --- convert prev_latent
    if self.thinker.is_discrete():
      prev_latent = one_hot(prev_latent, self.lat_dim)
    # ------

    # --- convert prev_action
    if self.actor.is_discrete():
      prev_action = one_hot(prev_action, self.action_dim)
    # ------

    latent, lat_log_prob = self.thinker.sample(obs, prev_latent, prev_action)
    # --- convert prev_latent
    if self.thinker.is_discrete():
      latent = one_hot(latent, self.lat_dim)
    # ------

    action, act_log_prob = self.actor.sample(obs, latent)
    # --- convert discrete action
    if self.actor.is_discrete():
      action = one_hot(action, self.action_dim)
    # ------

    current_Q = self._critic(obs, prev_latent, prev_action, latent, action)
    current_V = current_Q - self.alpha.detach() * (act_log_prob + lat_log_prob)
    return current_V

  def get_targetV(self, obs, prev_latent, prev_action):

    # --- convert state
    if self.discrete_obs:
      obs = one_hot(obs, self.obs_dim)
    # ------

    # --- convert prev_latent
    if self.thinker.is_discrete():
      prev_latent = one_hot(prev_latent, self.lat_dim)
    # ------

    # --- convert prev_action
    if self.actor.is_discrete():
      prev_action = one_hot(prev_action, self.action_dim)
    # ------

    latent, lat_log_prob = self.thinker.sample(obs, prev_latent, prev_action)
    # --- convert prev_latent
    if self.thinker.is_discrete():
      latent = one_hot(latent, self.lat_dim)
    # ------

    action, act_log_prob = self.actor.sample(obs, latent)
    # --- convert discrete action
    if self.actor.is_discrete():
      action = one_hot(action, self.action_dim)
    # ------

    current_Q = self.critic_target(obs, prev_latent, prev_action, latent,
                                   action)
    current_V = current_Q - self.alpha.detach() * (act_log_prob + lat_log_prob)
    return current_V

  def update(self, replay_buffer, logger, step):
    (obs, prev_lat, prev_act, next_obs, latent, action, reward,
     done) = replay_buffer.get_samples(self.batch_size, self.device)

    losses = self.update_critic(obs, prev_lat, prev_act, next_obs, latent,
                                action, reward, done, logger, step)
    if step % self.actor_update_frequency == 0:
      actor_alpha_losses = self.update_actor_and_alpha(obs, prev_lat, prev_act,
                                                       logger, step)
      losses.update(actor_alpha_losses)

    # NOTE: ----
    if step % self.critic_target_update_frequency == 0:
      soft_update(self._critic, self.critic_target, self.critic_tau)

    return losses

  def update_critic(self, obs, prev_lat, prev_act, next_obs, latent, action,
                    reward, done, logger, step):

    # --- convert state
    if self.discrete_obs:
      obs = one_hot(obs, self.obs_dim)
      next_obs = one_hot(next_obs, self.obs_dim)
    # ------

    # --- convert latent
    if self.thinker.is_discrete():
      prev_lat = one_hot(prev_lat, self.lat_dim)
      latent = one_hot(latent, self.lat_dim)
    # ------

    # --- convert action
    if self.actor.is_discrete():
      prev_act = one_hot(prev_act, self.action_dim)
      action = one_hot(action, self.action_dim)
    # ------

    with torch.no_grad():
      next_latent, lat_log_prob = self.thinker.sample(next_obs, latent, action)
      if self.thinker.is_discrete():
        next_latent = one_hot(next_latent, self.lat_dim)

      next_action, act_log_prob = self.actor.sample(next_obs, next_latent)
      if self.actor.is_discrete():
        next_action = one_hot(next_action, self.action_dim)

      target_Q = self.critic_target(next_obs, latent, action, next_latent,
                                    next_action)
      target_V = target_Q - self.alpha.detach() * (act_log_prob + lat_log_prob)
      target_Q = reward + (1 - done) * self.gamma * target_V

    # get current Q estimates
    current_Q = self._critic(obs, prev_lat, prev_act, latent, action, both=True)
    if isinstance(current_Q, tuple):
      q1_loss = F.mse_loss(current_Q[0], target_Q)
      q2_loss = F.mse_loss(current_Q[1], target_Q)
      critic_loss = q1_loss + q2_loss
    else:
      critic_loss = F.mse_loss(current_Q, target_Q)

    logger.log('train/critic_loss', critic_loss, step)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    if self.clip_grad_val is not None:
      nn.utils.clip_grad_norm_(self._critic.parameters(), self.clip_grad_val)
    self.critic_optimizer.step()

    # self.critic.log(logger, step)
    return {'loss/critic': critic_loss.item()}

  def update_actor_and_alpha(self, obs, prev_lat, prev_act, logger, step):

    # --- convert state
    if self.discrete_obs:
      obs = one_hot(obs, self.obs_dim)
    # ------
    if self.thinker.is_discrete():
      prev_lat = one_hot(prev_lat, self.lat_dim)

    if self.actor.is_discrete():
      prev_act = one_hot(prev_act, self.action_dim)

    latent, lat_log_prob = self.thinker.rsample(obs, prev_lat, prev_act)

    action, act_log_prob = self.actor.rsample(obs, latent)
    actor_Q = self._critic(obs, prev_lat, prev_act, latent, action)

    actor_loss = (self.alpha.detach() * (act_log_prob + lat_log_prob) -
                  actor_Q).mean()

    logger.log('train/actor_loss', actor_loss, step)
    logger.log('train/actor_entropy', -act_log_prob.mean(), step)
    logger.log('train/thinker_entropy', -lat_log_prob.mean(), step)

    # optimize the actor
    self.actor_optimizer.zero_grad()
    self.thinker_optimizer.zero_grad()
    actor_loss.backward()
    if self.clip_grad_val is not None:
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad_val)
      nn.utils.clip_grad_norm_(self.thinker.parameters(), self.clip_grad_val)
    self.actor_optimizer.step()
    self.thinker_optimizer.step()

    losses = {
        'loss/actor': actor_loss.item(),
        'actor_loss/actor_entropy': -act_log_prob.mean().item(),
        'actor_loss/thinker_entropy': -lat_log_prob.mean().item()
    }

    # if self.learn_temp:
    #   self.log_alpha_optimizer.zero_grad()
    #   alpha_loss = (self.log_alpha *
    #                 (-log_prob - self.target_entropy).detach()).mean()
    #   logger.log('train/alpha_loss', alpha_loss, step)
    #   logger.log('train/alpha_value', self.alpha, step)

    #   alpha_loss.backward()
    #   self.log_alpha_optimizer.step()

    #   losses.update({
    #       'alpha_loss/loss': alpha_loss.item(),
    #       'alpha_loss/value': self.alpha.item(),
    #   })
    return losses

  # Save model parameters
  def save(self, path, suffix=""):
    actor_path = f"{path}{suffix}_actor"
    thinker_path = f"{path}{suffix}_thinker"
    critic_path = f"{path}{suffix}_critic"

    # print('Saving models to {} and {}'.format(actor_path, critic_path))
    torch.save(self.actor.state_dict(), actor_path)
    torch.save(self.thinker.state_dict(), thinker_path)
    torch.save(self._critic.state_dict(), critic_path)

  # Load model parameters
  def load(self, path):
    actor_path = f'{path}_actor'
    thinker_path = f'{path}_thinker'
    critic_path = f'{path}_critic'
    print('Loading models from {}, {} and {}'.format(actor_path, thinker_path,
                                                     critic_path))
    if actor_path is not None:
      self.actor.load_state_dict(
          torch.load(actor_path, map_location=self.device))
    if thinker_path is not None:
      self.thinker.load_state_dict(
          torch.load(thinker_path, map_location=self.device))
    if critic_path is not None:
      self._critic.load_state_dict(
          torch.load(critic_path, map_location=self.device))
