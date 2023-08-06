import torch
from ai_coach_core.model_learning.IQLearn.utils.utils import (average_dicts,
                                                              soft_update,
                                                              hard_update)
from ai_coach_core.model_learning.IQLearn.iq import iq_loss
from ai_coach_core.model_learning.OptionIQL.helper.utils import (
    get_concat_samples)
from .option_softq import OptionSoftQ
from .option_sac import OptionSAC


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


class IQMixin:

  def minimal_iq_update(self,
                        policy_batch,
                        expert_batch,
                        logger,
                        step,
                        is_sqil=False,
                        use_target=False,
                        method_alpha=0.5):
    # args = self.args
    (obs, prev_lat, prev_act, next_obs, latent, action, reward, done,
     is_expert) = get_concat_samples(policy_batch, expert_batch, is_sqil)

    ######
    # IQ-Learn minimal implementation with X^2 divergence (~15 lines)
    # Calculate 1st term of loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
    current_Q = self.critic(obs, prev_lat, prev_act, latent, action)
    y = (1 - done) * self.gamma * self.getV(next_obs, latent, action)
    if use_target:
      with torch.no_grad():
        y = (1 - done) * self.gamma * self.get_targetV(next_obs, latent, action)

    reward = (current_Q - y)[is_expert]
    loss = -(reward).mean()

    # 2nd term of iq loss (use expert and policy states): E_(ρ)[Q(s,a) - γV(s')]
    value_loss = (self.getV(obs, prev_lat, prev_act) - y).mean()
    loss += value_loss

    # Use χ2 divergence (adds a extra term to the loss)
    chi2_loss = 1 / (4 * method_alpha) * (reward**2).mean()
    loss += chi2_loss
    ######

    self.critic_optimizer.zero_grad()
    loss.backward()
    self.critic_optimizer.step()
    return loss

  def iq_update_critic(self,
                       policy_batch,
                       expert_batch,
                       logger,
                       step,
                       is_sqil=False,
                       use_target=False,
                       method_loss="value",
                       method_regularize=True):
    batch = get_concat_samples(policy_batch, expert_batch, is_sqil)
    obs, prev_lat, prev_act, next_obs, latent, action = batch[0:6]

    agent = self
    current_V = self.getV(obs, prev_lat, prev_act)
    if use_target:
      with torch.no_grad():
        next_V = self.get_targetV(next_obs, latent, action)
    else:
      next_V = self.getV(next_obs, latent, action)

    current_Q = self.critic(obs, prev_lat, prev_act, latent, action, both=True)
    if isinstance(current_Q, tuple):
      q1_loss, loss_dict1 = iq_loss(agent, current_Q[0], current_V, next_V,
                                    batch, method_loss, method_regularize)
      q2_loss, loss_dict2 = iq_loss(agent, current_Q[1], current_V, next_V,
                                    batch, method_loss, method_regularize)
      critic_loss = 1 / 2 * (q1_loss + q2_loss)
      # merge loss dicts
      loss_dict = average_dicts(loss_dict1, loss_dict2)
    else:
      critic_loss, loss_dict = iq_loss(agent, current_Q, current_V, next_V,
                                       batch, method_loss, method_regularize)

    # logger.log('train/critic_loss', critic_loss, step)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    if hasattr(self, 'clip_grad_val') and self.clip_grad_val:
      nn.utils.clip_grad_norm_(self._critic.parameters(), self.clip_grad_val)
    # step critic
    self.critic_optimizer.step()
    return loss_dict

  def iq_update(self,
                policy_batch,
                expert_batch,
                logger,
                step,
                is_sqil=False,
                use_target=False,
                do_soft_update=False,
                method_loss="value",
                method_regularize=True):

    for _ in range(self.num_critic_update):
      losses = self.iq_update_critic(policy_batch, expert_batch, logger, step,
                                     is_sqil, use_target, method_loss,
                                     method_regularize)

    # args
    vdice_actor = False
    offline = False

    if self.actor:
      if not vdice_actor:

        if offline:
          obs = expert_batch[0]
          prev_lat = expert_batch[1]
          prev_act = expert_batch[2]
        else:
          # Use both policy and expert observations
          obs = torch.cat([policy_batch[0], expert_batch[0]], dim=0)
          prev_lat = torch.cat([policy_batch[1], expert_batch[1]], dim=0)
          prev_act = torch.cat([policy_batch[2], expert_batch[2]], dim=0)

        if self.num_actor_update:
          for i in range(self.num_actor_update):
            actor_alpha_losses = self.update_actor_and_alpha(
                obs, prev_lat, prev_act, logger, step)

        losses.update(actor_alpha_losses)

    if do_soft_update:
      soft_update(self.critic_net, self.critic_target_net, self.critic_tau)
    else:
      hard_update(self.critic_net, self.critic_target_net)
    return losses
