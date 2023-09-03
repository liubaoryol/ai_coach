import torch
import torch.nn as nn
from aic_ml.baselines.IQLearn.utils.utils import (average_dicts, soft_update,
                                                  hard_update)
from .option_sac_v2 import OptionSAC_V2
from aic_ml.OptionIQL.helper.utils import (get_concat_samples)
from aic_ml.baselines.IQLearn.iq import iq_loss


class OptionIQL_V2(OptionSAC_V2):

  def iq_update_critic(self,
                       policy_batch,
                       expert_batch,
                       logger,
                       step,
                       is_sqil=False,
                       use_target=False,
                       method_loss="value",
                       method_regularize=True):
    (obs, prev_lat, prev_act, next_obs, latent, action, _, done,
     is_expert) = get_concat_samples(policy_batch, expert_batch, is_sqil)
    vec_v_args = (obs, prev_lat, prev_act)
    vec_next_v_args = (next_obs, latent, action)
    vec_actions = (latent, action)

    agent = self

    current_Q = self.critic(*vec_v_args, *vec_actions, both=True)
    if isinstance(current_Q, tuple):
      q1_loss, loss_dict1 = iq_loss(agent, current_Q[0], vec_v_args,
                                    vec_next_v_args, vec_actions, done,
                                    is_expert, use_target, method_loss,
                                    method_regularize)
      q2_loss, loss_dict2 = iq_loss(agent, current_Q[1], vec_v_args,
                                    vec_next_v_args, vec_actions, done,
                                    is_expert, use_target, method_loss,
                                    method_regularize)
      critic_loss = 1 / 2 * (q1_loss + q2_loss)
      # merge loss dicts
      loss_dict = average_dicts(loss_dict1, loss_dict2)
    else:
      critic_loss, loss_dict = iq_loss(agent, current_Q, vec_v_args,
                                       vec_next_v_args, vec_actions, done,
                                       is_expert, use_target, method_loss,
                                       method_regularize)

    # logger.log_train('critic_loss', critic_loss, step)

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

    if self.policy:
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
