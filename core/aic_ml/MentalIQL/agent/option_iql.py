from typing import Type, Callable
import torch
import torch.nn as nn
from aic_ml.baselines.option_gail.utils.config import Config
from aic_ml.baselines.IQLearn.utils.utils import (average_dicts, soft_update,
                                                  hard_update)
from aic_ml.baselines.IQLearn.iq import iq_loss
from aic_ml.OptionIQL.helper.utils import (get_concat_samples)
from .option_softq import OptionSoftQ
from .option_sac import OptionSAC


class IQMixin:

  def get_iq_variables(self, batch):
    'return vec_v_args, vec_next_v_args, vec_actions, done'
    raise NotImplementedError

  def iq_update_critic(self,
                       policy_batch,
                       expert_batch,
                       logger,
                       update_count,
                       is_sqil=False,
                       use_target=False,
                       method_loss="value",
                       method_regularize=True):
    batch = get_concat_samples(policy_batch, expert_batch, is_sqil)

    vec_v_args, vec_next_v_args, vec_actions, done = self.get_iq_variables(
        batch[:-1])
    is_expert = batch[-1]

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
                update_count,
                is_sqil=False,
                use_target=False,
                do_soft_update=False,
                method_loss="value",
                method_regularize=True):

    for _ in range(self.num_critic_update):
      losses = self.iq_update_critic(policy_batch, expert_batch, logger,
                                     update_count, is_sqil, use_target,
                                     method_loss, method_regularize)

    # args
    vdice_actor = False
    offline = False

    if self.actor:
      if not vdice_actor:

        vec_v_args_policy, _, _, _ = self.get_iq_variables(policy_batch)
        vec_v_args_expert, _, _, _ = self.get_iq_variables(expert_batch)

        if offline:
          vec_v_args = vec_v_args_expert
        else:
          # Use both policy and expert observations
          vec_v_args = []
          for idx in range(len(vec_v_args_expert)):
            item = torch.cat([vec_v_args_policy[idx], vec_v_args_expert[idx]],
                             dim=0)
            vec_v_args.append(item)

        for i in range(self.num_actor_update):
          actor_alpha_losses = self.update_actor_and_alpha(
              *vec_v_args, logger, update_count)

        losses.update(actor_alpha_losses)

    if use_target and update_count % self.critic_target_update_frequency == 0:
      if do_soft_update:
        soft_update(self.critic_net, self.critic_target_net, self.critic_tau)
      else:
        hard_update(self.critic_net, self.critic_target_net)
    return losses


class IQLOptionSoftQ(IQMixin, OptionSoftQ):

  def __init__(self, config: Config, num_inputs, action_dim, option_dim,
               discrete_obs, q_net_base: Type[nn.Module],
               cb_get_iq_variables: Callable):
    super().__init__(config, num_inputs, action_dim, option_dim, discrete_obs,
                     q_net_base)
    self.cb_get_iq_variables = cb_get_iq_variables
    self.method_loss = config.method_loss
    self.method_regularize = config.method_regularize

  def get_iq_variables(self, batch):
    'return vec_v_args, vec_next_v_args, vec_actions, done'
    return self.cb_get_iq_variables(batch)


class IQLOptionSAC(IQMixin, OptionSAC):

  def __init__(self, config: Config, obs_dim, action_dim, option_dim,
               discrete_obs, critic_base: Type[nn.Module], actor,
               cb_get_iq_variables: Callable):
    super().__init__(config, obs_dim, action_dim, option_dim, discrete_obs,
                     critic_base, actor)
    self.cb_get_iq_variables = cb_get_iq_variables
    self.method_loss = config.method_loss
    self.method_regularize = config.method_regularize

  def get_iq_variables(self, batch):
    'return vec_v_args, vec_next_v_args, vec_actions, done'
    return self.cb_get_iq_variables(batch)
