import torch
import torch.nn.functional as F
import os
import numpy as np

from typing import Sequence, Optional
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from .base import T_InitLatent
from .ppo import PPO
from .vae import VAE
from ..network import GAILDiscrim, AbstractPolicy, StateFunction, AbstractTransition  # noqa: E501
from ..buffer import SerializedBuffer, RolloutBuffer
from .utils import one_hot


class DIGAIL(PPO):
  """
    Implementation of DIGAIL, using PPO as the backbone RL algorithm

    Parameters
    ----------
    buffer_exp: SerializedBuffer
        buffer of demonstrations
    discriminator: GAILDiscrim
        discriminator
    transition: AbstractTransition
        mental state transition
    buffer: RolloutBuffer,
        buffer
    actor: AbstractPolicy,
        actor
    critic: StateFunction,
        critic
    device: torch.device
        cpu or cuda
    seed: int
        random seed
    gamma: float
        discount factor
    rollout_length: int
        rollout length of the buffer
    batch_size: int
        batch size for sampling from current policy and demonstrations
    lr_actor: float
        learning rate of the actor
    lr_critic: float
        learning rate of the critic
    lr_disc: float
        learning rate of the discriminator
    epoch_ppo: int
        at each update period, update ppo for these times
    epoch_disc: int
        at each update period, update the discriminator for these times
    clip_eps: float
        clip coefficient in PPO's objective
    lambd: float
        lambd factor
    coef_ent: float
        entropy coefficient
    max_grad_norm: float
        maximum gradient norm
    """
  def __init__(self,
               state_size: torch.Tensor,
               latent_size: torch.Tensor,
               action_size: torch.Tensor,
               discrete_state: bool,
               discrete_latent: bool,
               discrete_action: bool,
               buffer_exp: SerializedBuffer,
               discriminator: GAILDiscrim,
               buffer: RolloutBuffer,
               actor: AbstractPolicy,
               critic: StateFunction,
               transition: AbstractTransition,
               cb_init_latent: T_InitLatent,
               device: torch.device,
               seed: int,
               gamma: float = 0.995,
               rollout_length: int = 50000,
               batch_size: int = 64,
               lr_actor: float = 3e-4,
               lr_critic: float = 3e-4,
               lr_disc: float = 3e-4,
               lr_vae: float = 3e-4,
               epoch_ppo: int = 50,
               epoch_disc: int = 10,
               epoch_vae: int = 20,
               clip_eps: float = 0.2,
               lambd: float = 0.97,
               coef_ent: float = 0.0,
               max_grad_norm: float = 10.0,
               vae_temperatue: float = 2.0):
    super().__init__(state_size, latent_size, action_size, discrete_state,
                     discrete_latent, discrete_action, buffer, actor, critic,
                     cb_init_latent, None, device, seed, gamma, rollout_length,
                     lr_actor, lr_critic, epoch_ppo, clip_eps, lambd, coef_ent,
                     max_grad_norm)

    # expert's buffer
    self.buffer_exp = buffer_exp

    # discriminator
    self.disc = discriminator

    # vae - here, we use the same actor for both vae and ppo
    self.vae = VAE(state_size, latent_size, action_size, discrete_state,
                   discrete_latent, discrete_action, buffer_exp, transition,
                   actor, device, seed, gamma, lr_vae, epoch_vae, batch_size,
                   vae_temperatue)

    self.learning_steps_disc = 0
    self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
    self.batch_size = batch_size
    self.epoch_disc = epoch_disc

    self.traj_exp_latents = None
    self.cur_latents = None
    self.cyclic_sample_counter = 0

  def gen_latent_seqs(self):
    self.traj_exp_latents = []
    for idx in range(len(self.buffer_exp.traj_states)):
      states = self.buffer_exp.traj_states[idx]  # type: list[torch.Tensor]
      actions = self.buffer_exp.traj_actions[idx]  # type: list[torch.Tensor]

      latents = torch.Tensor([], device=self.device)
      latents = torch.cat(
          (latents, self.cb_init_latent(states[0].unsqueeze(0))), dim=0)
      with torch.no_grad():
        for t in range(1, len(states)):
          next_state = states[t]
          latent = latents[t - 1]
          action = actions[t - 1]
          if self.discrete_state:
            next_state = one_hot(next_state.unsqueeze(0), self.state_size,
                                 self.device)
          if self.discrete_latent:
            latent = one_hot(latent.unsqueeze(0), self.latent_size, self.device)
          if self.discrete_action:
            action = one_hot(action.unsqueeze(0), self.action_size, self.device)

          next_latent = self.vae.trans.sample(next_state, latent, action)
          latents = torch.cat((latents, next_latent), dim=0)

      self.traj_exp_latents.append(latents)
      latents = torch.Tensor([], device=self.device)

  def sample_latent_seq(self, num_samples: int) -> Sequence[torch.Tensor]:
    if (self.cyclic_sample_counter >= len(self.traj_exp_latents)
        or self.traj_exp_latents is None):
      self.cyclic_sample_counter = 0
      self.gen_latent_seqs()

    self.cyclic_sample_counter += num_samples

    idxes = np.random.randint(0, len(self.traj_exp_latents), size=num_samples)
    sample_latent_seqs = []
    for idx in idxes:
      sample_latent_seqs.append(self.traj_exp_latents[idx])

    return sample_latent_seqs

  def is_max_time(self, t: int):
    return t >= len(self.cur_latents)

  def get_latent(self,
                 t: int,
                 state: Optional[np.array] = None,
                 prev_latent: Optional[np.array] = None,
                 prev_action: Optional[np.array] = None,
                 prev_state: Optional[np.array] = None):
    """
    return
      Latent state
    """
    if self.pretrain_mode:
      return self.vae.get_latent(t, state, prev_latent, prev_action, prev_state)
    else:
      if t == 0:
        self.cur_latents = self.sample_latent_seq(1)[0]

      if t >= len(self.cur_latents):
        # if t is greater than the length of the latent sequence,
        # just return the last latent again
        return self.cur_latents[-1].cpu().numpy()

      return self.cur_latents[t].cpu().numpy()

  def update(self, writer: SummaryWriter):
    """
        Update the algorithm

        Parameters
        ----------
        writer: SummaryWriter
            writer for logs
        """
    if self.pretrain_mode:
      self.vae.update(writer)
      self.pretraining_steps = self.vae.learning_steps
    else:
      self.learning_steps += 1

      for _ in range(self.epoch_disc):
        self.learning_steps_disc += 1

        # samples from current policy's trajectories
        states, _, actions = self.buffer.sample(self.batch_size)[:3]

        # samples from expert's demonstrations
        states_exp, _, actions_exp = self.buffer_exp.sample(self.batch_size)[:3]

        # update discriminator
        self.update_disc(states, actions, states_exp, actions_exp, writer)

      # we don't use reward signals here
      (states, latents, actions, _, dones, log_pis, next_states,
       next_latents) = self.buffer.get()

      # calculate rewards
      rewards = self.disc_reward(states, actions)
      rewards += self.trans_reward(next_latents, next_states, latents, actions,
                                   dones)

      # update PPO using estimated rewards
      self.update_ppo(states, latents, actions, rewards, dones, log_pis,
                      next_states, next_latents, writer)

  def disc_reward(self, states: torch.Tensor, actions: torch.Tensor):
    if self.discrete_state:
      states = one_hot(states, self.state_size, device=self.device)
    if self.discrete_action:
      actions = one_hot(actions, self.action_size, device=self.device)

    return self.disc.calculate_reward(states, actions)

  def trans_reward(self, next_latents: torch.Tensor, next_states: torch.Tensor,
                   latents: torch.Tensor, actions: torch.Tensor,
                   dones: torch.Tensor):
    if self.discrete_state:
      next_states = one_hot(next_states, self.state_size, device=self.device)
    if self.discrete_action:
      actions = one_hot(actions, self.action_size, device=self.device)
    if self.discrete_latent:
      latents = one_hot(latents, self.latent_size, device=self.device)
      next_latents = one_hot(next_latents, self.latent_size, device=self.device)

    with torch.no_grad():
      # At the terminal step, next_latent is not valid so we don't add anything
      log_Tx = (1 - dones) * self.vae.trans.evaluate_log_Tx(
          next_states, latents, actions, next_latents)
    return log_Tx

  def update_disc(self, states: torch.Tensor, actions: torch.Tensor,
                  states_exp: torch.Tensor, actions_exp: torch.Tensor,
                  writer: SummaryWriter):
    """
        Train the discriminator to distinguish the expert's behavior
        and the imitation learning policy's behavior

        Parameters
        ----------
        states: torch.Tensor
            states sampled from current IL policy
        actions: torch.Tensor
            actions sampled from current IL policy
        states_exp: torch.Tensor
            states sampled from demonstrations
        actions_exp: torch.Tensor
            actions sampled from demonstrations
        writer: SummaryWriter
            writer for logs
        """

    if self.discrete_state:
      states = one_hot(states, self.state_size, device=self.device)
      states_exp = one_hot(states_exp, self.state_size, device=self.device)

    if self.discrete_action:
      actions = one_hot(actions, self.action_size, device=self.device)
      actions_exp = one_hot(actions_exp, self.action_size, device=self.device)

    # output of discriminator is (-inf, inf), not [0, 1]
    logits_pi = self.disc(states, actions)
    logits_exp = self.disc(states_exp, actions_exp)

    # discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)]
    loss_pi = -F.logsigmoid(-logits_pi).mean()
    loss_exp = -F.logsigmoid(logits_exp).mean()
    loss_disc = loss_pi + loss_exp

    self.optim_disc.zero_grad()
    loss_disc.backward()
    self.optim_disc.step()

    if self.learning_steps_disc % self.epoch_disc == 0:
      writer.add_scalar('loss/disc', loss_disc.item(), self.learning_steps)

      # discriminator's accuracies
      with torch.no_grad():
        acc_pi = (logits_pi < 0).float().mean().item()
        acc_exp = (logits_exp > 0).float().mean().item()
      writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
      writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)

  def save_models(self, save_dir: str):
    """
        Save the model

        Parameters
        ----------
        save_dir: str
            path to save
        """
    if not os.path.isdir(save_dir):
      os.mkdir(save_dir)
    torch.save(self.disc.state_dict(), f'{save_dir}/disc.pkl')
    torch.save(self.actor.state_dict(), f'{save_dir}/actor.pkl')
