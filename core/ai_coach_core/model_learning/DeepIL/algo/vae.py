import os
import torch
import numpy as np
import itertools
from .base import Algorithm, Expert
from .utils import gumbel_softmax_sample, one_hot
from typing import Sequence

from torch.optim import Adam
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from ..network import AbstractPolicy, AbstractTransition  # noqa: E501
from ..buffer import SerializedBuffer
from ..utils import disable_gradient


class VAE(Algorithm):
  def __init__(self,
               state_size: torch.Tensor,
               latent_size: torch.Tensor,
               action_size: torch.Tensor,
               discrete_state: bool,
               discrete_latent: bool,
               discrete_action: bool,
               buffer_exp: SerializedBuffer,
               transition: AbstractTransition,
               actor: AbstractPolicy,
               device: torch.device,
               seed: int,
               gamma: float,
               lr_vae: float = 3e-4,
               batch_size: int = 64,
               temperature: float = 2.0):
    super().__init__(state_size, latent_size, action_size, discrete_state,
                     discrete_latent, discrete_action, actor, transition,
                     device, seed, gamma)
    self.buffer_exp = buffer_exp

    params = [self.actor.parameters(), self.trans.parameters()]
    self.optim_vae = Adam(itertools.chain(*params), lr=lr_vae)

    self.learning_steps_vae = 0
    self.batch_size = batch_size
    self.temperature = temperature

  def update(self, writer: SummaryWriter):
    """
        Update the algorithm

        Parameters
        ----------
        writer: SummaryWriter
            writer for logs
        """
    self.learning_steps += 1
    traj_states = self.buffer_exp.traj_states
    traj_actions = self.buffer_exp.traj_actions
    traj_next_states = self.buffer_exp.traj_next_states
    self.update_vae(traj_states, traj_actions, traj_next_states, writer)

  def initial_latent(self, states: torch.Tensor) -> torch.Tensor:
    if self.discrete_latent:
      # TODO: implement p(x|s)
      return torch.randint(low=0,
                           high=self.latent_size,
                           size=(len(states), ),
                           dtype=torch.float,
                           device=self.device).reshape(len(states), -1)
    else:
      # TODO: temporary code
      return torch.rand(size=(len(states), ),
                        dtype=torch.float,
                        device=self.device).reshape(len(states), -1)

  def encode_decode(self, states: torch.Tensor, latents: torch.Tensor,
                    actions: torch.Tensor, next_states: torch.Tensor,
                    next_actions: torch.Tensor):
    if self.discrete_state:
      next_states = one_hot(next_states, self.state_size, device=self.device)

    if self.discrete_action:
      actions = one_hot(actions, self.action_size, device=self.device)

    if self.discrete_latent:
      latents = one_hot(latents, self.latent_size, device=self.device)
      # encoder
      next_latent_logits = self.trans(next_states, latents, actions)
      next_latent_samples = gumbel_softmax_sample(next_latent_logits,
                                                  self.temperature)
      # decoder
      log_pis = self.actor.evaluate_log_pi(next_states, next_latent_samples,
                                           next_actions)
      return next_latent_logits, log_pis
    else:
      raise NotImplementedError

  def update_vae(self, traj_states: Sequence[torch.Tensor],
                 traj_actions: Sequence[torch.Tensor],
                 traj_next_states: Sequence[torch.Tensor],
                 writer: SummaryWriter):
    num_traj = len(traj_states)
    batch = np.zeros((self.batch_size, 3))  # N x (traj_idx, t, traj_len - 1)

    states = torch.zeros((self.batch_size, len(traj_states[0][0])),
                         dtype=torch.float,
                         device=self.device)
    latents = torch.zeros((self.batch_size, 1),
                          dtype=torch.float,
                          device=self.device)
    actions = torch.zeros((self.batch_size, len(traj_actions[0][0])),
                          dtype=torch.float,
                          device=self.device)
    next_states = torch.zeros((self.batch_size, len(traj_states[0][0])),
                              dtype=torch.float,
                              device=self.device)
    next_actions = torch.zeros((self.batch_size, len(traj_actions[0][0])),
                               dtype=torch.float,
                               device=self.device)
    epoch_vae = int(np.ceil(num_traj / self.batch_size))
    for _ in range(epoch_vae):
      self.learning_steps_vae += 1

      # replace terminated episodes with new ones
      reset_idxs = np.where(batch[:, 1] == batch[:, 2])[0]
      if len(reset_idxs) > 0:
        idxs = np.random.randint(low=0, high=num_traj, size=len(reset_idxs))
        t_maxs = [len(traj_states[idx]) - 1 for idx in idxs]
        batch[reset_idxs, 0] = idxs
        batch[reset_idxs, 1] = 0
        batch[reset_idxs, 2] = t_maxs
        latents[reset_idxs, :] = self.initial_latent(
            torch.cat([traj_states[idx][0].unsqueeze(0) for idx in idxs],
                      dim=0))

      # make inputs
      for idx in range(self.batch_size):
        traj_idx = batch[idx, 0]
        t = batch[idx, 1]
        states[idx] = traj_states[traj_idx][t]
        actions[idx] = traj_actions[traj_idx][t]
        next_states[idx] = traj_next_states[traj_idx][t]
        next_actions[idx] = traj_actions[traj_idx][t + 1]

      out_vars = self.encode_decode(states, latents, actions, next_states,
                                    next_actions)

      self.optim_vae.zero_grad()
      # compute loss
      if self.discrete_latent:
        next_latent_logits, log_pis = out_vars
        recon_loss = -log_pis.mean()

        log_prior = torch.log(1 / self.latent_size)
        dist = Categorical(logits=next_latent_logits)
        p_x = dist.probs
        log_px = dist.logits

        kld_loss = torch.sum(p_x * (log_px - log_prior), dim=-1).mean()

        loss = recon_loss + kld_loss
        loss.backward()

        # next latents
        latents[:, :] = dist.sample()
      else:
        raise NotImplementedError
      self.optim_vae.step()

      batch[:, 1] += 1  # next timestep

    writer.add_scalar('loss/recon', recon_loss.item(), self.learning_steps)
    writer.add_scalar('loss/kld', kld_loss.item(), self.learning_steps)

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
    torch.save(self.actor.state_dict(), f'{save_dir}/vae_actor.pkl')
    torch.save(self.trans.state_dict(), f'{save_dir}/vae_trans.pkl')


class VAEExpert(Expert):
  """
    Well-trained GAIL agent

    Parameters
    ----------
    device: torch.device
        cpu or cuda
    path: str
        path to the well-trained weights
    """
  def __init__(self, state_size: torch.Tensor, latent_size: torch.Tensor,
               action_size: torch.Tensor, discrete_state: bool,
               discrete_latent: bool, discrete_action: bool,
               actor: AbstractPolicy, transition: AbstractTransition,
               device: torch.device, path: str):
    super(VAEExpert, self).__init__(state_size, latent_size, action_size,
                                    discrete_state, discrete_latent,
                                    discrete_action, actor, transition, device)
    self.actor.load_state_dict(torch.load(path, map_location=device))
    disable_gradient(self.actor)
    disable_gradient(self.trans)
