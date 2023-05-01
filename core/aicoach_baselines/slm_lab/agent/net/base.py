from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class Net(ABC):
  '''Abstract Net class to define the API methods'''

  def __init__(self, device, in_dim, out_dim):
    '''
        @param {int|list} in_dim is the input dimension(s) for the network. 
                Usually use in_dim=body.state_dim
        @param {int|list} out_dim is the output dimension(s) for the network. 
                Usually use out_dim=body.action_dim
        '''
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.grad_norms = None  # for debugging
    self.device = torch.device(device)
    self.clip_grad_val = None

  @abstractmethod
  def forward(self):
    '''The forward step for a specific network architecture'''
    raise NotImplementedError

  def train_step(self, loss, optim, lr_scheduler=None, learning_steps=None):
    if lr_scheduler is not None:
      lr_scheduler.step(epoch=learning_steps)
    optim.zero_grad()
    loss.backward()
    if not self.clip_grad_val:
      nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_val)
    optim.step()
    return loss

  def store_grad_norms(self):
    '''Stores the gradient norms for debugging.'''
    norms = [param.grad.norm().item() for param in self.parameters()]
    self.grad_norms = norms
