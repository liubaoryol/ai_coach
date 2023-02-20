from ai_coach_core.slm_lab_test.agent.net import net_util
from .base import Net
import torch.nn as nn


class MLPNet(Net, nn.Module):
  '''
    Class for generating arbitrary sized feedforward neural network
    If more than 1 output tensors, will create a self.model_tails instead of making last layer part of self.model

    '''

  def __init__(self,
               in_dim,
               out_dim,
               device,
               hid_layers=[64, 64, 32],
               hid_layers_activation='relu',
               out_layer_activation=None,
               init_fn='orthogonal_',
               clip_grad_val=0.5,
               loss_spec={"name": "MSELoss"},
               optim_spec={
                   'name': 'Adam',
                   'lr': 0.005
               },
               lr_scheduler_spec={},
               update_type='polyak',
               update_frequency=1,
               polyak_coef=0.005):
    '''
        net_spec:
        hid_layers: list containing dimensions of the hidden layers
        hid_layers_activation: activation function for the hidden layers
        out_layer_activation: activation function for the output layer, same shape as out_dim
        init_fn: weight initialization function
        clip_grad_val: clip gradient norm if value is not None
        loss_spec: measure of error between model predictions and correct outputs
        optim_spec: parameters for initializing the optimizer
        lr_scheduler_spec: Pytorch optim.lr_scheduler
        update_type: method to update network weights: 'replace' or 'polyak'
        update_frequency: how many total timesteps per update
        polyak_coef: ratio of polyak weight update
        gpu: whether to train using a GPU. Note this will only work if a GPU is available, othewise setting gpu=True does nothing
        '''
    nn.Module.__init__(self)
    super().__init__(device, in_dim, out_dim)
    # set default

    self.hid_layers = hid_layers
    self.hid_layers_activation = hid_layers_activation
    self.out_layer_activation = out_layer_activation
    self.init_fn = init_fn
    self.clip_grad_val = clip_grad_val
    self.loss_spec = loss_spec
    self.optim_spec = optim_spec
    self.lr_scheduler_spec = lr_scheduler_spec
    self.update_type = update_type
    self.update_frequency = update_frequency
    self.polyak_coef = polyak_coef

    dims = [self.in_dim] + self.hid_layers
    self.model = net_util.build_fc_model(dims, self.hid_layers_activation)
    # add last layer with no activation
    # tails. avoid list for single-tail for compute speed
    self.model_tail = net_util.build_fc_model([dims[-1], self.out_dim],
                                              self.out_layer_activation)

    net_util.init_layers(self, self.init_fn)
    self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
    self.to(self.device)
    self.train()

  def forward(self, x):
    '''The feedforward step'''
    x = self.model(x)
    return self.model_tail(x)
