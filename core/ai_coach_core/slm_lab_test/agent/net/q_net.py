# special module for Q-networks, Q(s, a) -> q
from .base import Net
from .mlp import MLPNet
from ai_coach_core.slm_lab_test.agent.net import net_util
import torch
import torch.nn as nn


class QMLPNet(MLPNet):

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
    state_dim, action_dim = in_dim
    nn.Module.__init__(self)
    Net.__init__(self, device, in_dim, out_dim)

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

    dims = [state_dim + action_dim] + self.hid_layers
    self.model = net_util.build_fc_model(dims, self.hid_layers_activation)
    # add last layer with no activation
    self.model_tail = net_util.build_fc_model([dims[-1], self.out_dim],
                                              self.out_layer_activation)

    net_util.init_layers(self, self.init_fn)
    self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
    self.to(self.device)
    self.train()

  def forward(self, state, action):
    s_a = torch.cat((state, action), dim=-1)
    s_a = self.model(s_a)
    return self.model_tail(s_a)
