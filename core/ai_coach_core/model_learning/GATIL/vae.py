import torch
from torch import nn
from torch.autograd import Variable

from .models import Policy, Posterior


class VAE(nn.Module):
  def __init__(self):
    super(VAE, self).__init__()

    self.policy = Policy(state_size=8,
                         action_size=0,
                         latent_size=2,
                         output_size=4,
                         hidden_size=64,
                         output_activation='sigmoid')
    self.posterior = Posterior(state_size=8,
                               action_size=0,
                               latent_size=2,
                               hidden_size=64)

  def encode(self, x, c):
    return self.posterior(torch.cat((x, c), 1))

  def reparameterize(self, mu, logvar):
    if self.training:
      std = logvar.mul(0.5).exp_()
      eps = Variable(std.data.new(std.size()).normal_())
      return eps.mul(std).add_(mu)
    else:
      return mu

  def decode(self, x, c):
    action_mean, action_log_std, action_std = self.policy(torch.cat((x, c), 1))
    return action_mean

  def forward(self, x_t0, x_t1, x_t2, x_t3, c):
    mu, logvar = self.encode(torch.cat((x_t0, x_t1, x_t2, x_t3), 1), c)
    c[:, 0] = self.reparameterize(mu, logvar)
    return self.decode(torch.cat((x_t0, x_t1, x_t2, x_t3), 1), c), mu, logvar
