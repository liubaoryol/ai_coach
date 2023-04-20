from ..algorithm import policy_util
from .base import Algorithm
from ..net import net_util
from ...lib import util
import numpy as np
import torch
import torch.nn.functional as F
import copy


class SoftActorCritic(Algorithm):
  '''
    Implementation of Soft Actor-Critic (SAC)
    Original paper: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
    https://arxiv.org/abs/1801.01290
    Improvement of SAC paper: "Soft Actor-Critic Algorithms and Applications"
    https://arxiv.org/abs/1812.05905

    e.g. algorithm_spec
    "algorithm": {
        "name": "SoftActorCritic",
        "action_pdtype": "default",
        "action_policy": "default",
        "gamma": 0.99,
        "training_frequency": 1,
    }
    '''

  def init_algorithm_params(self):
    '''Initialize other algorithm parameters'''
    self.training_iter = 1
    self.training_start_step = self.batch_size

    self.action_type = policy_util.get_action_type(self.env.action_space)
    self.ActionPD = policy_util.get_action_pd_cls(self.action_pdtype,
                                                  self.action_type)

    if self.is_discrete():
      assert self.action_pdtype == 'GumbelSoftmax'
    self.to_train = 0
    self.action_policy = getattr(policy_util, self.action_policy)

  def is_discrete(self):
    return self.action_type == 'discrete'

  def init_nets(self, **kwargs):
    '''
        Networks: net(actor/policy), q1_net, target_q1_net, q2_net, target_q2_net
        All networks are separate, and have the same hidden layer architectures and optim specs, so tuning is minimal
        '''
    self.shared = False  # SAC does not share networks
    NetClass = kwargs['net_class']
    QNetClass = kwargs['qnet_class']
    kwargs.pop('net_class')
    kwargs.pop('qnet_class')

    # main actor network
    self.net = NetClass(**kwargs)
    self.net_names = ['net']
    # two critic Q-networks to mitigate positive bias in q_loss and speed up training, uses q_net.py with prefix Q
    # q_in_dim = [kwargs['in_dim'], kwargs['out_dim']]
    qnet_kwargs = copy.deepcopy(kwargs)
    qnet_kwargs['in_dim'] = [kwargs['in_dim'], kwargs['out_dim']]
    qnet_kwargs['out_dim'] = 1

    self.q1_net = QNetClass(**qnet_kwargs)
    self.target_q1_net = QNetClass(**qnet_kwargs)
    self.q2_net = QNetClass(**qnet_kwargs)
    self.target_q2_net = QNetClass(**qnet_kwargs)
    self.net_names += ['q1_net', 'target_q1_net', 'q2_net', 'target_q2_net']
    net_util.copy(self.q1_net, self.target_q1_net)
    net_util.copy(self.q2_net, self.target_q2_net)
    # temperature variable to be learned, and its target entropy
    self.log_alpha = torch.zeros(1, requires_grad=True, device=self.net.device)
    self.alpha = self.log_alpha.detach().exp()
    if self.is_discrete():
      self.target_entropy = -self.env.action_space.n
    else:
      self.target_entropy = -np.product(self.env.action_space.shape)

    # init net optimizer and its lr scheduler
    self.optim = net_util.get_optim(self.net, self.net.optim_spec)
    self.lr_scheduler = net_util.get_lr_scheduler(self.optim,
                                                  self.net.lr_scheduler_spec)
    self.q1_optim = net_util.get_optim(self.q1_net, self.q1_net.optim_spec)
    self.q1_lr_scheduler = net_util.get_lr_scheduler(
        self.q1_optim, self.q1_net.lr_scheduler_spec)
    self.q2_optim = net_util.get_optim(self.q2_net, self.q2_net.optim_spec)
    self.q2_lr_scheduler = net_util.get_lr_scheduler(
        self.q2_optim, self.q2_net.lr_scheduler_spec)
    self.alpha_optim = net_util.get_optim(self.log_alpha, self.net.optim_spec)
    self.alpha_lr_scheduler = net_util.get_lr_scheduler(
        self.alpha_optim, self.net.lr_scheduler_spec)
    self.end_init_nets()

  def act(self, state):
    action = self.action_policy(state, self, self.ActionPD)
    if not self.is_discrete():
      action = self.scale_action(torch.tanh(action))  # continuous action bound
    return action.cpu().squeeze().numpy()

  def scale_action(self, action):
    '''Scale continuous actions from tanh range'''
    action_space = self.env.action_space
    device = action.device
    low, high = torch.from_numpy(action_space.low).to(device), torch.from_numpy(
        action_space.high).to(device)
    return action * (high - low) / 2 + (low + high) / 2

  def guard_q_actions(self, actions):
    '''Guard to convert actions to one-hot for input to Q-network'''
    if self.is_discrete():
      # TODO support multi-discrete actions
      actions = F.one_hot(actions.long(), self.net.out_dim).float()
    return actions

  def calc_log_prob_action(self, action_pd, reparam=False):
    '''Calculate log_probs and actions with option to reparametrize from paper eq. 11'''
    samples = action_pd.rsample() if reparam else action_pd.sample()
    if self.is_discrete():  # this is straightforward using GumbelSoftmax
      actions = samples
      log_probs = action_pd.log_prob(actions)
    else:
      mus = samples
      actions = self.scale_action(torch.tanh(mus))
      if actions.dim() == 1:  # handle shape consistency for single actions
        actions = actions.unsqueeze(dim=-1)
      # paper Appendix C. Enforcing Action Bounds for continuous actions
      log_probs = (action_pd.log_prob(mus) -
                   torch.log(1 - actions.pow(2) + 1e-6).sum(1))
    return log_probs, actions

  def calc_pdparam(self, x, net=None):
    '''
        The pdparam will be the logits for discrete prob. dist., or the mean and std for continuous prob. dist.
        '''
    net = self.net if net is None else net
    out = net(x)

    if self.shared:
      if len(out) == 2:  # single policy
        pdparam = out[0]
      else:  # multiple-task policies, still assumes 1 value
        pdparam = out[:-1]
      self.v_pred = out[-1].view(
          -1)  # cache for loss calc to prevent double-pass
    else:  # out is pdparam
      pdparam = out
    return pdparam

  def calc_q(self, state, action, net):
    '''Forward-pass to calculate the predicted state-action-value from q1_net.'''
    if not self.is_discrete() and action.dim(
    ) == 1:  # handle shape consistency for single continuous action
      action = action.unsqueeze(dim=-1)
    q_pred = net(state, action).view(-1)
    return q_pred

  def calc_q_targets(self, batch):
    '''Q_tar = r + gamma * (target_Q(s', a') - alpha * log pi(a'|s'))'''
    next_states = batch['next_states']
    with torch.no_grad():
      pdparams = self.calc_pdparam(next_states)
      action_pd = policy_util.init_action_pd(self.ActionPD, pdparams)
      next_log_probs, next_actions = self.calc_log_prob_action(action_pd)
      next_actions = self.guard_q_actions(
          next_actions
      )  # non-reparam discrete actions need to be converted into one-hot

      next_target_q1_preds = self.calc_q(next_states, next_actions,
                                         self.target_q1_net)
      next_target_q2_preds = self.calc_q(next_states, next_actions,
                                         self.target_q2_net)
      next_target_q_preds = torch.min(next_target_q1_preds,
                                      next_target_q2_preds)
      q_targets = batch['rewards'] + self.gamma * (1 - batch['dones']) * (
          next_target_q_preds - self.alpha * next_log_probs)
    return q_targets

  def calc_reg_loss(self, preds, targets):
    '''Calculate the regression loss for V and Q values, using the same loss function from net_spec'''
    assert preds.shape == targets.shape, f'{preds.shape} != {targets.shape}'
    reg_loss = self.net.loss_fn(preds, targets)
    return reg_loss

  def calc_policy_loss(self, batch, log_probs, reparam_actions):
    '''policy_loss = alpha * log pi(f(a)|s) - Q1(s, f(a)), where f(a) = reparametrized action'''
    states = batch['states']
    q1_preds = self.calc_q(states, reparam_actions, self.q1_net)
    q2_preds = self.calc_q(states, reparam_actions, self.q2_net)
    q_preds = torch.min(q1_preds, q2_preds)
    policy_loss = (self.alpha * log_probs - q_preds).mean()
    return policy_loss

  def calc_alpha_loss(self, log_probs):
    alpha_loss = -(self.log_alpha *
                   (log_probs.detach() + self.target_entropy)).mean()
    return alpha_loss

  def try_update_per(self, q_preds, q_targets):
    if 'Prioritized' in util.get_class_name(self.memory):  # PER
      with torch.no_grad():
        errors = (q_preds - q_targets).abs().cpu().numpy()
      self.memory.update_priorities(errors)

  def train_alpha(self, alpha_loss, learning_steps):
    '''Custom method to train the alpha variable'''
    self.alpha_lr_scheduler.step(epoch=learning_steps)
    self.alpha_optim.zero_grad()
    alpha_loss.backward()
    self.alpha_optim.step()
    self.alpha = self.log_alpha.detach().exp()

  def sample(self):
    '''Samples a batch from memory'''
    batch = self.memory.sample()
    batch = util.to_torch_batch(batch, self.net.device, self.memory.is_episodic)
    return batch

  def train(self, learning_steps):
    '''Train actor critic by computing the loss in batch efficiently'''
    if self.to_train == 1:
      for _ in range(self.training_iter):
        batch = self.sample()

        states = batch['states']
        actions = self.guard_q_actions(batch['actions'])
        q_targets = self.calc_q_targets(batch)
        # Q-value loss for both Q nets
        q1_preds = self.calc_q(states, actions, self.q1_net)
        q1_loss = self.calc_reg_loss(q1_preds, q_targets)
        self.q1_net.train_step(q1_loss, self.q1_optim, self.q1_lr_scheduler,
                               learning_steps)

        q2_preds = self.calc_q(states, actions, self.q2_net)
        q2_loss = self.calc_reg_loss(q2_preds, q_targets)
        self.q2_net.train_step(q2_loss, self.q2_optim, self.q2_lr_scheduler,
                               learning_steps)

        # policy loss
        action_pd = policy_util.init_action_pd(self.ActionPD,
                                               self.calc_pdparam(states))
        log_probs, reparam_actions = self.calc_log_prob_action(action_pd,
                                                               reparam=True)
        policy_loss = self.calc_policy_loss(batch, log_probs, reparam_actions)
        self.net.train_step(policy_loss, self.optim, self.lr_scheduler,
                            learning_steps)

        # alpha loss
        alpha_loss = self.calc_alpha_loss(log_probs)
        self.train_alpha(alpha_loss, learning_steps)

        loss = q1_loss + q2_loss + policy_loss + alpha_loss
        # update target networks
        self.update_nets(learning_steps)
        # update PER priorities if availalbe
        self.try_update_per(torch.min(q1_preds, q2_preds), q_targets)

      # reset
      self.to_train = 0
      return loss.item(), policy_loss.item(), (
          q1_loss + q2_loss).item(), alpha_loss.item()
    else:
      return np.nan

  def update_nets(self, learning_steps):
    '''Update target networks'''
    if util.frame_mod(learning_steps, self.q1_net.update_frequency, 1):
      if self.q1_net.update_type == 'replace':
        net_util.copy(self.q1_net, self.target_q1_net)
        net_util.copy(self.q2_net, self.target_q2_net)
      elif self.q1_net.update_type == 'polyak':
        net_util.polyak_update(self.q1_net, self.target_q1_net,
                               self.q1_net.polyak_coef)
        net_util.polyak_update(self.q2_net, self.target_q2_net,
                               self.q2_net.polyak_coef)
      else:
        raise ValueError(
            'Unknown q1_net.update_type. Should be "replace" or "polyak". Exiting.'
        )

  def update(self):
    '''Override parent method to do nothing'''
    return None
