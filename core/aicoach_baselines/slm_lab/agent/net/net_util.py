from functools import partial, wraps
from ...lib import optimizer, util
import os
import torch
import torch.nn as nn

# register custom torch.optim
setattr(torch.optim, 'GlobalAdam', optimizer.GlobalAdam)
setattr(torch.optim, 'GlobalRMSprop', optimizer.GlobalRMSprop)
setattr(torch.optim, 'Lookahead', optimizer.Lookahead)
setattr(torch.optim, 'RAdam', optimizer.RAdam)


class NoOpLRScheduler:
  '''Symbolic LRScheduler class for API consistency'''

  def __init__(self, optim):
    self.optim = optim

  def step(self, epoch=None):
    pass

  def get_lr(self):
    if hasattr(self.optim, 'defaults'):
      return self.optim.defaults['lr']
    else:  # TODO retrieve lr more generally
      return self.optim.param_groups[0]['lr']


def build_fc_model(dims, activation=None):
  '''Build a full-connected model by interleaving nn.Linear and activation_fn'''
  assert len(dims) >= 2, 'dims need to at least contain input, output'
  # shift dims and make pairs of (in, out) dims per layer
  dim_pairs = list(zip(dims[:-1], dims[1:]))
  layers = []
  for in_d, out_d in dim_pairs:
    layers.append(nn.Linear(in_d, out_d))
    if activation is not None:
      layers.append(get_activation_fn(activation))
  model = nn.Sequential(*layers)
  return model


def get_nn_name(uncased_name):
  '''Helper to get the proper name in PyTorch nn given a case-insensitive name'''
  for nn_name in nn.__dict__:
    if uncased_name.lower() == nn_name.lower():
      return nn_name
  raise ValueError(f'Name {uncased_name} not found in {nn.__dict__}')


def get_activation_fn(activation):
  '''Helper to generate activation function layers for net'''
  ActivationClass = getattr(nn, get_nn_name(activation))
  return ActivationClass()


def omit_key(class_spec, omit_keys):
  class_kwargs = {}
  for key in class_spec:
    if key not in omit_keys:
      class_kwargs[key] = class_spec[key]

  return class_kwargs


def get_loss_fn(cls, loss_spec):
  '''Helper to parse loss param and construct loss_fn for net'''
  LossClass = getattr(nn, get_nn_name(loss_spec['name']))

  loss_spec = omit_key(loss_spec, ['name'])
  loss_fn = LossClass(**loss_spec)
  return loss_fn


def get_lr_scheduler(optim, lr_scheduler_spec):
  '''Helper to parse lr_scheduler param and construct Pytorch optim.lr_scheduler'''
  if len(lr_scheduler_spec) == 0:
    lr_scheduler = NoOpLRScheduler(optim)
  elif lr_scheduler_spec['name'] == 'LinearToZero':
    LRSchedulerClass = getattr(torch.optim.lr_scheduler, 'LambdaLR')
    frame = float(lr_scheduler_spec['frame'])
    lr_scheduler = LRSchedulerClass(optim, lr_lambda=lambda x: 1 - x / frame)
  else:
    LRSchedulerClass = getattr(torch.optim.lr_scheduler,
                               lr_scheduler_spec['name'])
    lr_scheduler_spec = omit_key(lr_scheduler_spec, ['name'])
    lr_scheduler = LRSchedulerClass(optim, **lr_scheduler_spec)
  return lr_scheduler


def get_optim(net, optim_spec):
  '''Helper to parse optim param and construct optim for net'''
  OptimClass = getattr(torch.optim, optim_spec['name'])
  optim_spec = omit_key(optim_spec, ['name'])
  if torch.is_tensor(net):  # for non-net tensor variable
    optim = OptimClass([net], **optim_spec)
  else:
    optim = OptimClass(net.parameters(), **optim_spec)
  return optim


def get_policy_out_dim(body):
  '''Helper method to construct the policy network out_dim for a body according to is_discrete, action_type'''
  action_dim = body.action_dim
  if body.is_discrete:
    if body.action_type == 'multi_discrete':
      policy_out_dim = action_dim
    else:
      policy_out_dim = action_dim
  else:
    if action_dim == 1:  # single action, use [loc, scale]
      policy_out_dim = 2
    else:  # multi-action, use [locs], [scales]
      policy_out_dim = [action_dim, action_dim]
  return policy_out_dim


def get_out_dim(body, add_critic=False):
  '''Construct the NetClass out_dim for a body according to is_discrete, action_type, and whether to add a critic unit'''
  policy_out_dim = get_policy_out_dim(body)
  if add_critic:
    if isinstance(policy_out_dim, list):
      out_dim = policy_out_dim + [1]
    else:
      out_dim = [policy_out_dim, 1]
  else:
    out_dim = policy_out_dim
  return out_dim


def init_layers(net, init_fn_name):
  '''Primary method to initialize the weights of the layers of a network'''
  if init_fn_name is None:
    return

  # get nonlinearity
  nonlinearity = get_nn_name(net.hid_layers_activation).lower()
  if nonlinearity == 'leakyrelu':
    nonlinearity = 'leaky_relu'  # guard name

  # get init_fn and add arguments depending on nonlinearity
  init_fn = getattr(nn.init, init_fn_name)
  if 'kaiming' in init_fn_name:  # has 'nonlinearity' as arg
    assert nonlinearity in [
        'relu', 'leaky_relu'
    ], f'Kaiming initialization not supported for {nonlinearity}'
    init_fn = partial(init_fn, nonlinearity=nonlinearity)
  elif 'orthogonal' in init_fn_name or 'xavier' in init_fn_name:  # has 'gain' as arg
    gain = nn.init.calculate_gain(nonlinearity)
    init_fn = partial(init_fn, gain=gain)
  else:
    pass

  # finally, apply init_params to each layer in its modules
  net.apply(partial(init_params, init_fn=init_fn))


def init_params(module, init_fn):
  '''Initialize module's weights using init_fn, and biases to 0.0'''
  bias_init = 0.0
  classname = util.get_class_name(module)
  if 'Net' in classname:  # skip if it's a net, not pytorch layer
    pass
  elif classname == 'BatchNorm2d':
    pass  # can't init BatchNorm2d
  elif any(k in classname for k in ('Conv', 'Linear')):
    init_fn(module.weight)
    nn.init.constant_(module.bias, bias_init)
  elif 'GRU' in classname:
    for name, param in module.named_parameters():
      if 'weight' in name:
        init_fn(param)
      elif 'bias' in name:
        nn.init.constant_(param, bias_init)
  else:
    pass


# params methods


def save(net, model_path):
  '''Save model weights to path'''
  torch.save(net.state_dict(), util.smart_path(model_path))


def copy(src_net, tar_net):
  '''Copy model weights from src to target'''
  tar_net.load_state_dict(src_net.state_dict())


def polyak_update(src_net, tar_net, old_ratio=0.5):
  '''
    Polyak weight update to update a target tar_net, retain old weights by its ratio, i.e.
    target <- old_ratio * source + (1 - old_ratio) * target
    '''
  for src_param, tar_param in zip(src_net.parameters(), tar_net.parameters()):
    tar_param.data.copy_(old_ratio * src_param.data +
                         (1.0 - old_ratio) * tar_param.data)


def to_check_train_step():
  '''Condition for running assert_trained'''
  return os.environ.get('PY_ENV') == 'test' or util.get_lab_mode() == 'dev'


def dev_check_train_step(fn):
  '''
    Decorator to check if net.train_step actually updates the network weights properly
    Triggers only if to_check_train_step is True (dev/test mode)
    @example

    @net_util.dev_check_train_step
    def train_step(self, ...):
        ...
    '''

  @wraps(fn)
  def check_fn(*args, **kwargs):
    if not to_check_train_step():
      return fn(*args, **kwargs)

    net = args[0]  # first arg self
    # get pre-update parameters to compare
    pre_params = [param.clone() for param in net.parameters()]

    # run train_step, get loss
    loss = fn(*args, **kwargs)
    assert not torch.isnan(loss).any(), loss

    # get post-update parameters to compare
    post_params = [param.clone() for param in net.parameters()]
    if loss == 0.0:
      # if loss is 0, there should be no updates
      # TODO if without momentum, parameters should not change too
      for p_name, param in net.named_parameters():
        assert param.grad.norm() == 0
    else:
      # check parameter updates
      try:
        assert not all(
            torch.equal(w1, w2) for w1, w2 in zip(pre_params, post_params)
        ), f'Model parameter is not updated in train_step(), check if your tensor is detached from graph. Loss: {loss:g}'
      except Exception as e:
        # logger.error(e)
        if os.environ.get('PY_ENV') == 'test':
          # raise error if in unit test
          raise (e)

      # check grad norms
      min_norm, max_norm = 0.0, 1e5
      for p_name, param in net.named_parameters():
        try:
          grad_norm = param.grad.norm()
          assert min_norm < grad_norm < max_norm, f'Gradient norm for {p_name} is {grad_norm:g}, fails the extreme value check {min_norm} < grad_norm < {max_norm}. Loss: {loss:g}. Check your network and loss computation.'
        except Exception as e:
          # logger.warning(e)
          pass
    # store grad norms for debugging
    net.store_grad_norms()
    return loss

  return check_fn


def get_grad_norms(algorithm):
  '''Gather all the net's grad norms of an algorithm for debugging'''
  grad_norms = []
  for net_name in algorithm.net_names:
    net = getattr(algorithm, net_name)
    if net.grad_norms is not None:
      grad_norms.extend(net.grad_norms)
  return grad_norms
