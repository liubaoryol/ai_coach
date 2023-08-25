from abc import ABC, abstractmethod


class Algorithm(ABC):
  '''Abstract Algorithm class to define the API methods'''

  def __init__(self,
               env,
               memory,
               net_kwargs,
               action_pdtype,
               batch_size,
               gamma=0.99,
               training_frequency=1):
    '''
        @param {*} agent is the container for algorithm and related components, and interfaces with env.
        '''
    self.env = env
    self.memory = memory
    self.action_pdtype = action_pdtype
    self.action_policy = 'default'
    self.gamma = gamma
    self.training_frequency = training_frequency
    self.batch_size = batch_size
    self.to_train = 0
    self.shared = False
    self.training_start_step = batch_size

    self.init_algorithm_params()
    self.init_nets(**net_kwargs)

  @abstractmethod
  def init_algorithm_params(self):
    '''Initialize other algorithm parameters'''
    raise NotImplementedError

  @abstractmethod
  def init_nets(self, **kwargs):
    '''Initialize the neural network from the spec'''
    raise NotImplementedError

  def end_init_nets(self):
    '''Checkers and conditional loaders called at the end of init_nets()'''
    # check all nets naming
    assert hasattr(self, 'net_names')
    for net_name in self.net_names:
      assert net_name.endswith(
          'net'
      ), f'Naming convention: net_name must end with "net"; got {net_name}'

  def calc_pdparam(self, x, net=None):
    '''
        To get the pdparam for action policy sampling, do a forward pass of the appropriate net, and pick the correct outputs.
        The pdparam will be the logits for discrete prob. dist., or the mean and std for continuous prob. dist.
        '''
    raise NotImplementedError

  def act(self, state):
    '''Standard act method.'''
    raise NotImplementedError
    return action

  @abstractmethod
  def sample(self):
    '''Samples a batch from memory'''
    raise NotImplementedError
    return batch

  @abstractmethod
  def train(self, learning_steps):
    '''Implement algorithm train, or throw NotImplementedError'''
    raise NotImplementedError

  @abstractmethod
  def update(self):
    '''Implement algorithm update, or throw NotImplementedError'''
    raise NotImplementedError

  # def save(self, ckpt=None):
  #   '''Save net models for algorithm given the required property self.net_names'''
  #   if hasattr(self, 'net_names'):
  #     net_util.save_algorithm(self, ckpt=ckpt)

  # def load(self):
  #   '''Load net models for algorithm given the required property self.net_names'''
  #   if hasattr(self, 'net_names'):
  #     net_util.load_algorithm(self)
  #   # set decayable variables to final values
  #   for k, v in vars(self).items():
  #     if k.endswith('_scheduler') and hasattr(v, 'end_val'):
  #       var_name = k.replace('_scheduler', '')
  #       setattr(self.body, var_name, v.end_val)
