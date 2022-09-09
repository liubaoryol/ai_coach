import abc


class SimulatorAgent:
  __metaclass__ = abc.ABCMeta

  def __init__(self, has_mind: bool, has_policy: bool) -> None:
    self.bool_mind = has_mind
    self.bool_policy = has_policy

  def has_mind(self):
    return self.bool_mind

  def has_policy(self):
    return self.bool_policy

  @abc.abstractmethod
  def init_latent(self, tup_states):
    raise NotImplementedError

  @abc.abstractmethod
  def get_current_latent(self):
    raise NotImplementedError

  @abc.abstractmethod
  def get_action(self, tup_states):
    raise NotImplementedError

  @abc.abstractmethod
  def set_latent(self, latent):
    'to set latent manually'
    raise NotImplementedError

  @abc.abstractmethod
  def set_action(self, action):
    'to set what to do as next actions manually'
    raise NotImplementedError

  @abc.abstractmethod
  def update_mental_state(self, tup_cur_state, tup_actions, tup_nxt_state):
    raise NotImplementedError


class InteractiveAgent(SimulatorAgent):
  def __init__(self, start_latent=None) -> None:
    super().__init__(has_mind=False, has_policy=False)
    self.current_latent = None
    self.start_latent = start_latent
    self.action_queue = []

  def init_latent(self, tup_states):
    self.current_latent = self.start_latent

  def get_current_latent(self):
    return self.current_latent

  def get_action(self, tup_states):
    if len(self.action_queue) == 0:
      return None

    return self.action_queue.pop()

  def set_latent(self, latent):
    self.current_latent = latent

  def set_action(self, action):
    self.action_queue = [action]

  def update_mental_state(self, tup_cur_state, tup_actions, tup_nxt_state):
    'do nothing'
    pass
