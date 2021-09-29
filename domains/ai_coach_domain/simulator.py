import abc
from tqdm import tqdm
from typing import Mapping, Hashable, Callable


class Simulator():
  __metaclass__ = abc.ABCMeta

  def __init__(self, id: Hashable) -> None:
    self.id = id
    # self.timer = None  # type: Timer
    # self.lock = Lock()
    self.history = []

    self.max_steps = 500  # make sure game ending
    self.current_step = 0

    self.cb_renderer = None  # type: Callable
    self.cb_upon_game_end = None  # type: Callable[[Hashable],None]

  # TODO: connect simulator with Agent object
  @abc.abstractmethod
  def take_a_step(self, map_agent_2_action: Mapping[Hashable,
                                                    Hashable]) -> None:
    # '''
    # map_agent_2_action: map from each agent to its current action to take.
    #                     For agents with no action assigned, the action is
    #                     automatically selected by the policy
    # '''
    self.current_step += 1

  @abc.abstractmethod
  def event_input(self, agent: Hashable, event_type: Hashable, value=None):
    raise NotImplementedError

  @abc.abstractmethod
  def get_joint_action(self) -> Mapping[Hashable, Hashable]:
    raise NotImplementedError

  @abc.abstractmethod
  def reset_game(self):
    self.current_step = 0
    self.history = []

  @abc.abstractmethod
  def get_env_info(self):
    raise NotImplementedError

  @abc.abstractmethod
  def get_num_agents(self):
    raise NotImplementedError

  def run_simulation(self, num_iter: int, file_name_prefix, *args, **kwargs):
    for idx in tqdm(range(num_iter)):
      while not self.is_finished():
        map_agent_2_action = self.get_joint_action()
        self.take_a_step(map_agent_2_action)
      file_name = file_name_prefix + "%d.txt" % (idx, )
      self.save_history(file_name, *args, **kwargs)
      self.reset_game()

  @classmethod
  def read_file(cls, file_name):
    pass

  @abc.abstractmethod
  def save_history(self, file_name, *args, **kwargs):
    raise NotImplementedError

  @abc.abstractmethod
  def get_changed_objects(self):
    raise NotImplementedError

  @abc.abstractmethod
  def is_finished(self) -> bool:
    if self.current_step > self.max_steps:
      return True
    return False

  @abc.abstractmethod
  def init_game(self, *args, **kwargs):
    raise NotImplementedError

  def get_current_step(self):
    return self.current_step

  # def __del__(self):
  #   self.stop_game()

  # def stop_game(self):
  #   with self.lock:
  #     if self.timer is not None:
  #       self.timer.cancel()

  # @abc.abstractmethod
  # def _get_policy_actions(self,
  #                         agents: Sequence[Hashable]) -> Sequence[Hashable]:
  #   raise NotImplementedError

  # def run_periodic_actions(self,
  #                          step_period: float = 0.5
  #                          ):  # TODO: think about better logic..
  #   # is finished
  #   if self.is_finished():
  #     return

  #   with self.lock:
  #     map_agent_2_action = self.get_action()
  #     self.take_a_step(map_agent_2_action)

  #     # render the game
  #     objects_to_render = self.get_changed_objects()
  #     if self.cb_renderer:
  #       self.cb_renderer(objects_to_render, self.id)
  #     time.sleep(0)

  #     timer = self.timer = Timer(step_period, self.run_periodic_actions,
  #                                [step_period])
  #     timer.start()
