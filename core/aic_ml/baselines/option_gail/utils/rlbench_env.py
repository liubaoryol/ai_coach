import os
import numpy as np
from inspect import getmembers, isclass
from rlbench import tasks
from rlbench.environment import SUPPORTED_ROBOTS
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.environment import Environment as RLEnvironment
from rlbench.task_environment import _DT, Quaternion
from pyrep.backend.utils import suppress_std_out_and_err
import random as rnd
import torch


def get_named_class(class_name: str, model):
  all_class_dict = {}
  for o in getmembers(model):
    if isclass(o[1]):
      all_class_dict[o[0]] = o[1]

  if class_name not in all_class_dict:
    raise NotImplementedError(
        f"No class {class_name} found in {model.__name__} !")
  return all_class_dict[class_name]


class RLBenchEnv(object):
  ROBOT_NAME = SUPPORTED_ROBOTS.keys()
  OBSERVATION_MODE = ("state", )
  ACTION_MODE = {
      "joint velocity": ArmActionMode.ABS_JOINT_VELOCITY,
      "delta joint velocity": ArmActionMode.DELTA_JOINT_VELOCITY,
      "joint position": ArmActionMode.ABS_JOINT_POSITION,
      "delta joint position": ArmActionMode.DELTA_JOINT_POSITION,
      "effector position": ArmActionMode.ABS_EE_POSE_WORLD_FRAME,
      "delta effector position": ArmActionMode.DELTA_EE_POSE_WORLD_FRAME
  }

  def __init__(self,
               task_name: str = "PlaceHangerOnRack",
               observation_mode: str = "state",
               action_mode: str = "delta joint position",
               robot_name: str = "panda"):
    self._task_name = task_name
    self._observation_mode = observation_mode
    self._action_mode = action_mode
    self._task_name = task_name
    self._robot_name = robot_name

    self._observation_config = ObservationConfig(
        left_shoulder_camera=CameraConfig(image_size=(256, 256)),
        right_shoulder_camera=CameraConfig(image_size=(256, 256)),
        wrist_camera=CameraConfig(image_size=(256, 256)))

    self._observation_config.set_all_low_dim(True)
    self._observation_config.set_all_high_dim(False)

    self._action_config = ActionMode(RLBenchEnv.ACTION_MODE[self._action_mode])

    self.max_step = 256
    self.env = None
    self.task = None

  def init(self, display=False):
    with suppress_std_out_and_err():
      self.env = RLEnvironment(action_mode=self._action_config,
                               obs_config=self._observation_config,
                               headless=not display,
                               robot_configuration=self._robot_name,
                               static_positions=False)
      self.env.launch()
      self.task = self.env.get_task(get_named_class(self._task_name, tasks))
    return self

  def __del__(self):
    del self.task
    if self.env is not None:
      self.env.shutdown()
    del self.env

  def reset(self, random: bool = False):
    random = True
    self.task._static_positions = not random
    if not random:
      np.random.seed(0)
    descriptions, obs = self.task.reset()
    self._i_step = 0
    return obs.get_low_dim_data()

  def step(self, a):
    a = a.copy()
    a[-1] += 0.5
    obs, reward, terminate = self.task.step(a)
    self._i_step += 1
    return obs.get_low_dim_data(
    ), reward, self._i_step >= self.max_step or terminate

  def state_action_size(self):
    if self.task is not None:
      dim_a = self.env.action_size
      dim_s = len(self.task.reset()[1].get_low_dim_data())
    else:
      with suppress_std_out_and_err():
        env = RLEnvironment(action_mode=self._action_config,
                            obs_config=self._observation_config,
                            headless=True,
                            robot_configuration=self._robot_name)
        dim_a = env.action_size
        env.launch()
        task = env.get_task(get_named_class(self._task_name, tasks))
        dim_s = len(task.reset()[1].get_low_dim_data())
        del task
        env.shutdown()
        del env
    return dim_s, dim_a

  def is_discrete_state_action(self):
    return False, False
