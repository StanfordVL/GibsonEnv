import gym
from gym import error, spaces, utils
from gym.utils import seeding
import realenv
from realenv.core.engine import Engine
from realenv.core.render.profiler import Profiler
from realenv.core.scoreboard.realtime_plot import MPRewardDisplayer, RewardDisplayer
from realenv.data.datasets import get_model_path
import numpy as np
import zmq
import time
import os
import random
import cv2
import gym
import traceback


class SimpleEnv(gym.Env):
  """Bare bone room environment with no addtional constraint (disturbance, friction, gravity change)
  """
  def __init__(self, human=False, debug=True, scale_up = 1):
    self.debug_mode = debug
    self.human      = human
    self.model_id   = get_model_path()[1]
    self.scale_up   = scale_up
    self.engine = None
    file_dir = os.path.dirname(__file__)

  def _engine_setup(self):
    self.r_visuals, self.r_physics, self.p_channel = self.engine.setup_all()
    if self.debug_mode:
      self.r_displayer = RewardDisplayer()

  def _step(self, action):
    try:
      with Profiler("Physics to screen"):
        obs, reward, done, meta = self.r_physics.step(action)

      pose = [meta['eye_pos'], meta['eye_quat']]

      with Profiler("Render to screen"):
        if not self.debug_mode:
          visuals = self.r_visuals.renderOffScreen(pose)
        else:

          visuals = self.r_visuals.renderToScreen(pose)

      return visuals, reward , done, {}
    except Exception as e:
      self._end()
      traceback.print_exc()
      raise(e)

  def _reset(self):
    self.r_physics.reset()
    return

  def _render(self, mode='human', close=False):
    return

  def _end(self):
    ## TODO (hzyjerry): this does not kill cleanly
    ## to reproduce bug, set human = false, debug_mode = false
    self.engine.cleanUp()
    return

  @property
  def action_space(self):
    return self.r_physics.action_space


class HumanoidWalkingEnv(SimpleEnv):
  def __init__(self):
    SimpleEnv.__init__(self)
    self.engine = Engine(self.model_id, self.human, self.debug_mode, "PhysicsHumanoidWalkingEnv-v0")
    self._engine_setup()


class AntWalkingEnv(SimpleEnv):
  def __init__(self):
    SimpleEnv.__init__(self)
    self.engine = Engine(self.model_id, self.human, self.debug_mode, "PhysicsAntWalkingEnv-v0")
    self._engine_setup()


class HuskyWalkingEnv(SimpleEnv):
  def __init__(self):
    SimpleEnv.__init__(self)
    self.engine = Engine(self.model_id, self.human, self.debug_mode, "PhysicsHuskyWalkingEnv-v0")
    self._engine_setup()
