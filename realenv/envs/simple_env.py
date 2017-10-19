import gym
from gym import error, spaces, utils
from gym.utils import seeding
import realenv
from realenv.main import RealEnv
from realenv.core.engine import Engine
from realenv.core.render.profiler import Profiler
from realenv.core.scoreboard.realtime_plot import MPRewardDisplayer, RewardDisplayer
import numpy as np
import zmq
import time
import os
import random
import cv2


class SimpleEnv(RealEnv):
  """Bare bone room environment with no addtional constraint (disturbance, friction, gravity change)
  """
  def __init__(self, human=False, debug=True, model_id="11HB6XZSh1Q", scale_up = 1):
    self.debug_mode = debug
    file_dir = os.path.dirname(__file__)

    self.model_id  = model_id
    self.state_old = None
    self.scale_up  = scale_up

    self.engine = Engine(model_id, human, debug)

    self.r_visuals, self.r_physics, self.p_channel = self.engine.setup_all()
    if self.debug_mode:
      self.r_displayer = RewardDisplayer()

  def testShow3D(self):
    return

  def _step(self, action):
    try:
      with Profiler("Physics to screen"):
        if not self.debug_mode:
          pose, state = self.r_physics.renderOffScreen(action)
        else:
          pose, state = self.r_physics.renderToScreen(action)

      if not self.state_old:
        reward = 0
      else:
        reward = 5 * (self.state_old['distance_to_target'] - state['distance_to_target'])
      #self.r_displayer.add_reward(reward)
      self.state_old = state

      with Profiler("Render to screen"):
        if not self.debug_mode:
          visuals = self.r_visuals.renderOffScreen(pose)
        else:
          visuals = self.r_visuals.renderToScreen(pose)

        done = False

      return visuals, reward, done, dict(state_old=self.state_old['distance_to_target'], state_new=state['distance_to_target'])
    except Exception as e:
      self._end()
      raise(e)

  def _reset(self):
    return

  def _render(self, mode='human', close=False):
    return

  def _end(self):
    ## TODO (hzyjerry): this does not kill cleanly
    ## to reproduce bug, set human = false, debug_mode = false
    self.engine.cleanUp()
    return


if __name__ == "__main__":
  env = SimpleEnv()
  t_start = time.time()
  r_current = 0
  try:
    while True:
      t0 = time.time()
      img, reward = env._step({})
      t1 = time.time()
      t = t1-t0
      r_current = r_current + 1
      print('(Round %d) fps %.3f total time %.3f' %(r_current, 1/t, time.time() - t0))
  except KeyboardInterrupt:
    env._end()
    print("Program finished")
  '''
  r_displayer = MPRewardDisplayer()
  for i in range(10000):
      num = random.random() * 100 - 30
      r_displayer.add_reward(num)
      if i % 40 == 0:
          r_displayer.reset()
  '''
