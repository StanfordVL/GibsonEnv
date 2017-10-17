import gym
from gym import error, spaces, utils
from gym.utils import seeding
from realenv.data.datasets import ViewDataSet3D
from realenv.core.render.show_3d2 import PCRenderer, sync_coords
from realenv.core.physics.render_physics import PhysRenderer
from realenv.core.render.profiler import Profiler
from realenv.core.channels.depth_render import run_depth_render
import numpy as np
import zmq
import time
import os
import random
import progressbar
from realenv.core.scoreboard.realtime_plot import MPRewardDisplayer, RewardDisplayer
from multiprocessing import Process

class SimpleEnv(gym.Env):
  def __init__(self, human=False, debug=True, model_id="11HB6XZSh1Q"):
    self.debug_mode = debug
    file_dir = os.path.dirname(__file__)
    
    self.model_id = model_id
    self.dataset  = ViewDataSet3D(transform = np.array, mist_transform = np.array, seqlen = 2, off_3d = False, train = False)

    self.p_channel = Process(target=run_depth_render)
    self.state_old = None


    try:
      self.p_channel.start()
      self.r_visuals = self._setupVisuals()
      pose_init = self.r_visuals.renderOffScreenInitialPose()
      print("initial pose", pose_init)
      self.r_physics = self._setupPhysics(human)
      self.r_physics.initialize(pose_init)
      if self.debug_mode:
        self.r_visuals.renderToScreenSetup()
        self.r_displayer = RewardDisplayer() #MPRewardDisplayer()
        self._setupRewardFunc()
    except Exception as e:
      self._end()
      raise(e)
    

  def _setupRewardFunc(self):
    def _getReward(state_old, state_new):
      if not state_old:
        return 0
      else:
        return 5 * (state_old['distance_to_target'] - state_new['distance_to_target'])
    self.reward_func = _getReward
    
  def _setupVisuals(self):
    scene_dict = dict(zip(self.dataset.scenes, range(len(self.dataset.scenes))))
    if not self.model_id in scene_dict.keys():
        print("model not found")
    else:
        scene_id = scene_dict[self.model_id]
    uuids, rts = self.dataset.get_scene_info(scene_id)
    targets = []
    sources = []
    source_depths = []
    poses = []
    pbar  = progressbar.ProgressBar(widgets=[
                        ' [ Initializing Environment ] ',
                        progressbar.Bar(),
                        ' (', progressbar.ETA(), ') ',
                        ])
    for k,v in pbar(uuids):
        data = self.dataset[v]
        source = data[0][0]
        target = data[1]
        target_depth = data[3]
        source_depth = data[2][0]
        pose = data[-1][0].numpy()
        targets.append(target)
        poses.append(pose)
        sources.append(target)
        source_depths.append(target_depth)
    context_mist = zmq.Context()
    socket_mist = context_mist.socket(zmq.REQ)
    socket_mist.connect("tcp://localhost:5555")
    
    sync_coords()
    
    renderer = PCRenderer(5556, sources, source_depths, target, rts)
    return renderer

  def _setupPhysics(self, human):
    framePerSec = 13
    renderer = PhysRenderer(self.dataset.get_model_obj(), framePerSec, debug = self.debug_mode, human = human)
    return renderer

  def testShow3D(self):
    return

  def _step(self, action):
    try:
      #renderer.renderToScreen(sources, source_depths, poses, model, target, target_depth, rts)
      if not self.debug_mode:
        pose, state = self.r_physics.renderOffScreen(action)
        #reward = random.randrange(-8, 20)
        reward = self.reward_func(self.state_old, state)
        self.state_old = state
        visuals = self.r_visuals.renderOffScreen(pose)
      else:
        with Profiler("Physics to screen"):
          pose, state = self.r_physics.renderOffScreen(action)
        #reward = random.randrange(-8, 20)
        with Profiler("Reward func"):
          reward = self.reward_func(self.state_old, state)
        #self.r_displayer.add_reward(reward)
        self.state_old = state
        #with Profiler("Display reward"):
        
        with Profiler("Render to screen"):
          visuals = self.r_visuals.renderToScreen(pose)

      return visuals, reward
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
    self.p_channel.terminate()
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
