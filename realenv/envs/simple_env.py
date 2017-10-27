import gym
from gym import error, spaces, utils
from gym.utils import seeding
import realenv
from realenv.core.engine import Engine
from realenv.core.render.profiler import Profiler
from realenv.core.scoreboard.realtime_plot import MPRewardDisplayer, RewardDisplayer
from realenv.data.datasets import get_model_path
from realenv.core.physics.simple_debug_env import PhysRenderer
from realenv.core.render.show_3d2 import PCRenderer, sync_coords
from realenv.data.datasets import ViewDataSet3D
from multiprocessing import Process
from realenv.core.channels.depth_render import run_depth_render
from realenv.data.datasets import get_model_path
from tqdm import *
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
    self.engine     = None
    self.r_physics  = None
    self.r_visuals  = None
    self.p_channel  = None
    file_dir = os.path.dirname(__file__)

  def configure(self, timestep, frame_skip):
    self.timestep = timestep
    self.frame_skip = frame_skip

  def _step(self, action):
    try:
      with Profiler("Physics to screen"):
        obs, reward, done, meta = self.r_physics.step(action)
      pose = [meta['eye_pos'], meta['eye_quat']]
      print(obs)
      ## Select the nearest points
      all_dist, all_pos = self.r_visuals.rankPosesByDistance(pose)
      top_k = self.r_physics.find_best_k_views(meta['eye_pos'], all_dist, all_pos)
      with Profiler("Render to screen"):
        if not self.debug_mode:
          visuals = self.r_visuals.renderOffScreen(pose, top_k)
        else:
          visuals = self.r_visuals.renderToScreen(pose, top_k)

      return visuals, reward , done, {}
    except Exception as e:
      self._end()
      traceback.print_exc()
      raise(e)

  def _reset(self):
    if self.r_physics:
      self.r_physics.reset()
    else:
      self.r_visuals, self.r_physics, self.p_channel = self.engine.setup_all(self.timestep, self.frame_skip)
      if self.debug_mode:
        self.r_displayer = RewardDisplayer()
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
    

class AntWalkingEnv(SimpleEnv):
  def __init__(self):
    SimpleEnv.__init__(self)
    self.engine = Engine(self.model_id, self.human, self.debug_mode, "PhysicsAntWalkingEnv-v0")
    

class HuskyWalkingEnv(SimpleEnv):
  def __init__(self):
    SimpleEnv.__init__(self)
    self.engine = Engine(self.model_id, self.human, self.debug_mode, "PhysicsHuskyWalkingEnv-v0")
    


########### Old interface: only use for playing ##################


class SimpleDebugEnv(gym.Env):
  def __init__(self, human=False, debug=True, overwrite_fofn=True):
    self.debug_mode = debug
    self.human_mode = human
    file_dir = os.path.dirname(__file__)

    self.model_id = get_model_path()[1]
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
      traceback.print_exc()
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

    print("loading visual scene", self.model_id)
    uuids, rts = self.dataset.get_scene_info(scene_id)
    targets = []
    sources = []
    source_depths = []
    poses = []
    for k,v in tqdm(uuids):
        data = self.dataset[v]
        source = data[0][0]
        target = data[1]
        target_depth = data[3]
        source_depth = data[2][0]
        pose = data[-1][0].numpy()
        targets.append(target)
        print(target)
        poses.append(pose)
        sources.append(target)
        source_depths.append(target_depth)
    context_mist = zmq.Context()
    socket_mist = context_mist.socket(zmq.REQ)
    socket_mist.connect("tcp://localhost:5555")

    sync_coords()

    renderer = PCRenderer(5556, sources, source_depths, target, rts, scale_up=1)
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
        #with Profiler("Physics to screen"):
        if self.human_mode:
          pose, state = self.r_physics.renderToScreen(action)
        else:
          pose, state = self.r_physics.renderOffScreen(action)
        print("Current robot pose", pose)
        #reward = random.randrange(-8, 20)
        #with Profiler("Reward func"):
        reward = self.reward_func(self.state_old, state)
        #self.r_displayer.add_reward(reward)
        self.state_old = state
        #with Profiler("Display reward"):

        #with Profiler("Render to screen"):
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
