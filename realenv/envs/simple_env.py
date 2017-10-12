import gym
from gym import error, spaces, utils
from gym.utils import seeding
import subprocess
from render.datasets import ViewDataSet3D
from render.show_3d2 import PCRenderer, sync_coords
from physics.render_physics import PhysRenderer
import numpy as np
import zmq
import time

class SimpleEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    cmd_channel = "./channels/depth_render/depth_render --datapath data -m 11HB6XZSh1Q"
    cmd_physics = "python show_3d2.py --datapath ../data/ --idx 10"
    cmd_render  = ""
    self.datapath = "data"
    self.model_id = "11HB6XZSh1Q"
    #self.p_channel = subprocess.Popen(cmd_channel.split(), stdout=subprocess.PIPE)
    #self.p_physics = subprocess.Popen()
    #self.p_render  = subprocess.Popen()
    self.r_physics = self._setupPhysics()
    self.r_visuals = self._setupVisuals()

    pose_init = self.r_visuals.renderOffScreenInitialPose()
    self.r_physics.initialize(pose_init)
    self.r_visuals.renderToScreenSetup()
    
  def _setupVisuals(self):
    d = ViewDataSet3D(root=self.datapath, transform = np.array, mist_transform = np.array, seqlen = 2, off_3d = False, train = False)

    scene_dict = dict(zip(d.scenes, range(len(d.scenes))))
    if not self.model_id in scene_dict.keys():
        print("model not found")
    else:
        scene_id = scene_dict[self.model_id]
    uuids, rts = d.get_scene_info(scene_id)
    #print(uuids, rts)
    targets = []
    sources = []
    source_depths = []
    poses = []
    for k,v in uuids:
        print(k,v)
        data = d[v]
        source = data[0][0]
        target = data[1]
        target_depth = data[3]
        source_depth = data[2][0]
        pose = data[-1][0].numpy()
        targets.append(target)
        poses.append(pose)
        sources.append(target)
        source_depths.append(target_depth)
    print('target', poses, poses[0])
    #print('no.1 pose', poses, poses[1])
    # print(source_depth)
    print(sources[0].shape, source_depths[0].shape)
    context_mist = zmq.Context()
    print("Connecting to hello world server...")
    socket_mist = context_mist.socket(zmq.REQ)
    socket_mist.connect("tcp://localhost:5555")
    
    sync_coords()
    
    renderer = PCRenderer(5556, sources, source_depths, target, rts)
    return renderer

  def _setupPhysics(self):
    framePerSec = 13
    renderer = PhysRenderer(self.datapath, self.model_id, framePerSec)
    #renderer.renderToScreen()
    return renderer

  def testShow3D(self):
    return

  def _step(self, action):
    #renderer.renderToScreen(sources, source_depths, poses, model, target, target_depth, rts)
    pose = self.r_physics.renderOffScreen(action)
    return self.r_visuals.renderToScreen(pose)

  def _reset(self):
    return

  def _render(self, mode='human', close=False):
    return
    
  def _end(self):
    #self.p_channel.kill()
    #self.p_physics.kill()
    #self.r_visuals.kill()
    return


if __name__ == "__main__":
  env = SimpleEnv()
  while True:
    t0 = time.time()
    img = env._step({})
    t1 = time.time()
    t = t1-t0
    print('fps', 1/t, np.mean(img))