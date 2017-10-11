import gym
from gym import error, spaces, utils
from gym.utils import seeding
import subprocess

class SimpleEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    cmd_channel = "./channels/depth_render/depth_render --datapath data -m 11HB6XZSh1Q"
    cmd_physics = "python show_3d2.py --datapath ../data/ --idx 10"
    cmd_render  = ""

    #self.p_channel = subprocess.Popen(cmd_channel.split(), stdout=subprocess.PIPE)
    #self.p_physics = subprocess.Popen()
    #self.p_render  = subprocess.Popen()

  def testShow3D(self):
    

  def _step(self, action):
    return

  def _reset(self):
    return

  def _render(self, mode='human', close=False):
    return
    
  def _end(self):
    #self.p_channel.kill()
    #self.p_physics.kill()
    #self.p_render.kill()
    return


if __name__ == "__main__":
  env = SimpleEnv()
  env.