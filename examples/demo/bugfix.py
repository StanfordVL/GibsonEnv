import os
from gibson.envs.husky_env import HuskyNavigateEnv

env = HuskyNavigateEnv(gpu_count=1, config=os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'play', 'play_ant_camera.yaml'))
env.reset()