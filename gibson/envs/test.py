from gibson.envs.husky_env import *
import os


os.environ['TEST_ENV'] = "True"

## Train Husky Navigate RGBD
env = HuskyNavigateEnv(human=True, is_discrete=True, mode="RGB", gpu_count=1, use_filler=True, resolution="MID")
env.reset()
## Train Husky Navigate SENSOR
env = HuskyNavigateEnv(human=True, is_discrete=True, mode="SENSOR", gpu_count=1, use_filler=True, resolution="MID")
env.reset()

## Play Husky Sensor
env = HuskyNavigateEnv(human=True, timestep=timestep, frame_skip=frame_skip, mode="SENSOR", is_discrete = True)
env.reset()
## PLay Husky RGBD
env = HuskyNavigateEnv(human=True, timestep=timestep, frame_skip=frame_skip, mode="RGBD", is_discrete = True)
env.reset()


print("All tests passed")