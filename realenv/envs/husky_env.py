from realenv.envs.env_modalities import CameraRobotEnv, SensorRobotEnv
from realenv.core.physics.robot_locomotors import Husky

HUMANOID_TIMESTEP  = 1.0/(4 * 22)
HUMANOID_FRAMESKIP = 4

class HuskyEnv:
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }
    def __init__(self, is_discrete=False):
        self.robot = Husky(is_discrete)
        

class HuskyCameraEnv(HuskyEnv, CameraRobotEnv):
    def __init__(self, human=True, timestep=HUMANOID_TIMESTEP, 
        frame_skip=HUMANOID_FRAMESKIP, enable_sensors=False,
        is_discrete=False):
        HuskyEnv.__init__(self, is_discrete)
        CameraRobotEnv.__init__(self, human, timestep=timestep, 
            frame_skip=frame_skip, enable_sensors=enable_sensors)
        self.tracking_camera['yaw'] = 80
        self.tracking_camera['pitch'] = -10
        self.tracking_camera['distance'] = 1.5
        self.tracking_camera['z_offset'] = 0.5

class HuskySensorEnv(HuskyEnv, SensorRobotEnv):
    def __init__(self, human=True, timestep=HUMANOID_TIMESTEP, 
        frame_skip=HUMANOID_FRAMESKIP, enable_sensors=False,
        is_discrete=False):
        HuskyEnv.__init__(self, is_discrete)
        SensorRobotEnv.__init__(self, human)





