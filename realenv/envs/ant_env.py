from realenv.envs.env_modalities import CameraRobotEnv, SensorRobotEnv
from realenv.core.physics.robot_locomotors import Ant
import numpy as np

ANT_TIMESTEP  = 1.0/(4 * 22)
ANT_FRAMESKIP = 4

class AntEnv:
    def __init__(self, is_discrete=False, mode="SENSOR"):
        self.is_discrete = is_discrete
        self.robot = Ant(is_discrete, mode)
        self.physicsClientId=-1

class AntCameraEnv(AntEnv, CameraRobotEnv):
    def __init__(self, human=True, timestep=ANT_TIMESTEP, 
        frame_skip=ANT_FRAMESKIP, enable_sensors=False,
        is_discrete=False, mode="GREY"):
        self.human = human
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.enable_sensors = enable_sensors
        AntEnv.__init__(self, is_discrete, mode=mode)
        CameraRobotEnv.__init__(self)

        #self.tracking_camera['pitch'] = -45 ## stairs
        yaw = 90     ## demo: living room
        #yaw = 30    ## demo: kitchen
        offset = 0.5
        distance = 1.2 ## living room
        #self.tracking_camera['yaw'] = 90     ## demo: stairs

        self.tracking_camera['yaw'] = yaw   ## living roon
        self.tracking_camera['pitch'] = -10
        
        self.tracking_camera['distance'] = distance
        self.tracking_camera['z_offset'] = offset

    def _step(self, action):
        visuals, sensor_reward, done, sensor_meta = CameraRobotEnv._step(self, action)
        return visuals, sensor_reward, done, sensor_meta


class AntSensorEnv(AntEnv, SensorRobotEnv):
    def __init__(self, human=True, timestep=ANT_TIMESTEP, 
        frame_skip=ANT_FRAMESKIP, is_discrete=False):
        self.human = human
        self.timestep = timestep
        self.frame_skip = frame_skip
        AntEnv.__init__(self, is_discrete)
        SensorRobotEnv.__init__(self)

        #self.tracking_camera['pitch'] = -45 ## stairs
        yaw = 90     ## demo: living room
        #yaw = 30    ## demo: kitchen
        offset = 0.5
        distance = 1.2 ## living room
        #self.tracking_camera['yaw'] = 90     ## demo: stairs
        
        self.tracking_camera['yaw'] = yaw   ## living roon
        self.tracking_camera['pitch'] = -10
        
        self.tracking_camera['distance'] = distance
        self.tracking_camera['z_offset'] = offset




