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
        self.physicsClientId=-1
        self.robot = Husky(is_discrete)

    def get_keys_to_action(self):
        return self.robot.keys_to_action
        

class HuskyCameraEnv(HuskyEnv, CameraRobotEnv):
    def __init__(self, human=True, timestep=HUMANOID_TIMESTEP, 
        frame_skip=HUMANOID_FRAMESKIP, enable_sensors=False,
        is_discrete=False):
        self.human = human
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.enable_sensors = enable_sensors
        HuskyEnv.__init__(self, is_discrete)
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

class HuskySensorEnv(HuskyEnv, SensorRobotEnv):
    def __init__(self, human=True, timestep=HUMANOID_TIMESTEP, 
        frame_skip=HUMANOID_FRAMESKIP, enable_sensors=False,
        is_discrete=False):
        self.human = human
        self.timestep = timestep
        self.frame_skip = frame_skip
        HuskyEnv.__init__(self, is_discrete)
        SensorRobotEnv.__init__(self)

        #self.tracking_camera['pitch'] = -45 ## stairs
        yaw = 90     ## demo: living room
        #yaw = 30    ## demo: kitchen
        offset = 0.5
        distance = 0.7 ## living room
        #self.tracking_camera['yaw'] = 90     ## demo: stairs

        
        self.tracking_camera['yaw'] = yaw   ## living roon
        self.tracking_camera['pitch'] = -10
        
        self.tracking_camera['distance'] = distance
        self.tracking_camera['z_offset'] = offset