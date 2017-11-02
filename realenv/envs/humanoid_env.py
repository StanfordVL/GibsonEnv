from realenv.envs.env_modalities import CameraRobotEnv, SensorRobotEnv
from realenv.core.physics.robot_locomotors import Humanoid
import gym

HUMANOID_TIMESTEP  = 1.0/(4 * 22)
HUMANOID_FRAMESKIP = 4

class HumanoidEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }
    timestep   = 1/(20 * 4)
    frame_skip = 20
    def __init__(self, mode="SENSOR"):
        self.robot = Humanoid(mode)
        self.physicsClientId=-1
        self.electricity_cost  = 4.25*SensorRobotEnv.electricity_cost
        self.stall_torque_cost = 4.25*SensorRobotEnv.stall_torque_cost


class HumanoidCameraEnv(HumanoidEnv, CameraRobotEnv):
    def __init__(self, human=True, timestep=HUMANOID_TIMESTEP, 
        frame_skip=HUMANOID_FRAMESKIP, enable_sensors=False, mode='RGBD'):
        self.human = human
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.enable_sensors = enable_sensors
        HumanoidEnv.__init__(self, mode)
        CameraRobotEnv.__init__(self)
        #self.tracking_camera['yaw'] = 30    ## living room
        #self.tracking_camera['distance'] = 1.5
        #self.tracking_camera['pitch'] = -45 ## stairs

        #distance=2.5 ## demo: living room ,kitchen
        self.tracking_camera['distance'] = 1.3   ## demo: stairs
        self.tracking_camera['pitch'] = -35 ## stairs

        #yaw = 0     ## demo: living room
        #yaw = 30    ## demo: kitchen
        self.tracking_camera['yaw'] = -60     ## demo: stairs


class HumanoidSensorEnv(HumanoidEnv, SensorRobotEnv):
    def __init__(self, human=True, timestep=HUMANOID_TIMESTEP, 
        frame_skip=HUMANOID_FRAMESKIP, enable_sensors=False):
        self.human = human
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.enable_sensors = enable_sensors
        HumanoidEnv.__init__(self)
        SensorRobotEnv.__init__(self)
        self.tracking_camera['distance'] = 1.3   ## demo: stairs
        self.tracking_camera['pitch'] = -35 ## stairs

        #yaw = 0     ## demo: living room
        #yaw = 30    ## demo: kitchen
        self.tracking_camera['yaw'] = -60     ## demo: stairs