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
    def __init__(self):
        self.robot = Humanoid()
        self.electricity_cost  = 4.25*SensorRobotEnv.electricity_cost
        self.stall_torque_cost = 4.25*SensorRobotEnv.stall_torque_cost


class HumanoidCameraEnv(HumanoidEnv, CameraRobotEnv):
    def __init__(self, human=True, timestep=HUMANOID_TIMESTEP, 
        frame_skip=HUMANOID_FRAMESKIP, enable_sensors=False):
        self.human = human
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.enable_sensors = enable_sensors
        HumanoidEnv.__init__(self)
        CameraRobotEnv.__init__(self)

class HumanoidSensorEnv(HumanoidEnv, SensorRobotEnv):
    def __init__(self, human=True, timestep=HUMANOID_TIMESTEP, 
        frame_skip=HUMANOID_FRAMESKIP, enable_sensors=False):
        self.human = human
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.enable_sensors = enable_sensors
        HumanoidEnv.__init__(self)
        SensorRobotEnv.__init__(self)
