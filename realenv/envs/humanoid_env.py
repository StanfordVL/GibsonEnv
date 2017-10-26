from realenv.envs.env_modalities import CameraRobotEnv, SensorRobotEnv
from realenv.core.physics.robot_locomotors import Humanoid, Ant, Husky
import gym

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
    def __init__(self, human=True, enable_sensors=False):
        HumanoidEnv.__init__(self)
        CameraRobotEnv.__init__(self, human, enable_sensors=enable_sensors)

class HumanoidSensorEnv(HumanoidEnv, SensorRobotEnv):
    def __init__(self, human=True):
        HumanoidEnv.__init__(self)
        SensorRobotEnv.__init__(self, human)
