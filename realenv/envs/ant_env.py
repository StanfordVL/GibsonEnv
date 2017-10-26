from realenv.envs.env_modalities import CameraRobotEnv, SensorRobotEnv
from realenv.core.physics.robot_locomotors import Ant


class AntEnv:
    def __init__(self):
        self.robot = Ant()
        PhysicsExtendedEnv.__init__(self, self.robot)


class AntCameraEnv(AntEnv, CameraRobotEnv):
    pass

class AntSensorEnv(AntEnv, SensorRobotEnv):
    pass




