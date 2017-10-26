
from realenv.envs.env_modalities import CameraRobotEnv, SensorRobotEnv


class AntEnv:
    def __init__(self):
        self.robot = Ant()
        PhysicsExtendedEnv.__init__(self, self.robot)


class AntCameraEnv(AntEnv, CameraRobotEnv):
    pass

class AntSensorEnv(AntEnv, SensorRobotEnv):
    pass




