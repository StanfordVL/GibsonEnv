from realenv.envs.env_modalities import CameraRobotEnv, SensorRobotEnv
from realenv.core.physics.robot_locomotors import Husky


class HuskyEnv:
    def __init__(self):
        self.robot = Husky()
        PhysicsExtendedEnv.__init__(self, self.robot)


class HuskyCameraEnv(HuskyEnv, CameraRobotEnv):
    pass

class HuskySensorEnv(HuskyEnv, SensorRobotEnv):
    pass





