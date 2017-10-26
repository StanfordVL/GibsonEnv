from realenv.envs.env_modalities import CameraRobotEnv, SensorRobotEnv


class HuskyEnv:
    def __init__(self):
        self.robot = Husky()
        PhysicsExtendedEnv.__init__(self, self.robot)


class HuskyCameraEnv(HuskyEnv, CameraRobotEnv):
    pass

class HuskySensorEnv(HuskyEnv, SensorRobotEnv):
    pass





