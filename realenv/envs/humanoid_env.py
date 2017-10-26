from realenv.envs.env_modalities import CameraRobotEnv, SensorRobotEnv


class HumanoidEnv:
    def __init__(self):
        self.robot = Humanoid()
        PhysicsExtendedEnv.__init__(self, self.robot)
        self.electricity_cost  = 4.25*PhysicsExtendedEnv.electricity_cost
        self.stall_torque_cost = 4.25*PhysicsExtendedEnv.stall_torque_cost



class HumanoidCameraEnv(HumanoidEnv, CameraRobotEnv):
    pass

class HumanoidSensorEnv(HumanoidEnv, SensorRobotEnv):
    pass
