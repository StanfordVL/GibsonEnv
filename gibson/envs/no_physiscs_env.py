from gibson.core.physics.robot_locomotors import Turtlebot
from gibson.envs.mobile_robots_env import TurtlebotNavigateEnv
from gibson.envs.mobile_robots_env import tracking_camera


class TurtlebotNavigateNoPhysicsEnv(TurtlebotNavigateEnv):
    def __init__(self, config, gpu_idx=0):
        TurtlebotNavigateEnv.__init__(self, config, gravity=0.0, collision_enabled=False)
