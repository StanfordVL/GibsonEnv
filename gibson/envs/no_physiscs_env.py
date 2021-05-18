from gibson.envs.mobile_robots_env import TurtlebotNavigateEnv


class TurtlebotNavigateNoPhysicsEnv(TurtlebotNavigateEnv):
    def __init__(self, config, gpu_idx=0):
        TurtlebotNavigateEnv.__init__(self, config, gravity=0.0, collision_enabled=False)
