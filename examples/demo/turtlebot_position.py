from gibson.envs.mobile_robots_env import TurtlebotNavigateSpeedControlEnv
from gibson.utils.play import play
import argparse
import os
import pybullet as p
import pybullet_data
import numpy as np

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'play', 'tr_position_control.yaml')
print(config_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()

    env = TurtlebotNavigateSpeedControlEnv(config = args.config)

    env.reset()

    vid = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'),
                                              rgbaColor=[1, 0, 0, 0.7])

    p.createMultiBody(baseVisualShapeIndex=vid, baseCollisionShapeIndex=-1,
                      basePosition=env.robot.get_target_position())

    while env.robot.dist_to_target() > 0.2:
        v_signal = min(env.robot.dist_to_target() * 0.2, 0.2)
        print(env.robot.angle_to_target, env.robot.dist_to_target(), env.robot.body_xyz, env.robot.get_target_position())
        omega_signal = (-env.robot.angle_to_target) / 10
        omega_signal = np.clip(omega_signal, -0.02, 0.02)

        obs, _, _, _ = env.step([v_signal, omega_signal])
        #print(obs["nonviz_sensor"])

    for i in range(1000):
        env.step([0, 0])