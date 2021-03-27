from gibson.envs.no_physiscs_env import TurtlebotNavigateNoPhysicsEnv
from gibson.utils.play import play
import argparse
import os

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'play', 'play_turtlebot_camera_no_physics.yaml')
print(config_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()

    env = TurtlebotNavigateNoPhysicsEnv(config=args.config)
    env.reset()
    while True:
        env.step([0.0, 0.0])

        # Now you can move the robot in space without constraints
        env.robot.set_position([1, 1, 1])

        #env.robot.move_forward(forward=1)
        #env.robot.move_backward(backward=0.2)

        #env.robot.turn_right(delta=0.1)
        #env.robot.turn_left(delta=0.1)
