from gibson.envs.mobile_robots_env import TurtlebotNavigateSpeedControlEnv
from gibson.utils.play import play
import argparse
import os

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'play', 'play_tr_speedcontrol_nonviz.yaml')
print(config_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()

    env = TurtlebotNavigateSpeedControlEnv(config = args.config)
    print(env.config)
    play(env, zoom=4)