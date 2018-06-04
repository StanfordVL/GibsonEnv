from gibson.envs.simple_env import RenderAllViewEnv 
from gibson.utils.play import simple_play
import argparse
import os

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'play', 'play_points_rendering.yaml')
print(config_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()

    env = RenderAllViewEnv(config = args.config)
    print(env.config)
    simple_play(env, zoom=4)
