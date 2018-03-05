from gibson.envs.ant_env import AntNavigateEnv
from gibson.utils.play import play
import os

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'ant_navigate.yaml')
print(config_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--resolution', type=str, default="NORMAL")
    args = parser.parse_args()
    env = AntNavigateEnv(is_discrete = True, config = config_file)
    play(env, zoom=4)