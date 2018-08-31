
from gibson.envs.ant_env import AntNavigateEnv
from gibson.utils.play import play
import os

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'configs', 'gif', 'play_ant_gif.yaml')
print(config_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()

    env = AntNavigateEnv(config=args.config, gpu_count = 0)
    play(env, zoom=4)