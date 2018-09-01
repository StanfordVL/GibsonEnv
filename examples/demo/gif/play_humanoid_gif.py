
from gibson.envs.humanoid_env import HumanoidNavigateEnv
from gibson.utils.play import play
import os

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'configs', 'gif', 'play_humanoid_gif2.yaml')
print(config_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()

    env = HumanoidNavigateEnv(config=args.config, gpu_count = 0)
    play(env, zoom=4)