from gibson.envs.husky_env import HuskyNavigateEnv
from gibson.utils.play import play
import argparse
import os

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'husky_navigate_nonviz.yaml')
print(config_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()

    env = HuskyNavigateEnv(is_discrete = True, config = args.config)
    print(env.config)
    play(env, zoom=4)