from gibson.envs.husky_env import HuskyNavigateEnv
import argparse
import os

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'test_control.yaml')
print(config_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()

    env = HuskyNavigateEnv(config = args.config)
    env.reset()
    for i in range(10000):
        env.step(action = [0.1,0,0.1,0])
