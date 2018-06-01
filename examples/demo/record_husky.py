from gibson.envs.husky_env import HuskyNavigateEnv
from gibson.utils.play_record_husky import play
import os, yaml

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_id', type=str, default="space7")
    args = parser.parse_args()

    config_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'recording')
    config_paths = [os.path.join(config_root, file) for file in os.listdir(config_root) if args.model_id in file]
    #config_paths = ["examples/configs/test_control.yaml"]
    print(config_paths)
    all_configs = []

    for path in config_paths:
        with open(path, 'r') as f:
            all_configs.append(yaml.load(f))
    env = HuskyNavigateEnv(config=config_paths[0], gpu_count = 0)
    play(env, all_configs, zoom=4)