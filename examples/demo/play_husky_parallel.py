'''
Running two Gibson Envionments in one machine (same GPU) in parallel
Commands:
	python examples/demo/play_husky_camera.py
	## In a separate terminal
	python examples/demo/play_husky_parallel.py
'''

from gibson.envs.husky_env import HuskyNavigateEnv
from gibson.utils.play import play
import os

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'play', 'play_husky_camera.yaml')
print(config_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()
    env = HuskyNavigateEnv(config=args.config, gpu_count = 1)
    play(env, zoom=4)