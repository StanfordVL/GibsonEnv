from gibson.envs.husky_env import HuskyNavigateEnv
from gibson.utils.play import play
import os

timestep = 1.0/(4 * 18)
frame_skip = 1

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'husky_navigate.yaml')
print(config_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()

    #env = HuskyNavigateEnv(human=True, timestep=timestep, frame_skip=frame_skip, mode="RGB", is_discrete = True, resolution=args.resolution)
    env = HuskyNavigateEnv(is_discrete = True, config=args.config)
    play(env, zoom=4, fps=int( 1.0/(timestep * frame_skip)))