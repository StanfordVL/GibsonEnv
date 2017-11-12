from realenv.envs.husky_env import HuskyNavigateEnv, HuskyClimbEnv
from realenv.utils.play import play

timestep = 1.0/(4 * 18)
frame_skip = 4

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--resolution', type=str, default="NORMAL")
    args = parser.parse_args()

    #env = HuskyNavigateEnv(human=True, timestep=timestep, frame_skip=frame_skip, mode="RGB", is_discrete = True, resolution=args.resolution)
    env = HuskyClimbEnv(human=True, timestep=timestep, frame_skip=frame_skip, mode="RGB", is_discrete = True, resolution=args.resolution)
    play(env, zoom=4, fps=int( 1.0/(timestep * frame_skip)))