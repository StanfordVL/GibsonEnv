from gibson.envs.humanoid_env import HumanoidNavigateEnv
from gibson.utils.play import play

timestep = 1.0/200
frame_skip = 1

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--resolution', type=str, default="LARGE")
    args = parser.parse_args()

    #env = HuskyNavigateEnv(human=True, timestep=timestep, frame_skip=frame_skip, mode="SENSOR", is_discrete = True, resolution="MID")
    env = HumanoidNavigateEnv(human=True, timestep=timestep, frame_skip=frame_skip, mode="RGBD", is_discrete = True, resolution=args.resolution)
    play(env, zoom=4, fps=int( 1.0/(timestep * frame_skip)))