from gibson.envs.husky_env import HuskyNavigateEnv
from gibson.utils.play import play
from gibson.core.render.profiler import Profiler
import os

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'benchmark.yaml')
print(config_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default=config_file)
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    #env = HuskyNavigateEnv(human=True, timestep=timestep, frame_skip=frame_skip, mode="RGB", is_discrete = True, resolution=args.resolution)
    env = HuskyNavigateEnv(config=args.config, gpu_idx=args.gpu)
    env.reset()
    frame = 0
    while frame < 10000:
        with Profiler("env step"):
            env.step(2)

        frame += 1
        print("frame {}".format(frame))