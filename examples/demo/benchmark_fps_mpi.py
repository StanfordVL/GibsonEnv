from gibson.envs.husky_env import HuskyNavigateEnv
from gibson.utils.play import play
from gibson.core.render.profiler import Profiler
import os
from mpi4py import MPI
import sys
import time


size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
comm = MPI.COMM_WORLD

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'benchmark.yaml')
print(config_file)

os.environ['CUDA_VISIBLE_DEVICES'] = "%i" % rank

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default=config_file)
    parser.add_argument('--gpu', type=int, default=rank)

    args = parser.parse_args()

    #env = HuskyNavigateEnv(human=True, timestep=timestep, frame_skip=frame_skip, mode="RGB", is_discrete = True, resolution=args.resolution)
    env = HuskyNavigateEnv(config=args.config, gpu_idx=args.gpu)
    env.reset()
    frame = 0
    start = time.time()
    while frame < 1000:
        if rank == 0:
            data = [2] * size
        else:
            data = None
        data = comm.scatter(data, root = 0)
        with Profiler("env step"):
            obs, _, _, _ = env.step(data)
            obss = comm.gather(obs, root = 0)
        frame += size
        print("frame {}".format(frame))

    dt = time.time()-start
    print("{} frame per sec".format(1000/dt))

