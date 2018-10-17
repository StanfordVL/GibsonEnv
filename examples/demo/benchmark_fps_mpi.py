#!/usr/bin/env python
"""
Example usage  mpiexec -n 2 --cpus-per-proc 3  python examples/demo/benchmark_fps_mpi.py
"""
from gibson.envs.husky_env import HuskyNavigateEnv
from gibson.utils.play import play
from gibson.core.render.profiler import Profiler
import os
from mpi4py import MPI
import numpy as np

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'benchmark.yaml')
print(config_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default=config_file)

    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    os.environ["CUDA_VISIBLE_DEVICES"] = str(comm.rank)

    env = HuskyNavigateEnv(config=args.config, gpu_idx=comm.rank)
    observation = env.reset()
    observation_all = comm.gather(observation, root=0)

    frame = 0

    while frame < 10000:
        with Profiler("env step"):

            if comm.rank == 0:
                action = [2] * comm.size
                print(action)
            else:
                action = None

            action = comm.scatter(action, root=0)

            observation, reward, _, _ = env.step(action)

            observation_all = comm.gather(observation, root=0)
            reward_all = comm.gather(reward, root=0)

            if comm.rank == 0:
                print(observation_all, reward_all)
