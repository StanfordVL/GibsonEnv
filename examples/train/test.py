#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import gym, logging
from baselines.common import set_global_seeds
from baselines.ppo1 import pposgd_simple
from baselines import deepq

from mpi4py import MPI
from realenv.envs.husky_env import HuskyCameraEnv, HuskySensorEnv, HuskyFlagRunEnv
from realenv.envs.ant_env import AntCameraEnv, AntSensorEnv
import resnet_policy
import tf_util as U
import datetime
from baselines import logger
from baselines import bench
import os.path as osp
import random

## Training code adapted from: https://github.com/openai/baselines/blob/master/baselines/ppo1/run_atari.py

def train(num_timesteps, seed):
    rank = MPI.COMM_WORLD.Get_rank()
    #sess = U.single_threaded_session()
    sess = U.make_gpu_session(args.num_gpu)
    sess.__enter__()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    if args.mode =="RGB" or args.mode == "rgb":
        env = AntCameraEnv(human=args.human, is_discrete=False, enable_sensors=True, mode="RGB")
    elif args.mode =="GREY" or args.mode == "grey":
        env = HuskyCameraEnv(human=args.human, is_discrete=True, enable_sensors=True, mode="GREY", gpu_count=args.gpu_count)
    elif args.mode =="RGBD" or args.mode == "rgbd":
        env = HuskyCameraEnv(human=args.human, is_discrete=True, enable_sensors=True, mode="RGBD", gpu_count=args.gpu_count)

    ob = env.reset()
    act_sp = env.action_space
    print(ob.shape)
    print(act_sp.shape)
    print(type(act_sp))
    print(act_sp.n)


def callback(lcl, glb):
    # stop training if reward exceeds 199
    total = sum(lcl['episode_rewards'][-101:-1]) / 100
    totalt = lcl['t']
    is_solved = totalt > 2000 and total >= -50
    return is_solved


def main():
    train(num_timesteps=1000000, seed=5)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default="rgb")
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--human', type=bool, default=False)
    parser.add_argument('--gpu_count', type=int, default=0)
    args = parser.parse_args()
    
    main()
