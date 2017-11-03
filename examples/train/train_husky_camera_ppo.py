#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import gym, logging
from mpi4py import MPI
from realenv.envs.husky_env import HuskyCameraEnv
from baselines.common import set_global_seeds
from baselines import deepq
from baselines.ppo1 import pposgd_simple
import cnn_policy
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
        env = HuskyCameraEnv(human=args.human, is_discrete=True, enable_sensors=True, mode="RGB")
    elif args.mode =="GREY" or args.mode == "grey":
        env = HuskyCameraEnv(human=args.human, is_discrete=True, enable_sensors=True, mode="GREY")
    elif args.mode =="RGBD" or args.mode == "rgbd":
        env = HuskyCameraEnv(human=args.human, is_discrete=True, enable_sensors=True, mode="RGBD")
    def policy_fn(name, ob_space, ac_space):
        return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space, save_per_acts=10000, session=sess)

    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)


    pposgd_simple.learn(env, policy_fn,
        max_timesteps=int(num_timesteps * 1.1),
        timesteps_per_actorbatch=1024,
        clip_param=0.2, entcoeff=0.01,
        optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64,
        gamma=0.99, lam=0.95,
        schedule='linear'
    )
    env.close()


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
    args = parser.parse_args()
    
    main()
