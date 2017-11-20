#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

from realenv import configs

import gym, logging
from mpi4py import MPI
from realenv.envs.ant_env import AntClimbEnv
from baselines.common import set_global_seeds
import deepq
import cnn_policy, mlp_policy
import pposgd_sensor, pposgd_fuse, fuse_policy
import utils
import baselines.common.tf_util as U
import datetime
from baselines import logger
from monitor import Monitor
import os.path as osp
import random
import sys

def train(num_timesteps, seed):
    rank = MPI.COMM_WORLD.Get_rank()
    sess = utils.make_gpu_session(args.num_gpu)
    sess.__enter__()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = AntClimbEnv(human=args.human, is_discrete=False, mode=args.mode)
    
    env = Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    def mlp_policy_fn(name, sensor_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=sensor_space, ac_space=ac_space, hid_size=64, num_hid_layers=2)

    def fuse_policy_fn(name, ob_space, sensor_space, ac_space):
        return fuse_policy.FusePolicy(name=name, ob_space=ob_space, sensor_space=sensor_space, hid_size=64, num_hid_layers=2, ac_space=ac_space, save_per_acts=10000, session=sess)

    if args.mode == "SENSOR":
        pposgd_sensor.learn(env, mlp_policy_fn,
            max_timesteps=int(num_timesteps * 1.1 * 5),
            timesteps_per_actorbatch=60000,
            clip_param=0.2, entcoeff=0.00,
            optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64,
            gamma=0.99, lam=0.95,
            schedule='linear',
            save_per_acts=500,
            save_name="ant_ppo_mlp"
        )
        env.close()        
    else:
        pposgd_fuse.learn(env, fuse_policy_fn,
            max_timesteps=int(num_timesteps * 1.1),
            timesteps_per_actorbatch=20000,
            clip_param=0.2, entcoeff=0.01,
            optim_epochs=4, optim_stepsize=configs.LEARNING_RATE, optim_batchsize=64,
            gamma=0.99, lam=0.95,
            schedule='linear',
            save_per_acts=50,
            save_name="ant_ppo_fuse"
        )
        env.close()


def callback(lcl, glb):
    # stop training if reward exceeds 199
    total = sum(lcl['episode_rewards'][-101:-1]) / 100
    totalt = lcl['t']
    is_solved = totalt > 2000 and total >= -50
    return is_solved


def main():
    train(num_timesteps=50000000, seed=5)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default="RGB")
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--human', action='store_true', default=False)
    parser.add_argument('--gpu_count', type=int, default=0)
    parser.add_argument('--disable_filler', action='store_true', default=False)
    parser.add_argument('--meta', type=str, default="")
    parser.add_argument('--resolution', type=str, default="NORMAL")
    parser.add_argument('--reload_name', type=str, default=None)
    parser.add_argument('--save_name', type=str, default=None)
    args = parser.parse_args()
    main()
