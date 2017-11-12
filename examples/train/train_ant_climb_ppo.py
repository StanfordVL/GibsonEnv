#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import gym, logging
from mpi4py import MPI
from realenv.envs.ant_env import AntClimbEnv
from baselines.common import set_global_seeds
import deepq
import cnn_policy, mlp_policy
import utils
import baselines.common.tf_util as U
import datetime
from baselines import logger
from baselines import bench
import pposgd_simple
import os.path as osp
import random

def train(num_timesteps, seed):
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = AntClimbEnv(human=args.human, is_discrete=False, mode=args.mode)
    
    def policy_fn(name, ob_space, ac_space): #pylint: disable=W0613
        if args.mode == "SENSOR":
            return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=64, num_hid_layers=2)
        else:
            return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space, save_per_acts=10000, session=sess)
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)


    pposgd_simple.learn(env, policy_fn,
        max_timesteps=int(num_timesteps * 1.1),
        timesteps_per_actorbatch=256,
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
    '''
    env = AntSensorEnv(human=True, is_discrete=False, enable_sensors=True)
    model = deepq.models.mlp([64])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=10000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print("Saving model to humanoid_sensor_model.pkl")
    act.save("humanoid_sensor_model.pkl")
    '''
    train(num_timesteps=10000, seed=5)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default="SENSOR")
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
