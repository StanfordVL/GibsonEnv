## Camrbria learning code using DQN, adapted from OpenAI baselines
#  Note this file might be a bit long, because original learning code is included, in order
#   to support tensorflow config for single GPU learning + rendering.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)
import gym
from realenv.envs.husky_env import HuskyNavigateEnv
import deepq
import numpy as np



def callback(lcl, glb):
    # stop training if reward exceeds 199
    total = sum(lcl['episode_rewards'][-101:-1]) / 100
    totalt = lcl['t']
    is_solved = totalt > 2000 and total >= -50
    is_solved = False
    return is_solved


def main():
    env = HuskyNavigateEnv(human=args.human, is_discrete=True, mode=args.mode, use_filler=not args.disable_filler, gpu_count=args.gpu_count, resolution=args.resolution)
    if args.mode in ["RGB", "DEPTH", "RGBD", "GREY"]:
        model = deepq.models.cnn_to_mlp(
            convs=[(64, 8, 4), (64, 4, 2), (64, 3, 1)],
            hiddens=[256],
            dueling=True,
        )
    elif args.mode == "SENSOR":
        model = deepq.models.mlp([64])
    else:
        raise AssertionError()
    
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=10000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback,
        mode=args.mode
    )
    print("Saving model to humanoid_sensor_model.pkl")
    act.save("humanoid_sensor_model.pkl")



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
    args = parser.parse_args()
    
    main()

