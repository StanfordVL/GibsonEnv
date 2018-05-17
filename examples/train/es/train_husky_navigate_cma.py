#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import gym, logging
from mpi4py import MPI
from gibson.envs.husky_env import HuskyNavigateEnv
from baselines.common import set_global_seeds
from gibson.utils import pposgd_simple
import baselines.common.tf_util as U
from gibson.utils import cnn_policy, mlp_policy
from gibson.utils import utils
import datetime
from baselines import logger
from gibson.utils.monitor import Monitor
import os.path as osp
import random
import sys
import matplotlib.pyplot as plt
from gibson.core.render.profiler import Profiler
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from es import CMAES
from tensorboardX import SummaryWriter


img_size = 128
fov = 90
sigma_init = 0.1
popsize = 24
nparam = 705

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(64, 10)
        self.fc2 = nn.Linear(10, 5)

        #print(self.state_dict())
        self.num_params = []
        for k in self.state_dict():
            self.num_params.append((k, self.state_dict()[k].view(-1).size(0)))

        #print(self.num_params)

        self.all_params = sum([item[1] for item in self.num_params])
        #print(self.all_params)
    def forward(self, x):
        x = F.relu(self.fc1(F.max_pool1d(x, 2)))
        x = self.fc2(x)
        return x

    def serialize(self):
        param = np.zeros((self.all_params,))
        c = 0
        for k,n in self.num_params:
            param[c:c+n] = self.state_dict()[k].view(-1).numpy()
            c += n

        return param

    def restore(self, param):
        c = 0
        new_dict = {}
        for k, n in self.num_params:
            new_dict[k] = torch.FloatTensor(param[c:c + n]).view(self.state_dict()[k].size())
            c += n

        self.load_state_dict(new_dict)
        return new_dict

def main(args):
    writer = SummaryWriter()
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'configs', 'husky_navigate_depth.yaml')
    print(config_file)
    env = HuskyNavigateEnv(gpu_count=args.gpu_count, config=config_file)

    solver = CMAES(nparam,
                sigma_init=sigma_init,
                popsize=popsize)

    for generation in range(200):
        solutions = solver.ask()
        #print(solutions.shape)
        rewards = np.zeros(solver.popsize)

        for i in range(popsize):
            net = Net()
            net.restore(solutions[i])
            obs = env.reset()
            done = False
            reward_episode = []
            while not done:
                laser = np.mean(obs['depth'][img_size // 2 - 4: img_size // 2 + 4, :, 0], axis=0)
                laser = torch.Tensor(laser).view(1, 1, 128)
                action = net(laser).squeeze()
                action = action.data.max(0)[1][0]

                #from IPython import embed; embed()
                obs, rew, done, meta = env.step(action)
                #print(action,rew)
                #print(laser)
                reward_episode.append(rew)
                #from IPython import embed; embed()

            rewards[i] = np.mean(np.array(reward_episode))

        solver.tell(rewards)

        writer.add_scalar('data/min_rew', np.min(rewards), generation)
        writer.add_scalar('data/max_rew', np.max(rewards), generation)
        writer.add_scalar('data/mean_rew', np.mean(rewards), generation)

        print(rewards)

        if generation % 50 == 0:
            np.save("data_{}.npy".format(generation), solutions)

    writer.export_scalars_to_json("./results.json")
    writer.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default="RGB")
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--gpu_count', type=int, default=0)
    parser.add_argument('--disable_filler', action='store_true', default=False)
    parser.add_argument('--meta', type=str, default="")
    parser.add_argument('--resolution', type=str, default="SMALL")
    parser.add_argument('--reload_name', type=str, default=None)
    parser.add_argument('--save_name', type=str, default=None)
    args = parser.parse_args()
    main(args = args)
