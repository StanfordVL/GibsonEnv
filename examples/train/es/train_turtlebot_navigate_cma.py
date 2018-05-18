#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import gym, logging
from mpi4py import MPI
from gibson.envs.mobile_robots_env import TurtlebotNavigateEnv
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
from es import CMAES, PEPG
from tensorboardX import SummaryWriter
import cv2
from torch.autograd import Variable
import transforms3d
from gibson.core.render.utils import quat_wxyz_to_xyzw

img_size = 128
fov = 90
sigma_init = 0.1
sigma_decay = 0.9999
popsize = 24
nparam = 1640

def eular2quat(a,b,c):
    return quat_wxyz_to_xyzw(transforms3d.euler.euler2quat(a,b,c))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def calculate_params(self):
        self.num_params = []
        for k in self.state_dict():
            self.num_params.append((k, self.state_dict()[k].view(-1).size(0)))

        self.all_params = sum([item[1] for item in self.num_params])


    def serialize(self):
        self.calculate_params()
        param = np.zeros((self.all_params,))
        c = 0
        for k,n in self.num_params:
            param[c:c+n] = self.state_dict()[k].view(-1).numpy()
            c += n

        return param

    def restore(self, param):
        self.calculate_params()
        c = 0
        new_dict = {}
        for k, n in self.num_params:
            new_dict[k] = torch.FloatTensor(param[c:c + n]).view(self.state_dict()[k].size())
            c += n

        self.load_state_dict(new_dict)
        return new_dict

class SimpleMLP(Net):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(64, 10)
        self.fc2 = nn.Linear(10, 5)

    def forward(self, x):
        x = F.relu(self.fc1(F.max_pool1d(x, 2)))
        x = self.fc2(x)
        return x

class SimpleRNN(Net):
    def __init__(self, hidden_size = 10):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size

        self.inp = nn.Linear(64, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, 1, dropout=0.05)
        self.out = nn.Linear(hidden_size, 5)


    def step(self, input, hidden=None):
        input = self.inp(F.max_pool1d(input.view(1, 1, 128), 2).view(1,-1)).unsqueeze(1)
        output, hidden = self.rnn(input, hidden)
        output = self.out(output.squeeze(1))
        return output#, hidden


def main(args):
    writer = SummaryWriter()
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'configs', 'turtlebot_navigate_depth.yaml')
    print(config_file)
    env = TurtlebotNavigateEnv(gpu_count=args.gpu_count, config=config_file)

    solver = CMAES(nparam,
                sigma_init=sigma_init,
                popsize=popsize)


    angle = [-0.2 * np.pi, 0, 0.2 * np.pi]
    orn = env.config["initial_orn"]
    #print(orientations)
    #print(eular2quat(orientations[0], orientations[1], orientations[2]))
    orns = []
    for i in range(3):
        orns.append(eular2quat(orn[0],orn[1],orn[2] + angle[i]))

    for generation in range(200):
        solutions = solver.ask()

        #print(solutions.shape)
        rewards = np.zeros(solver.popsize)

        for i in range(popsize):

            reward_episode = []

            for j in range(3):
                net = SimpleRNN()
                net.restore(solutions[i])

                env.reset()
                #orn = env.robot.get_orientation()
                #print(orn)
                env.robot.reset_new_pos(env.robot.get_position(), orns[j])

                done = False

                obs, _, _, _ = env.step(4)

                while not done:
                    laser = np.mean(obs['depth'][img_size // 2 - 4: img_size // 2 + 4, :, 0], axis=0)
                    #print(laser)
                    laser_image = np.zeros((128,128))
                    laser_idx = np.clip(laser * 5, 0, 127).astype(np.int32)
                    laser_image[np.arange(0,128), laser_idx] = 1
                    cv2.imshow("laser", laser_image)
                    cv2.waitKey(1)
                    laser = torch.Tensor(laser).view(1, 1, 128)
                    action = net.step(laser).squeeze()
                    #print(action)
                    action = action.data.max(0)[1][0]

                    #from IPython import embed; embed()
                    obs, rew, done, meta = env.step(action)
                    #print(rew)
                    #print(action,rew)
                    #print(laser)
                    reward_episode.append(rew)
                    #from IPython import embed; embed()
            #print(reward_episode)
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
