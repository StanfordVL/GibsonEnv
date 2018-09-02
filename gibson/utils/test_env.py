from __future__ import print_function
import time
import numpy as np
import sys
import gym
from PIL import Image
from gibson.core.render.profiler import Profiler
from gibson.envs.husky_env import *
from gibson.envs.ant_env import *
from gibson.envs.humanoid_env import *
from gibson.envs.drone_env import *
import pybullet as p

GPU_NUM = 0

class RandomAgent(object):
    """The world's simplest agent"""
    def __init__(self, action_space, is_discrete):
        self.action_space = action_space
        self.is_discrete = is_discrete
        
    def act(self, observation, reward=None):
        if self.is_discrete:
            action = np.random.randint(self.action_space.n)
        else:
            action = np.zeros(self.action_space.shape[0])
            if (np.random.random() < 0.5):
                action[np.random.choice(action.shape[0], 1)] = np.random.randint(-1, 2)
        return action

def testEnv(Env, config="test_filled.yaml", frame_total=10, is_discrete=False):
    print("Currently testing", Env)
    config = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'examples', 'configs', 'test', config)
    env = Env(config)
    obs = env.reset()
    agent = RandomAgent(env.action_space, is_discrete)
    frame = 0
    score = 0
    restart_delay = 0
    obs = env.reset()
    while True:
        time.sleep(0.01)
        a = agent.act(obs)
        obs, r, done, meta = env.step(a)
        score += r
        frame += 1
        if not done and frame < frame_total: continue
        env.close()
        break

    for port in range(5556-GPU_NUM*5-4, 5556-GPU_NUM*5+1):
        os.system("lsof -i tcp:" + str(port) + " | awk 'NR!=1 {print $2}' | xargs kill")
    os.system("pkill depth")

    
if __name__ == '__main__':
    testEnv(HuskyNavigateEnv, "test_semantics.yaml", 10, is_discrete=True)
    testEnv(HuskyNavigateEnv, "test_filled.yaml", 10, is_discrete=True)
    testEnv(HuskyNavigateEnv, "test_prefilled.yaml", 10, is_discrete=True)
    testEnv(HuskyNavigateEnv, "test_depth.yaml", 10, is_discrete=True)
    testEnv(HuskyNavigateEnv, "test_normal.yaml", 10, is_discrete=True)
    testEnv(HuskyNavigateEnv, "test_three.yaml", 10, is_discrete=True)
    testEnv(HuskyNavigateEnv, "test_four.yaml", 10, is_discrete=True)
    testEnv(HuskyNavigateEnv, "test_nonviz.yaml", 10, is_discrete=True)
    testEnv(HuskyGibsonFlagRunEnv, "test_nonviz.yaml", 10, is_discrete=True)
    testEnv(HuskyGibsonFlagRunEnv, "test_depth.yaml", 10, is_discrete=True)


    testEnv(AntGibsonFlagRunEnv, "test_nonviz_nondiscrete.yaml", 10, is_discrete=False)
    testEnv(AntFlagRunEnv, "test_nonviz_nondiscrete.yaml", 10, is_discrete=False)
    testEnv(AntClimbEnv, "test_nonviz_nondiscrete.yaml", 10, is_discrete=False)
    testEnv(AntNavigateEnv, "test_nonviz_nondiscrete.yaml", 10, is_discrete=False)
    testEnv(AntClimbEnv, "test_four_nondiscrete.yaml", 10, is_discrete=False)
    
    testEnv(HumanoidNavigateEnv, "test_nonviz_nondiscrete.yaml", 10, is_discrete=False)
    testEnv(HumanoidGibsonFlagRunEnv, "test_nonviz_nondiscrete.yaml", 10, is_discrete=False)
    testEnv(HumanoidNavigateEnv, "test_four_nondiscrete.yaml", 10, is_discrete=False)
    

    testEnv(DroneNavigateEnv, "test_nonviz_nondiscrete.yaml", 100, is_discrete=False)
    testEnv(DroneNavigateEnv, "test_four_nondiscrete.yaml", 100, is_discrete=False)