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
import pybullet as p

class RandomAgent(object):
    """The world's simplest agent"""
    def __init__(self, action_space, is_discrete = False):
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

def testEnv(Env, config="test.yaml", frame_total=10):
    print("Currently testing", Env)
    config = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'examples', 'configs', config)
    env = Env(config, human=True, is_discrete = False)
    obs = env.reset()
    agent = RandomAgent(env.action_space, is_discrete = False)
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
        return
    
if __name__ == '__main__':
    '''
    testEnv(HumanoidGibsonFlagRunEnv)
    testEnv(HumanoidNavigateEnv)
    testEnv(HuskyNavigateEnv)
    testEnv(HuskyGibsonFlagRunEnv)
    testEnv(AntClimbEnv)
    testEnv(AntFlagRunEnv)
    testEnv(AntGibsonFlagRunEnv)
    testEnv(AntNavigateEnv)
    '''
    testEnv(HuskyNavigateEnv, "test_semantics.yaml", 100)