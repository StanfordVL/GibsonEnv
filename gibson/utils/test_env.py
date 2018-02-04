from __future__ import print_function
import time
import numpy as np
import sys
import gym
from PIL import Image
from gibson.core.render.profiler import Profiler
from gibson.envs.husky_env import HuskyNavigateEnv, HuskyClimbEnv, HuskyFlagRunEnv, HuskyFetchEnv, HuskyFetchKernelizedRewardEnv, HuskyGoallessRunEnv
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

def testEnv(Env, mode="RGBD"):
    print("Currently testing", Env)
    env = Env(human=True, timestep=1.0/(4 * 22), frame_skip=4, is_discrete = False, mode=mode, resolution="NORMAL")
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
        if not done and frame < 10: continue
        env.close()
        return
    
if __name__ == '__main__':   
    testEnv(HuskyNavigateEnv, "RGBD")
    testEnv(HuskyClimbEnv, "RGBD")
    testEnv(HuskyFlagRunEnv, "RGBD")
    testEnv(HuskyFetchEnv, "RGBD")
    testEnv(HuskyFetchKernelizedRewardEnv, "RGBD")
    testEnv(HuskyGoallessRunEnv, "RGBD")

    testEnv(HuskyNavigateEnv, "SENSOR")
    testEnv(HuskyClimbEnv, "SENSOR")
    testEnv(HuskyFlagRunEnv, "SENSOR")
    testEnv(HuskyFetchEnv, "SENSOR")
    testEnv(HuskyFetchKernelizedRewardEnv, "SENSOR")
    testEnv(HuskyGoallessRunEnv, "SENSOR")