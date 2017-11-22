from __future__ import print_function
import time
import numpy as np
import sys
import gym
from PIL import Image
from realenv.core.render.profiler import Profiler
from realenv.envs.ant_env import AntNavigateEnv
import pybullet as p


class RandomAgent(object):
    """The world's simplest agent"""
    def __init__(self, action_space):
        self.action_space = action_space
        
    def act(self, observation, reward=None):
        action = np.zeros(self.action_space.shape[0])
        if (np.random.random() < 0.7):
            action[np.random.choice(action.shape[0], 1)] = np.random.randint(-1, 2)
        return action

if __name__ == '__main__':
    #env = gym.make('HumanoidSensor-v0')
    env = AntNavigateEnv(human=True, timestep=1.0/(4 * 22), frame_skip=4, mode="RGBD")
    obs = env.reset()
    #print(env.observation_space, env.action_space)
    #sys.exit()
    agent = RandomAgent(env.action_space)
    
    while 1:
        frame = 0
        score = 0
        restart_delay = 0
        obs = env.reset()
        while True:
            time.sleep(0.01)
            a = agent.act(obs)
            with Profiler("Agent step function"):
                obs, r, done, meta = env.step(a)
            score += r
            frame += 1
            
            if not done and frame < 60: continue
            if restart_delay==0:
                print("score=%0.2f in %i frames" % (score, frame))
                restart_delay = 200000 * 4  # 2 sec at 60 fps
            else:
                restart_delay -= 1
                if restart_delay==0: break
