from __future__ import print_function
import time
import numpy as np
import sys
import gym
import math
from PIL import Image
from gibson.core.render.profiler import Profiler
from gibson.envs.minitaur_env import MinitaurNavigateEnv
import pybullet as p

TIMESTEP = 1.0/ 22

class SineAgent(object):
    """The world's simplest agent"""
    def __init__(self, action_space, is_discrete = False):
        self.action_space = action_space
        self.is_discrete = is_discrete
        self.speed = 3
        self.amplitude = 0.5
        self.t = 0
        self.speed = 3
        
    def act(self, observation, reward=None):
        self.t += 0 #TIMESTEP / 100
        assert(not self.is_discrete)
        action = [math.sin(self.speed * self.t) * self.amplitude + math.pi / 2] * 8
        return action


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default="SENSOR")
    parser.add_argument('--human', action='store_true', default=True)
    args = parser.parse_args()
    
    env = MinitaurNavigateEnv(human=args.human, timestep=TIMESTEP, frame_skip=4, is_discrete = False, mode=args.mode)
    obs = env.reset()
    agent = SineAgent(env.action_space, is_discrete = False)
    assert(not obs is None)

    while 1:
        frame = 0
        score = 0
        restart_delay = 0
        obs = env.reset()
        while True:
            time.sleep(0.05)
            a = agent.act(obs)
            with Profiler("Agent step function"):
                obs, r, done, meta = env.step(a)
            score += r
            frame += 1

            if not done and frame < 60: continue
            if restart_delay==0:
                print("score=%0.2f in %i frames" % (score, frame))
                restart_delay = 20 * 6  # 2 sec at 60 fps
            else:
                restart_delay -= 1
                if restart_delay==0: break