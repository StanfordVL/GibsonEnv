from __future__ import print_function
import time
import numpy as np
import sys
import gym
import math
from PIL import Image
from realenv.core.render.profiler import Profiler
from realenv.envs.minitaur_env import MinitaurNavigateEnv
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
        self.step_counter = 0
        
    def act(self, observation, reward=None):
        self.t += 0 #TIMESTEP / 100
        assert(not self.is_discrete)
        #action = [math.sin(self.speed * self.t) * self.amplitude + math.pi / 2] * 8
        sum_reward = 0
        steps = 20000
        amplitude_1_bound = 0.1
        amplitude_2_bound = 0.1
        speed = 1
        self.step_counter += 1


        time_step = 0.01
        t = self.step_counter * time_step

        amplitude1 = amplitude_1_bound
        amplitude2 = amplitude_2_bound
        steering_amplitude = 0
        if t < 10:
          steering_amplitude = 0.1
        elif t < 20:
          steering_amplitude = -0.1
        else:
          steering_amplitude = 0

        # Applying asymmetrical sine gaits to different legs can steer the minitaur.
        a1 = math.sin(t * speed) * (amplitude1 + steering_amplitude)
        a2 = math.sin(t * speed + math.pi) * (amplitude1 - steering_amplitude)
        a3 = math.sin(t * speed) * amplitude2
        a4 = math.sin(t * speed + math.pi) * amplitude2
        action = [a1, a2, a2, a1, a3, a4, a4, a3]
        #action = [0, 0, 0, 0, 0, 0, 0, 0]
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

            if not done and frame < 600: continue
            if restart_delay==0:
                print("score=%0.2f in %i frames" % (score, frame))
                restart_delay = 20 * 6  # 2 sec at 60 fps
            else:
                restart_delay -= 1
                if restart_delay==0: break