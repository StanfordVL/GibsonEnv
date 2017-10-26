from __future__ import print_function
import time
import numpy as np
import sys
import gym
from PIL import Image
from realenv.core.render.profiler import Profiler
import pybullet as p


class RandomAgent(object):
    """The world's simplest agent"""
    def __init__(self, action_space):
        self.action_space = action_space
        self.time = 0
        self.repeat = 1
        self.action_last  = np.zeros(self.action_space.shape[0])

    def act(self, observation, reward=None):
        if self.time < self.repeat:
            self.time = self.time + 1
            return self.action_last
        else:
            self.time = 0
            action = np.zeros(self.action_space.shape[0])
            #action[np.random.randint(0, len(action))] = 1
            action = [0] * self.action_space.shape[0]
            self.action_last = action
            return action

if __name__ == '__main__':
    env = gym.make('HumanoidCamera-v0')
    env.reset()
    agent = RandomAgent(env.action_space)
    ob = None
    torsoId = -1

    for i in range (p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
           torsoId=i
    i = 0

    try:
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
                distance=2.5 ## demo: living room ,kitchen
                #distance=1.7   ## demo: stairs
                #yaw = 0     ## demo: living room
                yaw = 30    ## demo: kitchen
                #yaw = 90     ## demo: stairs
                humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
                p.resetDebugVisualizerCamera(distance,yaw,-35,humanPos);       ## demo: kitchen, living room
                #p.resetDebugVisualizerCamera(distance,yaw,-42,humanPos);        ## demo: stairs

                if not done and frame < 60: continue
                if restart_delay==0:
                    print("score=%0.2f in %i frames" % (score, frame))
                    restart_delay = 20 * 4  # 2 sec at 60 fps
                else:
                    restart_delay -= 1
                    if restart_delay==0: break


    except KeyboardInterrupt:
        env._end()
        print("Program finished")
