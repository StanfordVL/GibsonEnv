#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import gym
import numpy as np
import pybullet as p
import pybullet_envs
import time
from realenv.core.physics.drivers.simple_humanoid_env import SimpleHumanoidGymEnv
from realenv.core.physics.drivers.simple_humanoid import SimpleHumanoid

def relu(x):
    return np.maximum(x, 0)

class SmallReactivePolicy:
    "Simple multi-layer perceptron policy, no internal state"
    def __init__(self, observation_space, action_space):
        assert weights_dense1_w.shape == (observation_space.shape[0], 256)
        assert weights_dense2_w.shape == (256, 128)
        assert weights_final_w.shape  == (128, action_space.shape[0])

    def act(self, ob):
        ob[0] += -1.4 + 0.8
        x = ob
        x = relu(np.dot(x, weights_dense1_w) + weights_dense1_b)
        x = relu(np.dot(x, weights_dense2_w) + weights_dense2_b)
        x = np.dot(x, weights_final_w) + weights_final_b
        return x

def main():
    env = gym.make("HuskyWalkingEnv-v0")
    env.configure(timestep=1.0/(4 * 9), frame_skip=4)
    #pi = SmallReactivePolicy(env.observation_space, env.action_space)
    env.reset()
    huskyId = -1
    for i in range (p.getNumBodies()):
        print(p.getBodyInfo(i))
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            huskyId=i
            print("found husky robot")
    while 1:
        frame = 0
        score = 0
        restart_delay = 0
        obs = env.reset()
       
        while 1:
            time.sleep(0.01)
            #a = pi.act(obs)
            a = [1] * env.action_space.shape[0]
            obs, r, done, _ = env.step(a)
            print("observations", len(obs))
            score += r
            frame += 1
            distance=5
            yaw = 0
            huskyPos, huskyOrn = p.getBasePositionAndOrientation(huskyId)
            p.resetDebugVisualizerCamera(distance,yaw,-20,humanPos);

            still_open = env.render("human")
            if still_open==False:
                return
            if not done: continue
            if restart_delay==0:
                print("score=%0.2f in %i frames" % (score, frame))
                restart_delay = 60*2  # 2 sec at 60 fps
            else:
                restart_delay -= 1
                if restart_delay==0: break


if __name__=="__main__":
    main()
