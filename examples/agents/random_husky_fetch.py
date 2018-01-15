from __future__ import print_function
import time
import numpy as np
import sys
import gym
from PIL import Image
from realenv.core.render.profiler import Profiler
from realenv.envs.husky_env import HuskyFetchEnv
import pybullet as p


class RandomAgent(object):
    """The world's simplest agent"""

    def __init__(self, action_space, is_discrete=False):
        self.action_space = action_space
        self.is_discrete = is_discrete

    def act(self, observation, reward=None):
        if self.is_discrete:
            action = np.random.randint(self.action_space.n)
        else:
            action = np.zeros(self.action_space.shape[0])
            if (np.random.random() < 0.2):
                action[np.random.choice(action.shape[0], 1)] = np.random.randint(-1, 2)
        return action


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--human', action='store_true', default=False)
    args = parser.parse_args()

    env = HuskyFetchEnv(human=args.human, timestep=1.0 / (4 * 22), frame_skip=4, is_discrete=True,  mode="DEPTH", use_filler=False, resolution = "LARGE")
    #env = HuskyFetchEnv(human=args.human, timestep=1.0 / (4 * 22), frame_skip=4, is_discrete=True,  mode="DEPTH", use_filler=False, resolution = "NORMAL")
    obs = env.reset()
    agent = RandomAgent(env.action_space, is_discrete=True)
    assert (not obs is None)


    frame = 0
    score = 0
    restart_delay = 0
    obs = env.reset()
    ts = np.zeros((200,))
    assert (not obs is None)
    while frame < 200:
        #time.sleep(0.01)
        a = agent.act(obs)
        print("action", a)
        t = time.time()
        obs, r, done, meta = env.step(a)
        ts[frame] = time.time() - t
        score += r
        frame += 1

    print(np.mean(ts), "%.1f fps" % (1/np.mean(ts)))