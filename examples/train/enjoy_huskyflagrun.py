# add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
from realenv.envs.husky_env import HuskyFlagRunEnv

from baselines import deepq
import matplotlib.pyplot as plt
import datetime
from realenv.core.render.profiler import Profiler
import time

def callback(lcl, glb):
    # stop training if reward exceeds 199
    total = sum(lcl['episode_rewards'][-101:-1]) / 100
    totalt = lcl['t']
    is_solved = totalt > 20 and total >= 100
    return is_solved


def main():
    env = HuskyFlagRunEnv(human=True, is_discrete=True, enable_sensors=True)
    act = deepq.load("husky_flagrun_model.pkl")

    obs = env.reset()
    assert (not obs is None)
    obs = obs.reshape(1,20)
    while 1:
        frame = 0
        score = 0
        restart_delay = 0
        obs = env.reset()
        obs = obs.reshape(1, 20)
        assert (not obs is None)
        while True:
            time.sleep(0.01)

            a = act(obs)[0]
            print("action", a)
            with Profiler("Agent step function"):
                obs, r, done, meta = env.step(a)
                obs = obs.reshape(1, 20)
            score += r
            frame += 1
            print(obs)
            if not done and frame < 60: continue
            if restart_delay == 0:
                print("score=%0.2f in %i frames" % (score, frame))
                restart_delay = 200 * 4  # 2 sec at 60 fps
            else:
                restart_delay -= 1
                if restart_delay == 0: break


if __name__ == '__main__':
    main()
