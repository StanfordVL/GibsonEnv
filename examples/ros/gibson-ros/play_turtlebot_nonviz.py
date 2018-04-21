#!/usr/bin/python
from gibson.envs.mobile_robots_env import TurtlebotNavigateEnv
from gibson.utils.play import play
import argparse
import os
import rospy
from std_msgs.msg import Float32
import rospkg
import numpy as np

import gym
#import pygame
import sys
import time
import matplotlib
import time
import pygame
import pybullet as p
from gibson.core.render.profiler import Profiler

rospack = rospkg.RosPack()
path = rospack.get_path('gibson-ros')
config_file = os.path.join(path, 'play_turtlebot_nonviz.yaml')
print(config_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_file)
    args, unknown = parser.parse_known_args()

    rospy.init_node('gibson-sim')
    pub_vel_x = rospy.Publisher('/gibson/vel_x', Float32, queue_size=10)
    pub_vel_y = rospy.Publisher('/gibson/vel_y', Float32, queue_size=10)

    env = TurtlebotNavigateEnv(config = args.config)
    print(env.config)
    obs_s = env.observation_space
    # assert type(obs_s) == gym.spaces.box.Box
    # assert len(obs_s.shape) == 2 or (len(obs_s.shape) == 3 and obs_s.shape[2] in [1,3])

    keys_to_action=None
    callback=None


    if keys_to_action is None:
        if hasattr(env, 'get_keys_to_action'):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, 'get_keys_to_action'):
            keys_to_action = env.unwrapped.get_keys_to_action()
            # else:
            #    assert False, env.spec.id + " does not have explicit key to action mapping, " + \
            #                  "please specify one manually"
    relevant_keys = set(sum(map(list, keys_to_action.keys()), []))
    relevant_keys.add(ord('r'))

    '''
    if transpose:
        video_size = env.observation_space.shape[1], env.observation_space.shape[0]
    else:
        video_size = env.observation_space.shape[0], env.observation_space.shape[1]

    if zoom is not None:
        video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)
    '''
    pressed_keys = []
    running = True
    env_done = True

    record_num = 0
    record_total = 0
    obs = env.reset()

    do_restart = False

    vel = obs["nonviz_sensor"][3:6]

    while not rospy.is_shutdown():
        pub_vel_x.publish(vel[0])
        pub_vel_y.publish(vel[1])


        if do_restart:
            do_restart = False
            env.reset()
            pressed_keys = []
            continue
        if len(pressed_keys) == 0:
            action = keys_to_action[()]
            with Profiler("Play Env: step"):
                start = time.time()
                obs, rew, env_done, info = env.step(action)
                vel = obs["nonviz_sensor"][3:6]
                record_total += time.time() - start
                record_num += 1
            # print(info['sensor'])
            print("Play mode: reward %f" % rew)
        for p_key in pressed_keys:
            action = keys_to_action[(p_key,)]
            prev_obs = obs
            with Profiler("Play Env: step"):
                start = time.time()
                obs, rew, env_done, info = env.step(action)
                vel = obs["nonviz_sensor"][3:6]
                record_total += time.time() - start
                record_num += 1
            print("Play mode: reward %f" % rew)
        if callback is not None:
            callback(prev_obs, obs, action, rew, env_done, info)
        # process pygame events
        key_codes = env.get_key_pressed(relevant_keys)
        # print("Key codes", key_codes)
        pressed_keys = []

        for key in key_codes:
            print(key)
            if key == ord('r'):
                do_restart = True
            if key == ord('j'):
                env.robot.turn_left()
                continue
            if key == ord('l'):
                env.robot.turn_right()
                continue
            if key == ord('i'):
                env.robot.move_forward()
                continue
            if key == ord('k'):
                env.robot.move_backward()
                continue
            if key not in relevant_keys:
                continue
            # test events, set key states
            # print(relevant_keys)
            pressed_keys.append(key)

            # print(pressed_keys)
            '''
            elif event.type == pygame.QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                video_size = event.size
                screen = pygame.display.set_mode(video_size)
                print(video_size)
            '''