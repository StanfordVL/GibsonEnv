#!/usr/bin/python
from gibson.envs.mobile_robots_env import TurtlebotNavigateEnv
from gibson.utils.play import play
import argparse
import os
import rospy
from std_msgs.msg import Float32, Int64
from geometry_msgs.msg import Twist
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
config_file = os.path.join(path, 'turtlebot_nonviz.yaml')
print(config_file)

cmdx = 0.0
cmdy = 0.0

def callback(data):
    global cmdx, cmdy
    cmdx = data.linear.x/10.0 - data.angular.z / 50.0
    cmdy = data.linear.x/10.0 + data.angular.z / 50.0

def callback_step(data):
    global cmdx, cmdy
    env.step([cmdx, cmdy])


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=config_file)
args, unknown = parser.parse_known_args()

rospy.init_node('gibson-sim')

env = TurtlebotNavigateEnv(config = args.config)
print(env.config)

obs = env.reset()
rospy.Subscriber("/mobile_base/commands/velocity", Twist, callback)
rospy.Subscriber("/gibson_ros/sim_clock", Int64, callback_step)

rospy.spin()
