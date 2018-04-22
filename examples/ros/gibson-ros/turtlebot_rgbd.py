#!/usr/bin/python
from gibson.envs.mobile_robots_env import TurtlebotNavigateEnv
from gibson.utils.play import play
import argparse
import os
import rospy
from std_msgs.msg import Float32, Int64
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import rospkg
import numpy as np
from cv_bridge import CvBridge

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
config_file = os.path.join(path, 'turtlebot_rgbd.yaml')
print(config_file)

cmdx = 0.0
cmdy = 0.0
image_pub = rospy.Publisher("/gibson_ros/image",Image)
depth_pub = rospy.Publisher("/gibson_ros/depth_raw",Image)

bridge = CvBridge()


def callback(data):
    global cmdx, cmdy
    cmdx = data.linear.x/10.0 - data.angular.z / 50.0
    cmdy = data.linear.x/10.0 + data.angular.z / 50.0

def callback_step(data):
    global cmdx, cmdy, bridge
    obs, _, _, _ = env.step([cmdx, cmdy])
    rgb = obs["rgb_filled"]
    depth = (np.clip(obs["depth"], 0, 10.0) / 10.0 * 255).astype(np.uint8)
    image_message = bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
    image_pub.publish(image_message)
    depth_message = bridge.cv2_to_imgmsg(depth, encoding="mono8")
    depth_pub.publish(depth_message)


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
