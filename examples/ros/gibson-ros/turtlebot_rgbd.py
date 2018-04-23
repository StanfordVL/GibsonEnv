#!/usr/bin/python
from gibson.envs.mobile_robots_env import TurtlebotNavigateEnv
from gibson.utils.play import play
import argparse
import os
import rospy
from std_msgs.msg import Float32, Int64
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CameraInfo

import rospkg
import numpy as np
from cv_bridge import CvBridge

import tf

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
image_pub = rospy.Publisher("/gibson_ros/camera/rgb/image",Image, queue_size=10)
depth_pub = rospy.Publisher("/gibson_ros/camera/depth/image",Image, queue_size=10)
depth_raw_pub = rospy.Publisher("/gibson_ros/camera/depth/image_raw",Image, queue_size=10)

camera_info_pub = rospy.Publisher("/gibson_ros/camera/depth/camera_info", CameraInfo, queue_size=10)
bridge = CvBridge()
br = tf.TransformBroadcaster()


def callback(data):
    global cmdx, cmdy
    cmdx = data.linear.x/10.0 - data.angular.z / 50.0
    cmdy = data.linear.x/10.0 + data.angular.z / 50.0

def callback_step(data):
    global cmdx, cmdy, bridge
    obs, _, _, _ = env.step([cmdx, cmdy])
    rgb = obs["rgb_filled"]
    depth = obs["depth"].astype(np.float32)
    depth[depth > 10] = np.nan
    depth[depth < 0.45] = np.nan
    image_message = bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
    depth_raw_image = (obs["depth"] * 1000).astype(np.uint16)
    depth_raw_message = bridge.cv2_to_imgmsg(depth_raw_image, encoding="passthrough")
    depth_message = bridge.cv2_to_imgmsg(depth, encoding="passthrough")

    now = rospy.Time.now()

    image_message.header.stamp = now
    depth_message.header.stamp = now
    depth_raw_message.header.stamp = now

    image_pub.publish(image_message)
    depth_pub.publish(depth_message)
    depth_raw_pub.publish(depth_raw_message)
    msg = CameraInfo(height=256, width=256, distortion_model="plumb_bob", D=[0.0, 0.0, 0.0, 0.0, 0.0],
                     K=[128, 0.0, 128.5, 0.0, 128, 128.5, 0.0, 0.0, 1.0],
                     R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                     P=[128, 0.0, 128.5, -0.0, 0.0, 128, 128.5, 0.0, 0.0, 0.0, 1.0, 0.0])
    msg.header.stamp = now
    msg.header.frame_id="camera_depth_optical_frame"
    camera_info_pub.publish(msg)

    # odometry
    odom = env.get_odom()
    br.sendTransform((odom[0][0], odom[0][1], 0),
                         tf.transformations.quaternion_from_euler(0, 0, odom[-1][-1]),
                         rospy.Time.now(),
                         'base_footprint',
                         "odom")

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
