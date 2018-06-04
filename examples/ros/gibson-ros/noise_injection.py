#!/usr/bin/python
from gibson.envs.mobile_robots_env import TurtlebotNavigateEnv
from gibson.utils.play import play
import argparse
import os
import rospy
from std_msgs.msg import Float32, Int64
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
import rospkg
import numpy as np
from cv_bridge import CvBridge

import tf

class NoiseInjectionNode:
    def __init__(self):
        rospy.init_node('gibson-sim-noise')
        self.register_callback()
        self.pub = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=10)

        self.x, self.z = 0, 0


    def register_callback(self):
        rospy.Subscriber("/mobile_base/commands/velocity_raw", Twist, self.callback)

    def callback(self, msg):
        print(msg)

        if not (msg.linear.x == 0 and msg.angular.y == 0):
            msg.linear.x += self.x
            msg.angular.z += self.z

            self.x = self.x * 0.95 + np.random.normal(0, 0.2) * 0.05
            self.z = self.z * 0.95 + np.random.normal(0, 1) * 0.05

        #pass through for zero velocity command

        self.pub.publish(msg)


    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = NoiseInjectionNode()
    node.run()
