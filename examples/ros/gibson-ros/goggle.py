#!/usr/bin/python
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
from gibson import assets
from torchvision import datasets, transforms

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
from gibson.learn.completion import CompletionNet
import cv2
import torch.nn as nn
import torch
from torch.autograd import Variable

rospack = rospkg.RosPack()
path = rospack.get_path('gibson-ros')
assets_file_dir = os.path.dirname(assets.__file__)

class Goggle:
    def __init__(self):
        #self.rgb = None
        rospy.init_node('gibson-goggle')
        self.depth = None
        self.image_pub = rospy.Publisher("/gibson_ros/camera_goggle/rgb/image", Image, queue_size=10)
        self.depth_pub = rospy.Publisher("/gibson_ros/camera_goggle/depth/image", Image, queue_size=10)

        self.bridge = CvBridge()

        self.model = self.load_model()
        self.imgv = Variable(torch.zeros(1, 3, 240, 320), volatile=True).cuda()
        self.maskv = Variable(torch.zeros(1, 2, 240, 320), volatile=True).cuda()
        self.mean = torch.from_numpy(np.array([0.57441127, 0.54226291, 0.50356019]).astype(np.float32))
        self.mean = self.mean.view(3, 1, 1).repeat(1, 240, 320)

        self.rgb_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)


    def load_model(self):
        comp = CompletionNet(norm=nn.BatchNorm2d, nf=64)
        comp = nn.DataParallel(comp).cuda()
        comp.load_state_dict(
            torch.load(os.path.join(assets_file_dir, "unfiller_256.pth")))

        model = comp.module
        model.eval()
        print(model)
        return model

    def rgb_callback(self, data):
        img = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        img = cv2.resize(img, (320,240))
        rows, cols, _ = img.shape

        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
        img = cv2.warpAffine(img, M, (cols, rows))

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        depth = cv2.resize(self.depth, (320,240))
        depth = cv2.warpAffine(depth, M, (cols, rows))

        depth = depth.astype(np.float32) / 1000

        tf = transforms.ToTensor()
        source = tf(img)
        mask = (torch.sum(source[:3, :, :], 0) > 0).float().unsqueeze(0)
        source_depth = tf(np.expand_dims(depth, 2).astype(np.float32))
        mask = torch.cat([source_depth, mask], 0)

        self.imgv.data.copy_(source)
        self.maskv.data.copy_(mask)
        recon = self.model(self.imgv, self.maskv)
        goggle_img = (recon.data.clamp(0, 1).cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        goggle_msg = self.bridge.cv2_to_imgmsg(goggle_img, encoding="rgb8")
        self.image_pub.publish(goggle_msg)

        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="passthrough")
        self.depth_pub.publish(depth_msg)

    def depth_callback(self, data):
        self.depth = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        #cv2.imshow("depth", self.depth)
        #cv2.waitKey(10)
    def run(self):
        rospy.spin()


goggle = Goggle()
goggle.run()
