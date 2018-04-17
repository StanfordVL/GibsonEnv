
# coding: utf-8

# In[1]:

from __future__ import print_function
import numpy as np
import ctypes as ct
import os
import cv2
import sys
import torch
import argparse
import time
import utils
import transforms3d
import json
import zmq

from PIL import Image

from torchvision import datasets, transforms
from torch.autograd import Variable
from numpy import cos, sin
from profiler import Profiler
from multiprocessing.dummy import Process

from gibson.data.datasets import ViewDataSet3D


# In[2]:


# In[3]:

cuda_pc = np.ctypeslib.load_library('render_cuda_f', '.')


# In[ ]:

d = ViewDataSet3D(root='/home/fei/Downloads/highres_tiny/', transform = np.array, mist_transform = np.array, seqlen = 2, off_3d = False, train = True)


# In[ ]:

scene_dict = dict(zip(d.scenes, range(len(d.scenes))))

model_id = scene_dict.keys()[1]
scene_id = scene_dict[model_id]

uuids, rts = d.get_scene_info(scene_id)
print(uuids, rts)

targets = []
sources = []
source_depths = []
poses = []

for k,v in uuids:
    #print(k,v)
    data = d[v]
    source = data[0][0]
    target = data[1]
    target_depth = data[3]
    source_depth = data[2][0]
    pose = data[-1][0].numpy()
    targets.append(target)
    poses.append(pose)
    sources.append(target)
    source_depths.append(target_depth)


# In[ ]:

cpose = np.eye(4)


# In[ ]:

target_poses = rts
imgs = sources
depths = source_depths
relative_poses = np.copy(target_poses)
for i in range(len(relative_poses)):
    relative_poses[i] = np.dot(np.linalg.inv(relative_poses[i]), target_poses[0])

poses_after = [cpose.dot(np.linalg.inv(relative_poses[i])).astype(np.float32) for i in range(len(imgs))]
pose_after_distance = [np.linalg.norm(rt[:3,-1]) for rt in poses_after]
print(np.sort(pose_after_distance)[:3])
topk = (np.argsort(pose_after_distance))[:3]


# In[ ]:

imgs_topk = np.array([imgs[i] for i in topk[1:]])
depths_topk = np.array([depths[i] for i in topk[1:]]).flatten()
relative_poses_topk = [relative_poses[i] for i in topk[1:]]


# In[ ]:

pose = cpose
poses = relative_poses_topk
poses_after = [
    pose.dot(np.linalg.inv(poses[i])).astype(np.float32)
    for i in range(len(imgs_topk))]



# In[ ]:

showsz = 1024
show   = np.zeros((showsz,showsz * 2,3),dtype='uint8')

this_depth = (128 * depths[topk[0]]).astype(np.float32)
for i in range(5):
    with Profiler("Render pointcloud"):
        cuda_pc.render(ct.c_int(len(imgs_topk)),
                       ct.c_int(imgs_topk[0].shape[0]),
                       ct.c_int(imgs_topk[0].shape[1]),
                       ct.c_int(1),
                       imgs_topk.ctypes.data_as(ct.c_void_p),
                       depths_topk.ctypes.data_as(ct.c_void_p),
                       np.asarray(poses_after, dtype = np.float32).ctypes.data_as(ct.c_void_p),
                       show.ctypes.data_as(ct.c_void_p),
                       this_depth.ctypes.data_as(ct.c_void_p)
                      )

    Image.fromarray(show).save('imgs/test%04d.png' % i)
