import numpy as np
import ctypes as ct
import cv2
import sys
import argparse
from gibson.data.datasets import ViewDataSet3D
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
from numpy import cos, sin
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
from multiprocessing import Pool, cpu_count
from scipy.signal import convolve2d
from scipy.interpolate import griddata
import scipy
import torch.nn.functional as F
from torchvision import transforms


dll=np.ctypeslib.load_library('../core/render/render_cuda_f','.')

# In[6]:

def render(imgs, depths, pose, poses, tdepth):
    global fps
    t0 = time.time()
    showsz = imgs[0].shape[0]
    nimgs = len(imgs)
    show=np.zeros((showsz,showsz * 2,3),dtype='uint8')
    target_depth = (128 * tdepth[:,:,0]).astype(np.float32)

    imgs = np.array(imgs)
    depths = np.array(depths).flatten()

    rpose = np.eye(4).astype(np.float32)
    rpose[0,-1] = 1
    rpose[1,-1] = 2
    rpose[2,-1] = 1    
    
    pose_after = [rpose.dot(poses[i]).astype(np.float32) for i in range(len(imgs))]
    pose_after = np.array(pose_after)


    dll.render(ct.c_int(len(imgs)),
               ct.c_int(imgs[0].shape[0]),
               ct.c_int(imgs[0].shape[1]),
               ct.c_int(1),
               ct.c_int(1),
               imgs.ctypes.data_as(ct.c_void_p),
               depths.ctypes.data_as(ct.c_void_p),
               pose_after.ctypes.data_as(ct.c_void_p),
               show.ctypes.data_as(ct.c_void_p),
               target_depth.ctypes.data_as(ct.c_void_p)
              )

    return show, target_depth

# In[7]:

def generate_data(args):

    idx  = args[0]
    print(idx)
    d    = args[1]
    outf = args[2]
    filename = "%s/data_%d.npz" % (outf, idx)
    if not os.path.isfile(filename):
        print(idx)
        data = d[idx]   ## This operation stalls 95% of the time, CPU heavy
        sources = data[0]
        target = data[1]
        source_depths = data[2]
        target_depth = data[3]
        #target_normal = data[5]
        poses = [item.numpy() for item in data[-1]]

        show, _ =  render(sources, source_depths, poses[0], poses, target_depth)
        print(show.shape)

        Image.fromarray(show).save('%s/show%d.png' % (outf, idx))
        Image.fromarray(target).save('%s/target%d.png' % (outf, idx))

        np.savez(file = filename, source = show, depth = target_depth, target = target)

    return


parser = argparse.ArgumentParser()
parser.add_argument('--debug'  , action='store_true', help='debug mode')
parser.add_argument('--dataroot'  , required = True, help='dataset path')
parser.add_argument('--outf'  , type = str, default = '', help='path of output folder')
opt = parser.parse_args()


d = ViewDataSet3D(root=opt.dataroot, transform = np.array, mist_transform = np.array, seqlen = 5, off_3d = False, train = False)
print(len(d))

p = Pool(10)
p.map(generate_data, [(idx, d, opt.outf) for idx in range(len(d))])

#for i in range(len(d)):
#    filename = "%s/data_%d.npz" % (opt.outf, i)
#    print(filename)
#    if not os.path.isfile(filename):
#        generate_data([i, d, opt.outf])

