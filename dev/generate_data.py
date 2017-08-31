import numpy as np
import ctypes as ct
import cv2
import sys
import argparse
from datasets import ViewDataSet3D
from completion import CompletionNet
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
from numpy import cos, sin
import utils
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
from multiprocessing import Pool, cpu_count

dll=np.ctypeslib.load_library('render_cuda','.')


def render(imgs, depths, pose, poses):
    global fps
    t0 = time.time()
    showsz = imgs[0].shape[0]
    show=np.zeros((showsz,showsz * 2,3),dtype='uint8')
    target_depth = np.zeros((showsz,showsz * 2)).astype(np.uint32)
    
    target_depth[:] = 65535
    for i in range(len(imgs)):

        pose_after = pose.dot(np.linalg.inv(poses[0])).dot(poses[i]).astype(np.float32)
        #print('after',pose_after)

        dll.render(ct.c_int(imgs[i].shape[0]),
                   ct.c_int(imgs[i].shape[1]),
                   imgs[i].ctypes.data_as(ct.c_void_p),
                   depths[i].ctypes.data_as(ct.c_void_p),
                   pose_after.ctypes.data_as(ct.c_void_p),
                   show.ctypes.data_as(ct.c_void_p),
                   target_depth.ctypes.data_as(ct.c_void_p)
                  )
        
    return show, target_depth

## CPU heavy
def generate_data(args):
    idx  = args[0]
    print(idx)
    d    = args[1]
    outf = args[2]
    data = d[idx]   ## This operation stalls 95% of the time, CPU heavy
    sources = data[0]
    target = data[1]
    source_depths = data[2]
    poses = [item.numpy() for item in data[-1]]
    show, depth =  render(sources, source_depths, poses[0], poses)
    np.savez(file = "%s/data_%d.npz" % (outf, idx), source = show, depth = depth, target = target)

if __name__=='__main__':
    time_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug'  , action='store_true', help='debug mode')
    parser.add_argument('--dataroot'  , required = True, help='dataset path')
    parser.add_argument('--outf'  , type = str, default = '', help='path of output folder')
    opt = parser.parse_args()
    d = ViewDataSet3D(root=opt.dataroot, transform = np.array, mist_transform = np.array, seqlen = 5, off_3d = False)
    
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    cpu_c = cpu_count()
    p = Pool(10)

    # On a 8 core CPU, this gives ~4.2x boost
    p.map(generate_data, [(idx, d, opt.outf) for idx in range(200000,300000)])


    #plt.figure(1)
    #plt.imshow(show)
    #plt.figure(2)
    #plt.imshow(target)
    #plt.figure(3)
    #plt.imshow(depth)
    #plt.show()

    
    print('Total time %s', str(time.time() - time_start))
