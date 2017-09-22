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
from scipy.signal import convolve2d
import scipy

dll=np.ctypeslib.load_library('render_cuda_f','.')


def render(imgs, depths, pose, poses, tdepth):
    global fps
    t0 = time.time()
    showsz = imgs[0].shape[0]
    nimgs = len(imgs)
    show=np.zeros((nimgs, showsz,showsz * 2,3),dtype='uint8')
    target_depth = np.zeros((showsz,showsz * 2)).astype(np.uint32)
    target_depth[:] = (tdepth[:,:,0] * 12800).astype(np.int32)
    
    for i in range(len(imgs)):

        pose_after = pose.dot(np.linalg.inv(poses[0])).dot(poses[i]).astype(np.float32)
        #print('after',pose_after)

        dll.render(ct.c_int(imgs[i].shape[0]),
                   ct.c_int(imgs[i].shape[1]),
                   imgs[i].ctypes.data_as(ct.c_void_p),
                   depths[i].ctypes.data_as(ct.c_void_p),
                   pose_after.ctypes.data_as(ct.c_void_p),
                   show[i].ctypes.data_as(ct.c_void_p),
                   target_depth.ctypes.data_as(ct.c_void_p)
                  )
        
    return show, target_depth

def gkern(kernlen=10, nsig=2):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(scipy.stats.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


## CPU heavy
def generate_data(args):
   
    idx  = args[0]
    print(idx)
    d    = args[1]
    outf = args[2]
    print(idx)
    data = d[idx]   ## This operation stalls 95% of the time, CPU heavy
    sources = data[0]
    target = data[1]
    source_depths = data[2]
    target_depth = data[3]
    poses = [item.numpy() for item in data[-1]]
    show, _ =  render(sources, source_depths, poses[0], poses, target_depth)
    
    density = np.zeros((4, 1024, 2048))
    imgs = np.zeros((4,1024,2048,3))
    for i in range(4):
        #print(i)
        mask = np.sum(show[i], axis=2) > 0
        density[i] = convolve2d(mask, gkern(), mode = 'same')
        density[i] = convolve2d(density[i], gkern(), mode = 'same')
        density[i] = convolve2d(density[i], gkern(), mode = 'same')
    

    m = np.argmax(density, axis = 0)
    final = np.zeros((1024, 2048, 3))
    
    for i in range(4):
        final += show[i] * np.expand_dims(m == i, 2)
    
    np.savez(file = "%s/data_%d.npz" % (outf, idx), source = show, depth = target_depth, target = target)
    
    Image.fromarray(final.astype(np.uint8)).save("%s/data_%d.jpg"%(outf, idx))
    
    
    
    return show, target_depth, target
    

if __name__=='__main__':
    time_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug'  , action='store_true', help='debug mode')
    parser.add_argument('--dataroot'  , required = True, help='dataset path')
    parser.add_argument('--outf'  , type = str, default = '', help='path of output folder')
    parser.add_argument('--b', type = int, default = 0)
    opt = parser.parse_args()
    d = ViewDataSet3D(root=opt.dataroot, transform = np.array, mist_transform = np.array, seqlen = 5, off_3d = False)
    
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    cpu_c = cpu_count()
    p = Pool(6)

    # On a 8 core CPU, this gives ~4.2x boost
    p.map(generate_data, [(idx, d, opt.outf) for idx in range(opt.b, 1000000, 20)])

 
    
    print('Total time %s', str(time.time() - time_start))
