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

showsz = 256
mousex,mousey=0.5,0.5
changed=True
pitch,yaw,x,y,z = 0,0,0,0,0
roll = 0
org_pitch, org_yaw, org_x, org_y, org_z = 0,0,0,0,0
org_roll = 0
mousedown = False
clickstart = (0,0)
fps = 0

def show(img, depth):
    cv2.namedWindow('show')
    showimg = True
    changed = True
    depth = (depth * 255 * 5).astype(np.uint8)

    while True:
        if changed:
            if showimg:
                show = img
            else:
                show = depth
            cv2.imshow('show', show)
        cmd = cv2.waitKey(10)%256
        if cmd == ord('c'):
            showimg = not showimg
            changed = True

        if cmd == ord('q'):
            break




if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug'  , action='store_true', help='debug mode')
    parser.add_argument('--dataroot'  , required = True, help='dataset path')
    parser.add_argument('--idx'  , type = int, default = 0, help='index of data')
    opt = parser.parse_args()
    d = ViewDataSet3D(root=opt.dataroot, transform = np.array, mist_transform = np.array, seqlen = 2, off_3d = False)
    idx = opt.idx
    source = d[idx][0][0]
    target = d[idx][1]
    source_depth = d[idx][2][0]
    pose = d[idx][-1][0].numpy()
    model = None
    print(source.shape, source_depth.shape)
    show(source, source_depth)
    print(np.mean(source_depth))
