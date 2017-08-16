from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import progressbar
import sys
import torchvision.transforms as transforms
import utils
import argparse
import json
from numpy.linalg import inv
import pickle
from datasets import Places365Dataset, ViewDataSet3D

    
if __name__ == '__main__':
    print('test')
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug'  , action='store_true', help='debug mode')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--dataset'  , required = True, help='dataset type')
    opt = parser.parse_args()

    if opt.dataset == 'view3d':
        d = ViewDataSet3D(root=opt.dataroot, debug=opt.debug, seqlen = 2, dist_filter = 0.8, dist_filter2 = 2.0)
        print(len(d))
        means = []
        for i in range(10000):
            idx = np.random.randint(len(d))
            sample = d[idx]
            means.append(np.mean(np.array(sample), axis = (0,1)))
        means = np.array(means).mean(axis = 0)
        print(means/255.0)
    elif opt.dataset == 'places365':
        d = Places365Dataset(root = opt.dataroot)
        print(len(d))
        means = []
        for i in range(10000):
            idx = np.random.randint(len(d))
            sample = d[idx]
            means.append(np.mean(np.array(sample), axis = (0,1)))
        means = np.array(means).mean(axis = 0)
        print(means/255.0)