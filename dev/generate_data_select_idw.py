
# coding: utf-8

# In[5]:

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
from scipy.interpolate import griddata
import scipy
import torch.nn.functional as F
from torchvision import transforms
dll=np.ctypeslib.load_library('render_cuda_f','.')


# In[6]:

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


# In[7]:

def generate_data(args):
   
    idx  = args[0]
    print(idx)
    d    = args[1]
    #outf = args[2]
    print(idx)
    data = d[idx]   ## This operation stalls 95% of the time, CPU heavy
    sources = data[0]
    target = data[1]
    source_depths = data[2]
    target_depth = data[3]
    poses = [item.numpy() for item in data[-1]]
    show, _ =  render(sources, source_depths, poses[0], poses, target_depth)
    #np.savez(file = "%s/data_%d.npz" % (outf, idx), source = show, depth = depth, target = target)
    
    return show, target_depth, target


# In[8]:

parser = argparse.ArgumentParser()
parser.add_argument('--debug'  , action='store_true', help='debug mode')
parser.add_argument('--dataroot'  , required = True, help='dataset path')
parser.add_argument('--outf'  , type = str, default = '', help='path of output folder')
opt = parser.parse_args(['--dataroot', '/home/fei/Downloads/highres_tiny', '--outf', '.'])
d = ViewDataSet3D(root=opt.dataroot, transform = np.array, mist_transform = np.array, seqlen = 5, off_3d = False)


# In[9]:

idx = 210
show, depth, target = generate_data([idx, d])


# In[10]:



# In[11]:


show_tensor = torch.zeros(4,3,1024,2048)
tf = transforms.ToTensor()
for i in range(4):
    show_tensor[i, :, :, :] = tf(show[i])
    
show_tensor_v = Variable(show_tensor.cuda())
mask = (torch.sum(show_tensor_v, 1, keepdim = True) > 0).float().repeat(1,3,1,1)


# In[13]:

nlayer = 11
convs = torch.nn.Conv2d(3, nlayer*3, nlayer, padding=nlayer//2).cuda()
convs.bias.data.fill_(0)
convs.weight.data.fill_(0)

print(convs.weight.data.size())
for i in range(nlayer):
    radius = i // 2
    for c in range(3):
        convs.weight.data[i + c * nlayer, c, nlayer//2-radius:nlayer//2+radius + 1, nlayer//2-radius:nlayer//2+radius + 1] = 1


# In[14]:

conved = convs(show_tensor_v)
conved_mask = convs(mask)
avg = conved / conved_mask
avg[avg != avg] = 0
avg = avg.view(4,3,nlayer,1024,2048)
avg.size()
img = Variable(torch.zeros(4,3,1024,2048)).cuda()
for i in range(nlayer):
    img[img == 0] = avg[:,:,i,:,:][img == 0]


# In[15]:

def gkern(kernlen=11, nsig=2):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(scipy.stats.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


# In[16]:

conv_mask = torch.nn.Conv2d(1, 1, 11, padding=5).cuda()
conv_mask.bias.data.fill_(0)
conv_mask.weight.data[0,0,:,:] = torch.from_numpy(gkern())
density = conv_mask(conv_mask(conv_mask(mask[:,0:1,:,:])))


# In[17]:

selection = F.softmax(5 * density.view(4,1024 * 2048).transpose(1,0)).transpose(1,0).contiguous().view(4, 1024, 2048)
occu = (torch.sum(img,1)>0).float()
selection[occu == 0] = 0
selection = (selection / torch.sum(selection, 0, keepdim = True)).view(4,1,1024,2048).repeat(1,3,1,1)

img_combined = torch.sum(img * selection, 0)


# In[18]:

img_numpy = (img_combined.cpu().data.numpy().transpose(1,2,0) * 255).astype(np.uint8)
Image.fromarray(img_numpy).save('interpolated%d.png' % idx)



