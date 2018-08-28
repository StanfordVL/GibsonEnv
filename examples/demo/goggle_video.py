import numpy as np
import cv2
import sys
#!/usr/bin/python
import argparse
import os
from torchvision import datasets, transforms
import gym
import sys
import time
import time
import pygame
import pybullet as p
from gibson.core.render.profiler import Profiler
from gibson.learn.completion import CompletionNet
import cv2
import torch.nn as nn
import torch
from torch.autograd import Variable
from gibson import assets

assets_file_dir = os.path.dirname(assets.__file__)


cap = cv2.VideoCapture(sys.argv[1])

def load_model():
    comp = CompletionNet(norm=nn.BatchNorm2d, nf=64)
    comp = nn.DataParallel(comp).cuda()
    comp.load_state_dict(
        torch.load(os.path.join(assets_file_dir, "unfiller_rgb.pth")))

    model = comp.module
    model.eval()
    return model

model = load_model()
imgv = Variable(torch.zeros(1, 3, 256, 256), volatile=True).cuda()
maskv = Variable(torch.zeros(1, 2, 256, 256), volatile=True).cuda()

while(cap.isOpened()):
    ret, frame = cap.read()
    w,h,_ = frame.shape
    frame = frame.transpose(1,0,2)
    frame = cv2.resize(frame[h//2 - w//2:h//2 + w//2, :], (256,256))
    tf = transforms.ToTensor()
    source = tf(frame)
    imgv.data.copy_(source)
    maskv[:,0,:,:].data.fill_(0.05)
    maskv[:,1,:,:].data.fill_(1)
    #print(source)
    #print(imgv.size(), maskv.size()) 
    recon = model(imgv, maskv)
    goggle_img = (recon.data.clamp(0, 1).cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)

    cv2.imshow('frame',goggle_img)
    if cv2.waitKey(16) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()