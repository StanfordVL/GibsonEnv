from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import json
import codecs
import numpy as np
import progressbar
import sys
import torchvision.transforms as transforms
import utils
import argparse
import ctypes as ct
import json
from numpy.linalg import inv
import pickle
from datasets import Places365Dataset, ViewDataSet3D
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from torch import sin, cos
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from datetime import datetime


class Depth3DGridGen(nn.Module):
    def __init__(self, height, width):
        super(Depth3DGridGen, self).__init__()
        self.height, self.width = height, width
        self.grid = np.zeros( [self.height, self.width, 3], dtype=np.float32)
        self.grid[:,:,0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.height), 0), repeats = self.width, axis = 0).T, 0)
        self.grid[:,:,1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.width), 0), repeats = self.height, axis = 0), 0)
        self.grid[:,:,2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

        self.theta = self.grid[:,:,0] * np.pi/2 + np.pi/2
        self.phi = self.grid[:,:,1] * np.pi

        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)

        self.grid3d = torch.from_numpy(np.zeros( [self.height, self.width, 4], dtype=np.float32))

        self.grid3d[:,:,0] = self.x
        self.grid3d[:,:,1] = self.y
        self.grid3d[:,:,2] = self.z
        self.grid3d[:,:,3] = self.grid[:,:,2]


    def forward(self, depth, transformation):
        
        batchsize = torch.Size([depth.size(0)])
        bs = depth.size(0)
        
        self.batchgrid3d = torch.zeros(batchsize + self.grid3d.size())

        for i in range(depth.size(0)):
            self.batchgrid3d[i] = self.grid3d

        self.batchgrid3d = Variable(self.batchgrid3d)
        self.batchgrid = torch.zeros(batchsize + self.grid.size())

        for i in range(depth.size(0)):
            self.batchgrid[i] = self.grid

        self.batchgrid = Variable(self.batchgrid)

        if depth.is_cuda:
             self.batchgrid3d =  self.batchgrid3d.cuda()
        
        x = self.batchgrid3d[:,:,:,0:1] * depth
        y = self.batchgrid3d[:,:,:,1:2] * depth
        z = self.batchgrid3d[:,:,:,2:3] * depth
                
        points = torch.cat([x,y,z,self.batchgrid3d[:,:,:,3:]], -1)
        points = points.view(bs, 1024 * 2048, 4)
        points = torch.bmm(points, transformation)
        points = points.view(bs, 1024, 2048, 4)
        
        x = points[:,:,:,0:1]
        y = points[:,:,:,1:2]
        z = points[:,:,:,2:3]
        
        r = torch.sqrt(x**2 + y**2 + z**2) + 1e-4
                
        theta = torch.acos(z/r)/(np.pi/2)  - 1
        phi = torch.atan2(y,x)
        phi = phi/np.pi
        output = torch.cat([phi,theta], 3)
        return output

class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()
        nf = 16
        self.nf = nf
        alpha = 0.05
        self.convs = nn.Sequential(
            nn.Conv2d(8, nf, kernel_size = 5, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(nf, nf , kernel_size = 5, stride = 2, padding = 1),
            nn.BatchNorm2d(nf, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf, nf, kernel_size = 5, stride = 2, padding = 1),
            nn.BatchNorm2d(nf, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf, nf * 2, kernel_size = 5, stride = 2, padding = 1),
            nn.BatchNorm2d(nf * 2, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf * 2, nf * 2, kernel_size = 5, stride = 2, padding = 1),
            nn.BatchNorm2d(nf * 2, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf * 2, nf * 2, kernel_size = 5, stride = 2, padding = 1),
            nn.BatchNorm2d(nf * 2, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf * 2, nf * 2, kernel_size = 5, stride = 2, padding = 1),
            nn.ReLU()
        )
        self.lin = nn.Linear(210 * nf, 6)
        
    def forward(self, x):
        return self.lin(self.convs(x).view(-1, 210 * self.nf))
    
    
    
class PoseMat(nn.Module):
    def __init__(self):
        super(PoseMat, self).__init__()
    def forward(self, pose):
        bs = pose.size(0)
        alpha = pose[:,0]
        beta = pose[:,1]
        gamma = pose[:,2]
        
        x = pose[:,3]
        y = pose[:,4]
        z = pose[:,5]
          

        mat0 = torch.stack([cos(alpha) * cos(beta), cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma),
                            cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma),x], 1)
        
        mat1 = torch.stack([sin(alpha) * cos(beta), sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma),
                            sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma),y], 1)

        mat2 = torch.stack([-sin(beta), cos(beta) * sin(gamma), cos(beta) * cos(gamma), z], 1)
        mat3 = Variable(torch.zeros(bs, 4).cuda()) 
        mat = torch.stack([mat0, mat1, mat2, mat3], -1)
        
        return mat
        

#net = PoseNet().cuda()
#input = Variable(torch.rand(1,6,1024,2048)).cuda()
#print(net(input).size())
#
#mat = PoseMat().cuda()
#input = Variable(torch.zeros(1,6).cuda())
#print(mat(input))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



parser = argparse.ArgumentParser()
parser.add_argument('--debug'  , action='store_true', help='debug mode')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--nepoch'  ,type=int, default = 50, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.002, help='learning rate, default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--outf', type=str, default="pose_adj", help='output folder')
parser.add_argument('--model', type=str, default="", help='model path')

opt = parser.parse_args(['--dataroot', '/home/fei/Downloads/highres_tiny/'])



try:
    os.makedirs(opt.outf)
except OSError:
    pass
writer = SummaryWriter(opt.outf + '/runs/'+datetime.now().strftime('%B%d  %H:%M:%S'))

    
tf = transforms.Compose([
    transforms.Scale(1024, 1024 * 2),
    transforms.ToTensor(),
])
    
mist_tf = transforms.Compose([
    transforms.ToTensor(),
])

d = ViewDataSet3D(root=opt.dataroot, debug=opt.debug, transform=tf, mist_transform=mist_tf, seqlen = 2, off_3d = False, off_pc_render = True)
print(len(d))

cudnn.benchmark = True

dataloader = torch.utils.data.DataLoader(d, batch_size=1, shuffle=True, num_workers=1, drop_last = True, pin_memory = False)
gridgen = Depth3DGridGen(1024, 2048)
net = PoseNet().cuda()
mat = PoseMat().cuda()
net.apply(weights_init)


dll2=np.ctypeslib.load_library('occinf','.')
l1 = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr = 0.002, betas = (0.5, 0.999))

for epoch in range(0, 10):
    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()
        step = i + epoch * len(dataloader)
        source = Variable(data[0][0]).cuda()
        target = Variable(data[1]).cuda()
        depth = Variable(data[2][0]).cuda().transpose(2,1).transpose(3,2).float() * 255 * 128
        pose = Variable(data[-1][0]).cuda().float().transpose(2,1)

        depth_np = (depth.cpu().data[0].numpy()[:,:,0]/128.0).astype(np.float32)
        pose_after_np = pose.transpose(2,1).cpu().data[0].numpy().astype(np.float32)
        occ_np = np.zeros((1024,1024 * 2)).astype(np.bool)
        target_depth_np = np.zeros((1024,1024 * 2)).astype(np.uint32)
        target_depth_np[:] = 65535
        
        dll2.occinf(ct.c_int(1024),
            ct.c_int(2048),
            depth_np.ctypes.data_as(ct.c_void_p),
            pose_after_np.ctypes.data_as(ct.c_void_p),
            occ_np.ctypes.data_as(ct.c_void_p),
            target_depth_np.ctypes.data_as(ct.c_void_p)
        )

        mask = torch.from_numpy(occ_np.astype(np.float32))
        mask_v = Variable(mask.view(1,1,1024,2048)).cuda()
        
        grid = gridgen(depth, pose)
        sample = F.grid_sample(target, grid).detach()
        
        mask1 = 1 - (F.grid_sample(mask_v, grid) > 0).float()
        masked_source0 = source * mask1.repeat(1,3,1,1)
        masked_source_pred0 = sample * mask1.repeat(1,3,1,1)
        before = l1(masked_source_pred0, masked_source0)
        input_deck = torch.cat([sample, source, depth.view(1,1,1024,2048)/128.0, mask_v], 1)
        
        adj_pose = net(input_deck)
        pose_mat2 = mat(adj_pose)
        final_mat = torch.bmm(pose, pose_mat2)
        
        grid2 = gridgen(depth, final_mat)
        final_source = F.grid_sample(target, grid2)
        mask2 = 1 - (F.grid_sample(mask_v, grid2) > 0).float()
        
        masked_source_pred = final_source * mask2.repeat(1,3,1,1)
        masked_source = source * mask2.repeat(1,3,1,1)
        
        loss = l1(masked_source_pred, masked_source) + torch.sum(adj_pose ** 2)
        
        after = l1(masked_source_pred, masked_source)
        loss.backward()
        
        optimizer.step()
        print(loss.data[0], before.data[0], after.data[0])
        if i%100 == 0:
            visual = torch.cat([sample.data, masked_source_pred.data, masked_source.data, masked_source_pred.data * 0.5 + masked_source.data * 0.5, masked_source_pred0.data * 0.5 + masked_source0.data * 0.5], 3)
            visual = vutils.make_grid(visual, normalize=True)
            vutils.save_image(visual, '%s/compare%d_%d.png' % (opt.outf, epoch, i), nrow=1)

        if i%10 == 0:
            writer.add_scalar('loss', loss.data[0], step)
            writer.add_scalar('before', before.data[0], step)
            writer.add_scalar('after', after.data[0], step)

        if i%10000 == 0:
                torch.save(net.state_dict(), '%s/net_epoch%d_%d.pth' % (opt.outf, epoch, i))