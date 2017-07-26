from __future__ import print_function

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
from   torchvision import datasets, transforms
from   torch.autograd import Variable
import torch.nn.functional as F
import shutil
import time

class CompletionNet(nn.Module):

    def __init__(self):
        super(CompletionNet, self).__init__()
            
        self.convs = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, dilation = 2, padding = 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, dilation = 4, padding = 4),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, dilation = 8, padding = 8),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, dilation = 16, padding = 16),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size = 3, stride = 1, padding = 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, mask):
        return self.convs(torch.cat([x, mask], 1))
        
class Discriminator(nn.Module):

    def __init__(self, pano = False):
        super(Discriminator, self).__init__()
        self.pano = pano
        self.convs_local = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 5, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size = 5, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size = 5, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size = 5, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 5, stride = 2, padding = 1),
            nn.ReLU()
        )
        
        self.convs_global = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 5, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size = 5, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size = 5, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size = 5, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 5, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 5, stride = 2, padding = 1),
            nn.ReLU()
        )
        if self.pano:
            self.fc_global = nn.Linear(512 * 3 * 7, 1000)
        else:
            self.fc_global = nn.Linear(512 * 3 * 3, 1000)
        
        self.fc_local = nn.Linear(512 * 3 * 3, 1000)
        self.fc = nn.Linear(2000, 2)
    def forward(self, img, patch):
        x = self.convs_local(patch)
        y = self.convs_global(img)
        
        x = x.view(x.size(0), 512 * 3 * 3)

        if self.pano:
            y = y.view(y.size(0), 512 * 3 * 7)
        else:
            y = y.view(y.size(0), 512 * 3 * 3)
        
        x = F.relu(self.fc_local(x))
        y = F.relu(self.fc_global(y))
        
        x = torch.cat([x,y], 1)
        x = F.log_softmax(self.fc(x))
        
        return x
        
if __name__ == '__main__':

    
    img = Variable(torch.rand(1,3, 256, 256)).cuda()
    patch = Variable(torch.rand(1,3, 128, 128)).cuda()
    
    dis = Discriminator().cuda()
    cls = dis(img, patch)
    print(cls)
    
    x = Variable(torch.rand(1,3, 256, 256)).cuda()
    mask = Variable(torch.rand(1,1, 256, 256)).cuda()
    comp = CompletionNet(with_mask = True).cuda()
    print(comp)
    print(x.size(), comp(x, mask).size())
    
    