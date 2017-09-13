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
        nf = 64
        alpha = 0.05
        self.convs = nn.Sequential(
            nn.Conv2d(5, nf, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(),
            nn.Conv2d(nf, nf*2, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(nf*2, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf*2, nf*2, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(nf*2, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf*2, nf*4, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(nf * 4, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf*4, nf * 4, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(nf * 4, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf * 4, nf * 4, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(nf * 4, momentum=alpha),
            nn.ReLU(),

            nn.Conv2d(nf * 4, nf * 4, kernel_size = 3, stride = 1, dilation = 2, padding = 2),
            nn.BatchNorm2d(nf * 4, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf * 4, nf * 4, kernel_size = 3, stride = 1, dilation = 4, padding = 4),
            nn.BatchNorm2d(nf * 4, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf * 4, nf * 4, kernel_size = 3, stride = 1, dilation = 8, padding = 8),
            nn.BatchNorm2d(nf * 4, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf * 4, nf * 4, kernel_size = 3, stride = 1, dilation = 16, padding = 16),
            nn.BatchNorm2d(nf * 4, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf * 4, nf * 4, kernel_size = 3, stride = 1, dilation = 32, padding = 32),
            nn.BatchNorm2d(nf * 4, momentum=alpha),
            nn.ReLU(),

            nn.Conv2d(nf * 4, nf * 4, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(nf * 4, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf * 4, nf * 4, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(nf * 4, momentum=alpha),
            nn.ReLU(),

            nn.ConvTranspose2d(nf * 4, nf * 2, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(nf * 2, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf * 2, nf * 2, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(nf * 2, momentum=alpha),
            nn.ReLU(),

            nn.ConvTranspose2d(nf * 2, nf, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(nf, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf, nf/2, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(nf/2, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf/2, 3, kernel_size = 3, stride = 1, padding = 1),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        return self.convs(torch.cat([x, mask], 1))

class Discriminator(nn.Module):

    def __init__(self, pano = False):
        super(Discriminator, self).__init__()
        alpha = 0.05
        self.pano = pano
        nf = 12
        self.nf = nf
        self.convs_local = nn.Sequential(
            nn.Conv2d(3, nf, kernel_size = 5, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(nf, nf * 2, kernel_size = 5, stride = 2, padding = 1),
            nn.BatchNorm2d(nf * 2, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf * 2, nf * 4, kernel_size = 5, stride = 2, padding = 1),
            nn.BatchNorm2d(nf * 4, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf * 4, nf * 8, kernel_size = 5, stride = 2, padding = 1),
            nn.BatchNorm2d(nf * 8, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf * 8, nf * 8, kernel_size = 5, stride = 2, padding = 1),
            nn.ReLU()
        )

        self.convs_global = nn.Sequential(
            nn.Conv2d(3, nf, kernel_size = 5, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(nf, nf * 2, kernel_size = 5, stride = 2, padding = 1),
            nn.BatchNorm2d(nf * 2, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf * 2, nf * 4, kernel_size = 5, stride = 2, padding = 1),
            nn.BatchNorm2d(nf * 4, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf * 4, nf * 8, kernel_size = 5, stride = 2, padding = 1),
            nn.BatchNorm2d(nf * 8, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf * 8, nf * 8, kernel_size = 5, stride = 2, padding = 1),
            nn.BatchNorm2d(nf * 8, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf * 8, nf * 8, kernel_size = 5, stride = 2, padding = 1),
            nn.ReLU()
        )
        if self.pano:
            self.fc_global = nn.Linear(nf * 8 * 3 * 7, 1000)
        else:
            self.fc_global = nn.Linear(nf * 8 * 3 * 3, 1000)

        self.fc_local = nn.Linear(nf * 8 * 3 * 3, 1000)
        self.fc = nn.Linear(2000, 2)

    def forward(self, img, patch):
        x = self.convs_local(patch)
        y = self.convs_global(img)

        x = x.view(x.size(0), self.nf * 8 * 3 * 3)

        if self.pano:
            y = y.view(y.size(0), self.nf * 8 * 3 * 7)
        else:
            y = y.view(y.size(0), self.nf * 8 * 3 * 3)

        x = F.relu(self.fc_local(x))
        y = F.relu(self.fc_global(y))

        x = torch.cat([x,y], 1)
        x = F.log_softmax(self.fc(x))

        return x

    
    
class Discriminator2(nn.Module):

    def __init__(self, pano = False):
        super(Discriminator2, self).__init__()
        alpha = 0.05
        self.pano = pano
        nf = 64
        self.nf = nf

        self.convs_global = nn.Sequential(
            nn.Conv2d(3, nf, kernel_size = 5, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(nf, nf * 2, kernel_size = 5, stride = 2, padding = 1),
            nn.BatchNorm2d(nf * 2, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf * 2, nf * 4, kernel_size = 5, stride = 2, padding = 1),
            nn.BatchNorm2d(nf * 4, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf * 4, nf * 8, kernel_size = 5, stride = 2, padding = 1),
            nn.BatchNorm2d(nf * 8, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf * 8, nf * 8, kernel_size = 5, stride = 2, padding = 1),
            nn.BatchNorm2d(nf * 8, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf * 8, nf * 8, kernel_size = 5, stride = 2, padding = 1),
            nn.ReLU()
        )
        
        if self.pano:
            self.fc_global = nn.Linear(nf * 8 * 3 * 7, 1000)
        else:
            self.fc_global = nn.Linear(nf * 8 * 3 * 3, 1000)


    def forward(self, img):
        y = self.convs_global(img)

        if self.pano:
            y = y.view(y.size(0), self.nf * 8 * 3 * 7)
        else:
            y = y.view(y.size(0), self.nf * 8 * 3 * 3)

        y = F.relu(self.fc_global(y))

        x = F.log_softmax(y)

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


