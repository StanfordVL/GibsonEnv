from __future__ import print_function

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import shutil
import time

cudnn.benchmark = True


class AdaptiveNorm2d(nn.Module):
    def __init__(self, nchannel, momentum=0.05):
        super(AdaptiveNorm2d, self).__init__()
        self.nm = nn.BatchNorm2d(nchannel, momentum=momentum)
        self.w0 = nn.Parameter(torch.zeros(1))
        self.w1 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.w0.repeat(x.size()) * self.nm(x) + self.w1.repeat(x.size()) * x


class CompletionNet(nn.Module):
    def __init__(self, norm=AdaptiveNorm2d, nf=64, skip_first_bn=False):
        super(CompletionNet, self).__init__()

        self.nf = nf
        alpha = 0.05
        if skip_first_bn:
            self.convs = nn.Sequential(
                nn.Conv2d(5, nf // 4, kernel_size=5, stride=1, padding=2),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf // 4, nf, kernel_size=5, stride=2, padding=2),
                norm(nf, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=2),
                norm(nf, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf, nf * 4, kernel_size=5, stride=2, padding=1),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, dilation=2, padding=2),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, dilation=4, padding=4),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, dilation=8, padding=8),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, dilation=16, padding=16),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, dilation=32, padding=32),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.ConvTranspose2d(nf * 4, nf, kernel_size=4, stride=2, padding=1),
                norm(nf, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
                norm(nf, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.ConvTranspose2d(nf, nf // 4, kernel_size=4, stride=2, padding=1),
                norm(nf // 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf // 4, nf // 4, kernel_size=3, stride=1, padding=1),
                norm(nf // 4, momentum=alpha),
                nn.LeakyReLU(0.1),
                nn.Conv2d(nf // 4, 3, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(5, nf // 4, kernel_size=5, stride=1, padding=2),
                norm(nf//4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf // 4, nf, kernel_size=5, stride=2, padding=2),
                norm(nf, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=2),
                norm(nf, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf, nf * 4, kernel_size=5, stride=2, padding=1),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, dilation=2, padding=2),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, dilation=4, padding=4),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, dilation=8, padding=8),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, dilation=16, padding=16),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, dilation=32, padding=32),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1),
                norm(nf * 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.ConvTranspose2d(nf * 4, nf, kernel_size=4, stride=2, padding=1),
                norm(nf, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
                norm(nf, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.ConvTranspose2d(nf, nf // 4, kernel_size=4, stride=2, padding=1),
                norm(nf // 4, momentum=alpha),
                nn.LeakyReLU(0.1),

                nn.Conv2d(nf // 4, nf // 4, kernel_size=3, stride=1, padding=1),
                norm(nf // 4, momentum=alpha),
                nn.LeakyReLU(0.1),
                nn.Conv2d(nf // 4, 3, kernel_size=3, stride=1, padding=1),
            )

    def forward(self, x, mask):
        return F.tanh(self.convs(torch.cat([x, mask], 1)))


def identity_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.fill_(0)
        o, i, k1, k2 = m.weight.data.size()
        cx, cy = k1 // 2, k2 // 2
        nc = min(o, i)
        print(nc)
        for i in range(nc):
            m.weight.data[i, i, cx, cy] = 1
        m.bias.data.fill_(0)

        if m.stride[0] == 2:
            for i in range(nc):
                m.weight.data[i + nc, i, cx + 1, cy] = 1
                m.weight.data[i + nc * 2, i, cx, cy + 1] = 1
                m.weight.data[i + nc * 3, i, cx + 1, cy + 1] = 1

    elif classname.find('ConvTranspose2d') != -1:
        o, i, k1, k2 = m.weight.data.size()
        nc = min(o, i)
        cx, cy = k1 // 2 - 1, k2 // 2 - 1
        m.weight.data.fill_(0)
        for i in range(nc):
            m.weight.data[i, i, cx, cy] = 1
            m.weight.data[i + nc, i, cx + 1, cy] = 1
            m.weight.data[i + nc * 2, i, cx, cy + 1] = 1
            m.weight.data[i + nc * 3, i, cx + 1, cy + 1] = 1
        m.bias.data.fill_(0)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)


class Perceptual(nn.Module):
    def __init__(self, features):
        super(Perceptual, self).__init__()
        self.features = features

    def forward(self, x):
        bs = x.size(0)
        x = self.features[0](x)
        x = self.features[1](x)
        x = self.features[2](x)
        x0 = x.view(bs, -1, 1)
        x = F.relu(x)
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        x = self.features[7](x)
        x1 = x.view(bs, -1, 1)
        x = F.relu(x)
        x = self.features[9](x)
        x = self.features[10](x)
        x = self.features[11](x)
        x = self.features[12](x)
        x2 = x.view(bs, -1, 1)
        x = F.relu(x)
        x = self.features[14](x)
        x = self.features[15](x)
        x = self.features[16](x)
        x = self.features[17](x)
        x = self.features[18](x)
        x = self.features[19](x)
        x3 = x.view(bs, -1, 1)
        x = F.relu(x)
        x = self.features[21](x)
        x = self.features[22](x)
        x = self.features[23](x)
        x = self.features[24](x)
        x = self.features[25](x)
        x = self.features[26](x)
        x4 = x.view(bs, -1, 1)

        perfeat = torch.cat([x0, x1, x2, x3, x4], 1)

        return perfeat
