import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.utils as vutils
from datasets import Places365Dataset
from completion import CompletionNet
from tensorboard import SummaryWriter
from datetime import datetime

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--debug'  , action='store_true', help='debug mode')
    parser.add_argument('--imgsize'  ,type=int, default = 256, help='image size')
    parser.add_argument('--batchsize'  ,type=int, default = 76, help='batchsize')
    parser.add_argument('--workers'  ,type=int, default = 6, help='number of workers')
    parser.add_argument('--nepoch'  ,type=int, default = 50, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate, default=0.002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--outf', type=str, default="ae", help='output folder')



    opt = parser.parse_args()
    print(opt)
    
    writer = SummaryWriter(opt.outf + '/runs/'+datetime.now().strftime('%B%d  %H:%M:%S'))
    
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass


    if opt.debug:
        d = Places365Dataset(root = opt.dataroot, transform=transforms.Compose([
                                   transforms.Scale(opt.imgsize),
                                   transforms.ToTensor(),
                               ]), train = False)
    else:
        d = Places365Dataset(root = opt.dataroot, transform=transforms.Compose([
                                   transforms.Scale(opt.imgsize),
                                   transforms.ToTensor(),
                               ]))
    
    
    cudnn.benchmark = True
    
    dataloader = torch.utils.data.DataLoader(d, batch_size=opt.batchsize, shuffle=True, num_workers=int(opt.workers), drop_last = True, pin_memory = True)
    
    img = Variable(torch.rand(opt.batchsize,3, 256, 256)).cuda()
    patch = Variable(torch.rand(opt.batchsize,3, 128, 128)).cuda()
    comp = CompletionNet()
    comp =  torch.nn.DataParallel(comp).cuda()
    comp.apply(weights_init)
    l2 = nn.MSELoss()
    optimizer = torch.optim.Adam(comp.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))

    
    for epoch in range(opt.nepoch):
        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
            img.data.copy_(data)
            recon = comp(img)
            loss = l2(recon, img)
            loss.backward()
            optimizer.step()
            #print(img, recon)
            print('[%d/%d][%d/%d] loss: %f' % (epoch, opt.nepoch, i, len(dataloader), loss.data[0]))
            if i%500 == 0:
                visual = torch.cat([img.data, recon.data], 3)
                #vutils.save_image(visual, '%s/results%d_%d.png' % (opt.outf, epoch, i), nrow=1)
                
                visual = vutils.make_grid(visual, normalize=True)
                writer.add_image('image', visual, i + epoch * len(dataloader))
            
            if i%10 == 0:
                writer.add_scalar('loss', loss.data[0], i + epoch * len(dataloader))
        
        torch.save(comp.state_dict(), '%s/comp_epoch%d.pth' % (opt.outf, epoch))

            
if __name__ == '__main__':
    main()
    
        