import argparse
import os
import re
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.utils as vutils
from datasets import ViewDataSet3D
from completion import CompletionNet, Discriminator
from tensorboard import SummaryWriter
from datetime import datetime
import vision_utils
import torch.nn.functional as F


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


def generate_patches_position(input_imgs):
    # create patch position
    batchsize = input_imgs.size(0)
    imgsize = input_imgs.size(2)

    center1 = np.zeros((batchsize, 2)).astype(np.int64)
    center2 = np.zeros((batchsize, 2)).astype(np.int64)

    hole = np.zeros((batchsize, 2)).astype(np.int64)

    for i in range(batchsize):
        center1[i,0] = np.random.randint(64, imgsize - 64)
        center1[i,1] = np.random.randint(64, imgsize * 2 - 64)
        center2[i,0] = np.random.randint(64, imgsize - 64)
        center2[i,1] = np.random.randint(64, imgsize * 2 - 64)
        hole[i,0] = np.random.randint(48,64 + 1)
        hole[i,1] = np.random.randint(48,64 + 1)

    return hole, center1, center2

def generate_patches(input_imgs, center):
    # create patch, generate new Variable from variable
    batchsize = input_imgs.size(0)
    patches = Variable(torch.zeros(batchsize, 3, 128, 128)).cuda()
    for i in range(batchsize):
        patches[i] = input_imgs[i, :, center[i,0] - 64 : center[i,0] + 64, center[i,1] - 64 : center[i,1] + 64]
    return patches

def prepare_completion_input(input_imgs, center, hole, mean):
    # fill in mean value into holes
    batchsize = input_imgs.size(0)
    img_holed = input_imgs.clone()
    mask = torch.zeros(batchsize, 1, input_imgs.size(2), input_imgs.size(3))
    for i in range(batchsize):
        img_holed[i, :, center[i,0] - hole[i,0] : center[i,0] + hole[i,0], center[i,1] - hole[i,1] : center[i,1] + hole[i,1]] = mean.view(3,1,1).repeat(1,hole[i,0]*2, hole[i,1]*2)
        mask[i, :, center[i,0] - hole[i,0] : center[i,0] + hole[i,0], center[i,1] - hole[i,1] : center[i,1] + hole[i,1]] = 1

    return img_holed, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--debug'  , action='store_true', help='debug mode')
    parser.add_argument('--imgsize'  ,type=int, default = 256, help='image size')
    parser.add_argument('--batchsize'  ,type=int, default = 20, help='batchsize')
    parser.add_argument('--workers'  ,type=int, default = 6, help='number of workers')
    parser.add_argument('--nepoch'  ,type=int, default = 50, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate, default=0.002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--outf', type=str, default="filler_pano_pc", help='output folder')
    parser.add_argument('--model', type=str, default="", help='model path')
    parser.add_argument('--cepoch'  ,type=int, default = 0, help='current epoch')

    mean = torch.from_numpy(np.array([ 0.45725039,  0.44777581,  0.4146058 ]).astype(np.float32))

    opt = parser.parse_args()
    print(opt)

    writer = SummaryWriter(opt.outf + '/runs/'+datetime.now().strftime('%B%d  %H:%M:%S'))

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    tf = transforms.Compose([
        transforms.Scale(opt.imgsize, opt.imgsize * 2),
        transforms.ToTensor(),
    ])
    
    mist_tf = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    
    d = ViewDataSet3D(root = opt.dataroot, transform=tf, mist_transform = mist_tf, seqlen = 2, debug=opt.debug, off_3d = False, off_pc_render = False)

    cudnn.benchmark = True

    dataloader = torch.utils.data.DataLoader(d, batch_size=opt.batchsize, shuffle=True, num_workers=int(opt.workers), drop_last = True, pin_memory = True)

    img = Variable(torch.zeros(opt.batchsize,3, 256, 512)).cuda()
    maskv = Variable(torch.zeros(opt.batchsize,1, 256, 512)).cuda()
    img_original = Variable(torch.zeros(opt.batchsize,3, 256, 512)).cuda()
    label = Variable(torch.LongTensor(opt.batchsize)).cuda()

    comp = CompletionNet()
    dis = Discriminator(pano = True)
    current_epoch = 0

    comp =  torch.nn.DataParallel(comp).cuda()
    comp.apply(weights_init)
    dis = torch.nn.DataParallel(dis).cuda()
    dis.apply(weights_init)

    if opt.model != '':
        comp.load_state_dict(torch.load(opt.model))
        dis.load_state_dict(torch.load(opt.model.replace("G", "D")))
        current_epoch = opt.cepoch

    l2 = nn.MSELoss()
    optimizerG = torch.optim.Adam(comp.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
    optimizerD = torch.optim.Adam(dis.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))

    curriculum = (30000, 50000) # step to start D training and G training, slightly different from the paper
    alpha = 0.0004

    errG_data = 0
    errD_data = 0

    for epoch in range(current_epoch, opt.nepoch):
        for i, data in enumerate(dataloader, 0):
            optimizerG.zero_grad()
            source = data[-2]
            source_depth = data[-1]
            target = data[1]
            print(source.size(),source_depth.size(),target.size())
            step = i + epoch * len(dataloader)
            
            img.data.copy_(source)
            maskv.data.copy_(source_depth)
            img_original.data.copy_(target)
            recon = comp(img, maskv)
            loss = l2(recon, img_original)
            loss.backward(retain_variables = True)
            
            optimizerG.step()
            
            print('[%d/%d][%d/%d] MSEloss: %f' % (epoch, opt.nepoch, i, len(dataloader), loss.data[0]))
            
            if i%500 == 0:
                visual = torch.cat([source, target], 3)
                visual = vutils.make_grid(visual, normalize=True)
                writer.add_image('image', visual, step)
                vutils.save_image(visual, '%s/compare%d_%d.png' % (opt.outf, epoch, i), nrow=1)

            if i%10 == 0:
                writer.add_scalar('MSEloss', loss.data[0], step)


            if i%10000 == 0:
                torch.save(comp.state_dict(), '%s/compG_epoch%d_%d.pth' % (opt.outf, epoch, i))

            
if __name__ == '__main__':
    main()