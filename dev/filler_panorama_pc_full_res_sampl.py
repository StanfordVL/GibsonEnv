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
from datasets import PairDataset
from completion import CompletionNet, Discriminator2
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

def crop(source, source_depth, target):
    bs = source.size(0)
    source_cropped = Variable(torch.zeros(4*bs, 3, 256, 256)).cuda()
    source_depth_cropped = Variable(torch.zeros(4*bs, 2, 256, 256)).cuda()
    target_cropped = Variable(torch.zeros(4*bs, 3, 256, 256)).cuda()
    
    for i in range(bs):
        for j in range(4):
            idx = i * 4 + j
            centerx = np.random.randint(128, 1024 - 128)
            centery = np.random.randint(128, 1024 * 2 - 128)
            source_cropped[idx] = source[i, :, centerx-128:centerx + 128, centery - 128:centery + 128]
            source_depth_cropped[idx] = source_depth[i, :, centerx-128:centerx + 128, centery - 128:centery + 128]
            target_cropped[idx] = target[i, :, centerx-128:centerx + 128, centery - 128:centery + 128]
           
    return source_cropped, source_depth_cropped, target_cropped, 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--debug'  , action='store_true', help='debug mode')
    parser.add_argument('--imgsize'  ,type=int, default = 256, help='image size')
    parser.add_argument('--batchsize'  ,type=int, default = 20, help='batchsize')
    parser.add_argument('--workers'  ,type=int, default = 9, help='number of workers')
    parser.add_argument('--nepoch'  ,type=int, default = 50, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate, default=0.002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--outf', type=str, default="filler_pano_pc_full", help='output folder')
    parser.add_argument('--model', type=str, default="", help='model path')
    parser.add_argument('--cepoch'  ,type=int, default = 0, help='current epoch')

    mean = torch.from_numpy(np.array([0.57441127,  0.54226291,  0.50356019]).astype(np.float32))

    opt = parser.parse_args()
    print(opt)

    writer = SummaryWriter(opt.outf + '/runs/'+datetime.now().strftime('%B%d  %H:%M:%S'))

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    tf = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    mist_tf = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    
    d = PairDataset(root = opt.dataroot, transform=tf, mist_transform = mist_tf)

    cudnn.benchmark = True

    dataloader = torch.utils.data.DataLoader(d, batch_size=opt.batchsize, shuffle=True, num_workers=int(opt.workers), drop_last = True, pin_memory = False)

    img = Variable(torch.zeros(opt.batchsize,3, 1024, 2048)).cuda()
    maskv = Variable(torch.zeros(opt.batchsize,2, 1024, 2048)).cuda()
    img_original = Variable(torch.zeros(opt.batchsize,3, 1024, 2048)).cuda()
    label = Variable(torch.LongTensor(opt.batchsize * 4)).cuda()

    comp = CompletionNet()
    dis = Discriminator2(pano = False)
    current_epoch = opt.cepoch

    comp =  torch.nn.DataParallel(comp).cuda()
    comp.apply(weights_init)
    dis = torch.nn.DataParallel(dis).cuda()
    dis.apply(weights_init)

    if opt.model != '':
        comp.load_state_dict(torch.load(opt.model))
        #dis.load_state_dict(torch.load(opt.model.replace("G", "D")))
        current_epoch = opt.cepoch

    l2 = nn.MSELoss()
    optimizerG = torch.optim.Adam(comp.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
    optimizerD = torch.optim.Adam(dis.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))

    curriculum = (20000, 30000) # step to start D training and G training, slightly different from the paper
    alpha = 0.0004

    errG_data = 0
    errD_data = 0

    for epoch in range(current_epoch, opt.nepoch):
        for i, data in enumerate(dataloader, 0):
            optimizerG.zero_grad()
            source = data[0]
            source_depth = data[1]
            target = data[2]
            step = i + epoch * len(dataloader)
            
            mask = (torch.sum(source,1)==0).float().unsqueeze(1)
            source += mask.repeat(1,3,1,1) * F.upsample(F.max_pool2d(source, 2), scale_factor=2, mode = 'nearest').data
            
            mask2 = (torch.sum(source,1)==0).float().unsqueeze(1)
            source += mask2.repeat(1,3,1,1) * F.upsample(F.max_pool2d(source, 4), scale_factor=4, mode = 'nearest').data
            
            mask3 = (torch.sum(source,1)==0).float().unsqueeze(1)
            source += mask3.repeat(1,3,1,1) * F.upsample(F.max_pool2d(source, 4), scale_factor=8, mode = 'nearest').data
            
            
            #from IPython import embed; embed()
            
            source_depth = source_depth[:,:,:,0].unsqueeze(1)
            
            #print(source_depth.size(), mask.size())
            source_depth = torch.cat([source_depth, mask], 1)
            
            img.data.copy_(source)
            maskv.data.copy_(source_depth)
            img_original.data.copy_(target)
            
            imgc, maskvc, img_originalc = crop(img, maskv, img_original)
            #from IPython import embed; embed()
            recon = comp(imgc, maskvc)
            loss = l2(recon, img_originalc)
            loss.backward(retain_variables = True)
            
            
            if step > curriculum[1]:
                label.data.fill_(1)
                output = dis(recon)
                errG = alpha * F.nll_loss(output, label)
                errG.backward()
                errG_data = errG.data[0]
                
            optimizerG.step()
             
            # Train D:
            if step > curriculum[0]:
                optimizerD.zero_grad()
                label.data.fill_(0)
                output = dis(recon.detach())
                #print(output)
                errD_fake = alpha * F.nll_loss(output, label)
                errD_fake.backward(retain_variables = True)

                output = dis(img_originalc)
                #print(output)
                label.data.fill_(1)
                errD_real = alpha * F.nll_loss(output, label)
                errD_real.backward()
                optimizerD.step()
                errD_data = errD_real.data[0] + errD_fake.data[0]
            
            
            print('[%d/%d][%d/%d] %d MSEloss: %f G_loss %f D_loss %f' % (epoch, opt.nepoch, i, len(dataloader), step, loss.data[0], errG_data, errD_data))
            
            if i%500 == 0:
                visual = torch.cat([imgc.data, recon.data, img_originalc.data], 3)
                visual = vutils.make_grid(visual, normalize=True)
                writer.add_image('image', visual, step)
                vutils.save_image(visual, '%s/compare%d_%d.png' % (opt.outf, epoch, i), nrow=1)

            if i%10 == 0:
                writer.add_scalar('MSEloss', loss.data[0], step)
                writer.add_scalar('G_loss', errG_data, step)
                writer.add_scalar('D_loss', errD_data, step)


            if i%10000 == 0:
                torch.save(comp.state_dict(), '%s/compG_epoch%d_%d.pth' % (opt.outf, epoch, i))
                torch.save(dis.state_dict(), '%s/compD_epoch%d_%d.pth' % (opt.outf, epoch, i))
            
if __name__ == '__main__':
    main()
