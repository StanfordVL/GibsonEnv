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
from gibson.data.datasets import PairDataset
from gibson.learn.completion import CompletionNet, identity_init, Perceptual
from tensorboardX import SummaryWriter
from datetime import datetime
import gibson.learn.vision_utils
import torch.nn.functional as F
import torchvision.models as models


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


def crop(source, source_depth, target, zoom = 1, patchsize = 256):
    bs = source.size(0)
    if zoom > 1:
        source = F.avg_pool2d(source, zoom, zoom)
        source_depth = F.avg_pool2d(source_depth, zoom, zoom)
        target = F.avg_pool2d(target, zoom, zoom)

    source_cropped = Variable(torch.zeros(4 * bs, 3, patchsize, patchsize)).cuda()
    source_depth_cropped = Variable(torch.zeros(4 * bs, 2, patchsize, patchsize)).cuda()
    target_cropped = Variable(torch.zeros(4 * bs, 3, patchsize, patchsize)).cuda()
    p = patchsize//2
    for i in range(bs):
        for j in range(4):
            idx = i * 4 + j
            blurry_margin = 1024 // 8 // zoom
            centerx = np.random.randint(blurry_margin + p, 1024//zoom - blurry_margin - p)
            centery = np.random.randint(p, 1024//zoom * 2 - p)
            source_cropped[idx] = source[i, :, centerx - p:centerx + p, centery - p:centery + p]
            source_depth_cropped[idx] = source_depth[i, :, centerx - p:centerx + p, centery - p:centery + p]
            target_cropped[idx] = target[i, :, centerx - p:centerx + p, centery - p:centery + p]

    return source_cropped, source_depth_cropped, target_cropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--imgsize', type=int, default=256, help='image size')
    parser.add_argument('--nf', type=int, default=64, help='number of filters')
    parser.add_argument('--batchsize', type=int, default=20, help='batchsize')
    parser.add_argument('--workers', type=int, default=9, help='number of workers')
    parser.add_argument('--nepoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate, default=0.002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--outf', type=str, default="filler_pano_pc_full", help='output folder')
    parser.add_argument('--model', type=str, default="", help='model path')
    parser.add_argument('--cepoch', type=int, default=0, help='current epoch')
    parser.add_argument('--loss', type=str, default="perceptual", help='l1 only')
    parser.add_argument('--init', type=str, default="iden", help='init method')
    parser.add_argument('--l1', type=float, default=0, help='add l1 loss')
    parser.add_argument('--color_coeff', type=float, default=0, help='add color match loss')
    parser.add_argument('--unfiller', action='store_true', help='debug mode')
    parser.add_argument('--joint', action='store_true', help='debug mode')
    parser.add_argument('--use_depth', action='store_true', default=False, help='debug mode')
    parser.add_argument('--zoom', type=int, default=1, help='debug mode')
    parser.add_argument('--patchsize', type=int, default=256, help='debug mode')

    mean = torch.from_numpy(np.array([0.57441127, 0.54226291, 0.50356019]).astype(np.float32)).clone()
    opt = parser.parse_args()
    print(opt)
    writer = SummaryWriter(opt.outf + '/runs/' + datetime.now().strftime('%B%d  %H:%M:%S'))
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    zoom = opt.zoom
    patchsize = opt.patchsize

    tf = transforms.Compose([
        transforms.ToTensor(),
    ])

    mist_tf = transforms.Compose([
        transforms.ToTensor(),
    ])

    d = PairDataset(root=opt.dataroot, transform=tf, mist_transform=mist_tf)
    d_test = PairDataset(root=opt.dataroot, transform=tf, mist_transform=mist_tf, train=False)

    cudnn.benchmark = True

    dataloader = torch.utils.data.DataLoader(d, batch_size=opt.batchsize, shuffle=True, num_workers=int(opt.workers),
                                             drop_last=True, pin_memory=False)
    dataloader_test = torch.utils.data.DataLoader(d_test, batch_size=opt.batchsize, shuffle=True,
                                                  num_workers=int(opt.workers), drop_last=True, pin_memory=False)

    img = Variable(torch.zeros(opt.batchsize, 3, 1024, 2048)).cuda()
    maskv = Variable(torch.zeros(opt.batchsize, 2, 1024, 2048)).cuda()
    img_original = Variable(torch.zeros(opt.batchsize, 3, 1024, 2048)).cuda()
    label = Variable(torch.LongTensor(opt.batchsize * 4)).cuda()

    comp = CompletionNet(norm=nn.BatchNorm2d, nf=opt.nf)

    current_epoch = opt.cepoch

    comp = torch.nn.DataParallel(comp).cuda()

    if opt.init == 'iden':
        comp.apply(identity_init)
    else:
        comp.apply(weights_init)

    if opt.model != '':
        comp.load_state_dict(torch.load(opt.model))
        # dis.load_state_dict(torch.load(opt.model.replace("G", "D")))
        current_epoch = opt.cepoch

    if opt.unfiller:
        comp2 = CompletionNet(norm=nn.BatchNorm2d, nf=64)
        comp2 = torch.nn.DataParallel(comp2).cuda()
        if opt.model != '':
            comp2.load_state_dict(torch.load(opt.model.replace('G', 'G2')))
        optimizerG2 = torch.optim.Adam(comp2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    l2 = nn.MSELoss()
    # if opt.loss == 'train_init':
    #    params = list(comp.parameters())
    #    sel = np.random.choice(len(params), len(params)/2, replace=False)
    #    params_sel = [params[i] for i in sel]
    #    optimizerG = torch.optim.Adam(params_sel, lr = opt.lr, betas = (opt.beta1, 0.999))
    #
    # else:
    optimizerG = torch.optim.Adam(comp.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    curriculum = (200000, 300000)  # step to start D training and G training, slightly different from the paper
    alpha = 0.004

    errG_data = 0
    errD_data = 0

    vgg16 = models.vgg16(pretrained=False)
    vgg16.load_state_dict(torch.load('vgg16-397923af.pth'))
    feat = vgg16.features
    p = torch.nn.DataParallel(Perceptual(feat)).cuda()

    for param in p.parameters():
        param.requires_grad = False

    imgnet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]).astype(np.float32)).clone()
    imgnet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225]).astype(np.float32)).clone()

    imgnet_mean_img = Variable(imgnet_mean.view(1, 3, 1, 1).repeat(opt.batchsize * 4, 1, patchsize, patchsize)).cuda()
    imgnet_std_img = Variable(imgnet_std.view(1, 3, 1, 1).repeat(opt.batchsize * 4, 1, patchsize, patchsize)).cuda()

    test_loader_enum = enumerate(dataloader_test)
    for epoch in range(current_epoch, opt.nepoch):
        for i, data in enumerate(dataloader, 0):
            optimizerG.zero_grad()
            source = data[0]
            source_depth = data[1]
            target = data[2]
            step = i + epoch * len(dataloader)

            mask = (torch.sum(source[:, :3, :, :], 1) > 0).float().unsqueeze(1)
            # img_mean = torch.sum(torch.sum(source[:,:3,:,:], 2),2) / torch.sum(torch.sum(mask, 2),2).view(opt.batchsize,1)

            source[:, :3, :, :] += (1 - mask.repeat(1, 3, 1, 1)) * mean.view(1, 3, 1, 1).repeat(opt.batchsize, 1, 1024,
                                                                                                2048)
            source_depth = source_depth[:, :, :, 0].unsqueeze(1)
            # print(source_depth.size(), mask.size())
            source_depth = torch.cat([source_depth, mask], 1)
            img.data.copy_(source)
            maskv.data.copy_(source_depth)
            img_original.data.copy_(target)
            imgc, maskvc, img_originalc = crop(img, maskv, img_original, zoom, patchsize)
            # from IPython import embed; embed()
            recon = comp(imgc, maskvc)

            if opt.loss == "train_init":
                loss = l2(recon, imgc[:, :3, :, :])
            elif opt.loss == 'l1':
                loss = l2(recon, img_originalc)
            elif opt.loss == 'perceptual':
                loss = l2(p(recon), p(img_originalc).detach()) + opt.l1 * l2(recon, img_originalc)
            elif opt.loss == 'color_stable':
                loss = l2(p(recon.view(recon.size(0) * 3, 1, patchsize, patchsize).repeat(1, 3, 1, 1)),
                          p(img_originalc.view(img_originalc.size(0) * 3, 1, patchsize, patchsize).repeat(1, 3, 1, 1)).detach())
            elif opt.loss == 'color_correction':
                recon_percept = p((recon - imgnet_mean_img) / imgnet_std_img)
                org_percept = p((img_originalc - imgnet_mean_img) / (imgnet_std_img)).detach()
                loss = l2(recon_percept, org_percept)
                for scale in [32]:
                    img_originalc_patch = img_originalc.view(opt.batchsize * 4, 3, patchsize // scale, scale, patchsize // scale,
                                                             scale).transpose(4, 3).contiguous().view(opt.batchsize * 4,
                                                                                                      3, patchsize // scale,
                                                                                                      patchsize // scale, -1)
                    recon_patch = recon.view(opt.batchsize * 4, 3, patchsize // scale, scale, patchsize // scale, scale).transpose(4,
                                                                                                                     3).contiguous().view(
                        opt.batchsize * 4, 3, patchsize // scale, patchsize // scale, -1)
                    img_originalc_patch_mean = img_originalc_patch.mean(dim=-1)
                    recon_patch_mean = recon_patch.mean(dim=-1)
                    # recon_patch_cov = []
                    # img_originalc_patch_cov = []

                    # for j in range(3):
                    #    recon_patch_cov.append((recon_patch * recon_patch[:,j:j+1].repeat(1,3,1,1,1)).mean(dim=-1))
                    #    img_originalc_patch_cov.append((img_originalc_patch * img_originalc_patch[:,j:j+1].repeat(1,3,1,1,1)).mean(dim=-1))

                    # recon_patch_cov_cat = torch.cat(recon_patch_cov,1)
                    # img_originalc_patch_cov_cat = torch.cat(img_originalc_patch_cov, 1)

                    color_loss = l2(recon_patch_mean,
                                    img_originalc_patch_mean)  # + l2(recon_patch_cov_cat, img_originalc_patch_cov_cat.detach())

                    loss += opt.color_coeff * color_loss

                    print("color loss %f" % color_loss.data[0])

            loss.backward(retain_graph=True)

            if opt.unfiller:
                optimizerG2.zero_grad()

                recon2 = comp2(img_originalc, maskvc)

                if not opt.joint:
                    recon2_percept = p((recon2 - imgnet_mean_img) / imgnet_std_img)
                    recon_percept = p((recon - imgnet_mean_img) / imgnet_std_img)
                    loss2 = l2(recon2_percept, recon_percept.detach())
                else:
                    recon_percept = p((recon - imgnet_mean_img) / imgnet_std_img)
                    z = Variable(torch.zeros(recon_percept.size()).cuda())
                    recon2_percept = p((recon2 - imgnet_mean_img) / imgnet_std_img)

                    loss2 = l2(recon2_percept - recon_percept, z)

                    loss2 += 0.2 * l2(recon2_percept, org_percept)

                for scale in [32]:
                    img_originalc_patch = recon.detach().view(opt.batchsize * 4, 3, patchsize / scale, scale, patchsize / scale,
                                                              scale).transpose(4, 3).contiguous().view(
                        opt.batchsize * 4, 3, patchsize / scale, patchsize / scale, -1)
                    recon2_patch = recon2.view(opt.batchsize * 4, 3, patchsize / scale, scale, patchsize / scale, scale).transpose(
                        4, 3).contiguous().view(opt.batchsize * 4, 3, patchsize / scale, patchsize / scale, -1)
                    img_originalc_patch_mean = img_originalc_patch.mean(dim=-1)
                    recon2_patch_mean = recon2_patch.mean(dim=-1)
                    recon2_patch_cov = []
                    img_originalc_patch_cov = []

                    for j in range(3):
                        recon2_patch_cov.append(
                            (recon2_patch * recon2_patch[:, j:j + 1].repeat(1, 3, 1, 1, 1)).mean(dim=-1))
                        img_originalc_patch_cov.append(
                            (img_originalc_patch * img_originalc_patch[:, j:j + 1].repeat(1, 3, 1, 1, 1)).mean(dim=-1))

                    recon2_patch_cov_cat = torch.cat(recon2_patch_cov, 1)
                    img_originalc_patch_cov_cat = torch.cat(img_originalc_patch_cov, 1)

                    z = Variable(torch.zeros(img_originalc_patch_mean.size()).cuda())
                    if opt.joint:
                        color_loss = l2(recon2_patch_mean - img_originalc_patch_mean, z)
                    else:
                        color_loss = l2(recon2_patch_mean, img_originalc_patch_mean)

                    loss2 += opt.color_coeff * color_loss

                    print("color loss %f" % color_loss.data[0])

                loss2 = loss2 * 0.3
                loss2.backward(retain_graph=True)
                print("loss2 %f" % loss2.data[0])
                optimizerG2.step()

                if i % 10 == 0:
                    writer.add_scalar('MSEloss2', loss2.data[0], step)

            # from IPython import embed; embed()
            if opt.loss == "train_init":
                for param in comp.parameters():
                    if len(param.size()) == 4:
                        # print(param.size())
                        nk = param.size()[2] // 2
                        if nk > 5:
                            param.grad[:nk, :, :, :] = 0

            optimizerG.step()

            print('[%d/%d][%d/%d] %d MSEloss: %f G_loss %f D_loss %f' % (
            epoch, opt.nepoch, i, len(dataloader), step, loss.data[0], errG_data, errD_data))

            if i % 200 == 0:

                test_i, test_data = next(test_loader_enum)
                if test_i > len(dataloader_test) - 5:
                    test_loader_enum = enumerate(dataloader_test)

                source = test_data[0]
                source_depth = test_data[1]
                target = test_data[2]

                mask = (torch.sum(source[:, :3, :, :], 1) > 0).float().unsqueeze(1)

                source[:, :3, :, :] += (1 - mask.repeat(1, 3, 1, 1)) * mean.view(1, 3, 1, 1).repeat(opt.batchsize, 1,
                                                                                                    1024, 2048)
                source_depth = source_depth[:, :, :, 0].unsqueeze(1)
                source_depth = torch.cat([source_depth, mask], 1)
                img.data.copy_(source)
                maskv.data.copy_(source_depth)
                img_original.data.copy_(target)
                imgc, maskvc, img_originalc = crop(img, maskv, img_original, zoom, patchsize)
                comp.eval()
                recon = comp(imgc, maskvc)
                comp.train()

                if opt.unfiller:
                    comp2.eval()
                    # maskvc.data.fill_(0)
                    recon2 = comp2(img_originalc, maskvc)
                    comp2.train()
                    visual = torch.cat([imgc.data[:, :3, :, :], recon.data, recon2.data, img_originalc.data], 3)
                else:
                    visual = torch.cat([imgc.data[:, :3, :, :], recon.data, img_originalc.data], 3)

                visual = vutils.make_grid(visual, normalize=True)
                writer.add_image('image', visual, step)
                vutils.save_image(visual, '%s/compare%d_%d.png' % (opt.outf, epoch, i), nrow=1)

            if i % 10 == 0:
                writer.add_scalar('MSEloss', loss.data[0], step)
                writer.add_scalar('G_loss', errG_data, step)
                writer.add_scalar('D_loss', errD_data, step)

            if i % 2000 == 0:
                torch.save(comp.state_dict(), '%s/compG_epoch%d_%d.pth' % (opt.outf, epoch, i))

                if opt.unfiller:
                    torch.save(comp2.state_dict(), '%s/compG2_epoch%d_%d.pth' % (opt.outf, epoch, i))


if __name__ == '__main__':
    main()
