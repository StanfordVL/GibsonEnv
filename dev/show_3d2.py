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



mousex,mousey=0.5,0.5
changed=True
pitch,yaw,x,y,z = 0,0,0,0,0
roll = 0
org_pitch, org_yaw, org_x, org_y, org_z = 0,0,0,0,0
org_roll = 0
mousedown = False
clickstart = (0,0)
fps = 0

dll=np.ctypeslib.load_library('render_cuda','.')


def onmouse(*args):
    global mousex,mousey,changed
    global pitch,yaw,x,y,z
    global org_pitch, org_yaw, org_x, org_y, org_z
    global org_roll, roll
    global clickstart

    if args[0] == cv2.EVENT_LBUTTONDOWN:
        org_pitch, org_yaw, org_x, org_y, org_z =\
        pitch,yaw,x,y,z
        clickstart = (mousex, mousey)

    if args[0] == cv2.EVENT_RBUTTONDOWN:
        org_roll = roll
        clickstart = (mousex, mousey)

    if (args[3] & cv2.EVENT_FLAG_LBUTTON):
        pitch = org_pitch + (mousex - clickstart[0])/10
        yaw = org_yaw + (mousey - clickstart[1])
        changed=True

    if (args[3] & cv2.EVENT_FLAG_RBUTTON):
        roll = org_roll + (mousex - clickstart[0])/50
        changed=True

    my=args[1]
    mx=args[2]
    mousex=mx/float(256)
    mousey=my/float(256 * 2)



def showpoints(img, depth, pose, model, target):
    global mousex,mousey,changed
    global pitch,yaw,x,y,z,roll
    global fps

    showsz = target.shape[0]

    show=np.zeros((showsz,showsz * 2,3),dtype='uint8')
    target_depth = np.zeros((showsz,showsz * 2)).astype(np.float32)
    overlay = False
    show_depth = False
    cv2.namedWindow('show3d')
    cv2.moveWindow('show3d',0,0)
    cv2.setMouseCallback('show3d',onmouse)

    imgv = Variable(torch.zeros(1,3, showsz, showsz*2)).cuda()
    maskv = Variable(torch.zeros(1,1, showsz, showsz*2)).cuda()

    cpose = np.eye(4)

    def render(img, depth, pose, model):
        global fps
        t0 = time.time()
        dll.render(ct.c_int(img.shape[0]),
                   ct.c_int(img.shape[1]),
                   img.ctypes.data_as(ct.c_void_p),
                   depth.ctypes.data_as(ct.c_void_p),
                   pose.ctypes.data_as(ct.c_void_p),
                   show.ctypes.data_as(ct.c_void_p),
                   target_depth.ctypes.data_as(ct.c_void_p)
                  )
        if model:
            tf = transforms.ToTensor()
            source = tf(show)
            source_depth = tf(np.expand_dims(target_depth, 2))
            #print(source.size(), source_depth.size())

            imgv.data.copy_(source)
            maskv.data.copy_(source_depth)

            recon = model(imgv, maskv)
            #print(recon.size())
            show2 = recon.data.cpu().numpy()[0].transpose(1,2,0)
            show[:] = (show2[:] * 255).astype(np.uint8)

        t1 = time.time()
        t = t1-t0
        fps = 1/t

        cv2.waitKey(5)%256

    while True:

        if changed:
            alpha = yaw
            beta = pitch
            gamma = roll
            cpose = cpose.flatten()

            cpose[0] = cos(alpha) * cos(beta);
            cpose[1] = cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma);
            cpose[2] = cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma);
            cpose[3] = 0

            cpose[4] = sin(alpha) * cos(beta);
            cpose[5] = sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma);
            cpose[6] = sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma);
            cpose[7] = 0

            cpose[8] = -sin(beta);
            cpose[9] = cos(beta) * sin(gamma);
            cpose[10] = cos(beta) * cos(gamma);
            cpose[11] = 0

            cpose[12:16] = 0
            cpose[15] = 1

            cpose = cpose.reshape((4,4))

            cpose2 = np.eye(4)
            cpose2[0,3] = x
            cpose2[1,3] = y
            cpose2[2,3] = z

            cpose = np.dot(cpose, cpose2)

            print('cpose',cpose)
            render(img, depth, cpose.astype(np.float32), model)
            changed = False

        if overlay:
            show_out = (show/2 + target/2).astype(np.uint8)
        elif show_depth:
            show_out = (target_depth * 10).astype(np.uint8)
        else:
            show_out = show

        cv2.putText(show,'pitch %.3f yaw %.2f roll %.3f x %.2f y %.2f z %.2f'%(pitch, yaw, roll, x, y, z),(15,showsz-15),0,0.5,cv2.cv.CV_RGB(255,255,255))
        cv2.putText(show,'fps %.1f'%(fps),(15,15),0,0.5,cv2.cv.CV_RGB(255,255,255))

        show_rgb = cv2.cvtColor(show_out, cv2.COLOR_BGR2RGB)
        cv2.imshow('show3d',show_rgb)

        cmd=cv2.waitKey(5)%256

        if cmd==ord('q'):
            break
        elif cmd == ord('w'):
            x -= 0.05
            changed = True
        elif cmd == ord('s'):
            x += 0.05
            changed = True
        elif cmd == ord('a'):
            y += 0.05
            changed = True
        elif cmd == ord('d'):
            y -= 0.05
            changed = True

        elif cmd == ord('z'):
            z += 0.01
            changed = True
        elif cmd == ord('x'):
            z -= 0.01
            changed = True

        elif cmd == ord('r'):
            pitch,yaw,x,y,z = 0,0,0,0,0
            roll = 0
            changed = True
        elif cmd == ord('t'):
            print('pose', pose)
            RT = pose.reshape((4,4))

            R = RT[:3,:3]
            T = RT[:3,-1]

            x,y,z = np.dot(np.linalg.inv(R),T)
            roll, pitch, yaw = (utils.rotationMatrixToEulerAngles(R))


            changed = True


        elif cmd == ord('o'):
            overlay = not overlay
        elif cmd == ord('f'):
            show_depth = not show_depth


def show_target(target_img):
    cv2.namedWindow('target')
    cv2.moveWindow('target',0,256 + 50)
    show_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('target', show_rgb)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug'  , action='store_true', help='debug mode')
    parser.add_argument('--dataroot'  , required = True, help='dataset path')
    parser.add_argument('--idx'  , type = int, default = 0, help='index of data')
    parser.add_argument('--model'  , type = str, default = '', help='path of model')

    opt = parser.parse_args()
    d = ViewDataSet3D(root=opt.dataroot, transform = np.array, mist_transform = np.array, seqlen = 2, off_3d = False)
    idx = opt.idx

    data = d[idx]

    source = data[0][0]
    target = data[1]
    source_depth = data[2][0]
    pose = data[-1][0].numpy()
    model = None
    if opt.model != '':
        comp = CompletionNet()
        comp = torch.nn.DataParallel(comp).cuda()
        comp.load_state_dict(torch.load(opt.model))
        model = comp.module
        model.eval()
    print(model)
    print(pose)
    #print(source_depth)
    print(source.shape, source_depth.shape)
    show_target(target)
    showpoints(source, source_depth, pose, model, target)
