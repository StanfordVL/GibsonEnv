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
#import matplotlib
#matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
from numpy import cos, sin

import utils


showsz = 256
mousex,mousey=0.5,0.5
changed=True
pitch,yaw,x,y,z = 0,0,0,0,0
roll = 0
org_pitch, org_yaw, org_x, org_y, org_z = 0,0,0,0,0
org_roll = 0
mousedown = False
clickstart = (0,0)
fps = 0

dll=np.ctypeslib.load_library('render','.')


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
    mousex=mx/float(showsz)
    mousey=my/float(showsz * 2)



def showpoints(img, depth, pose, model, xyzs, rts):
    global mousex,mousey,changed
    global pitch,yaw,x,y,z,roll
    global fps
    show=np.zeros((showsz,showsz * 2,3),dtype='uint8')
    minimap=np.zeros((showsz,showsz,3),dtype='uint8')
    target_depth = np.zeros((showsz,showsz * 2)).astype(np.float32)
    overlay = False
    show_depth = False
    cv2.namedWindow('show3d')
    cv2.moveWindow('show3d',0,0)
    cv2.setMouseCallback('show3d',onmouse)
    cv2.namedWindow('minimap')
    cv2.moveWindow('minimap',showsz*3,0)
    
    xs = [item[0] for item in  xyzs.values()]
    ys = [item[1] for item in  xyzs.values()]


    imgv = Variable(torch.zeros(1,3, 256, 512)).cuda()
    maskv = Variable(torch.zeros(1,1, 256, 512)).cuda()


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
      
        minimap[:] = 0
        xs = [item[0] for item in  xyzs.values()]
        ys = [item[1] for item in  xyzs.values()]
        
        maxx = np.max(xs)
        minx = np.min(xs)
        maxy = np.max(ys)
        miny = np.min(ys)
        
        for i in range(len(xs)):
            cv2.circle(minimap,(int((xs[i] - minx) * showsz / (maxx - minx)),int((ys[i] - miny) * showsz / (maxy - miny))), 5, (0,0,255), -1)

        cv2.circle(minimap,(int((x - minx) * showsz / (maxx - minx)),int((y - miny) * showsz / (maxy - miny))), 5, (0,255,255), -1)
        cv2.waitKey(5)%256
        

    while True:

        if changed:            
            
            current_t = np.eye(4)
            current_t[0,-1] = x
            current_t[1,-1] = y
            current_t[2,-1] = z
            
            alpha = yaw
            beta = pitch
            gamma = roll
            
            current_r = np.array([[cos(alpha) * cos(beta), cos(alpha) * sin(beta) * sin(gamma) - sin(alpha)* cos(gamma), cos(alpha) * sin(beta) * cos(gamma) + sin(alpha)*sin(gamma), 0],
                                 [sin(alpha)*cos(beta), sin(alpha)*sin(beta)*sin(gamma)+cos(alpha)*cos(gamma), sin(alpha)*sin(gamma)*cos(gamma) - cos(alpha)*sin(gamma), 0],
                                 [-sin(beta), cos(beta)*sin(gamma), cos(beta)*cos(gamma),0],
                                 [0,0,0,1]])
            
            
            current_rt = np.dot(current_t, current_r)
            dist = []
            for i in range(len(rts)):
                
                rt = rts[i]
                dist.append( np.sum(np.dot(current_rt, np.linalg.inv(rt))[0:3, -1] **2))
            
            
            #print(np.dot(current_rt, np.linalg.inv(rt))[0:3, -1])
            
            idx = np.argmin(dist)
            
            img = sources[idx]
            depth = source_depths[idx]
            
            
            rt = rts[idx]
            relative = np.dot(current_rt, np.linalg.inv(rt))
            
            #relative = np.linalg.inv(relative)
            print(relative)
            render(img, depth, relative.astype(np.float32), model)
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
        cv2.imshow('minimap',minimap)
        


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

    model = None
    if opt.model != '':
        comp = CompletionNet()
        comp = torch.nn.DataParallel(comp).cuda()
        comp.load_state_dict(torch.load(opt.model))
        model = comp.module
        model.eval()
    print(model)
    
    
    idx = opt.idx
    uuids, xyzs, rts = d.get_scene_info(idx)
    
    sources = []
    source_depths = []
    poses = []
    
    for k,v in uuids.items():
        print(v)
    
        source = d[v][0][0]
        source_depth = d[v][2][0]
        pose = d[v][-1][0].numpy()
        
        sources.append(source)
        source_depths.append(source_depth)
        poses.append(pose)
    
    showpoints(sources, source_depths, poses, model, xyzs, rts)
    
