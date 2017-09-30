from __future__ import print_function

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
from profiler import Profiler

import zmq
from cube2equi import find_corresponding_pixel
from transfer import transfer2


from multiprocessing.dummy import Process

class InImg(object):
    def __init__(self):
        self.grid = 768

    def getpixel(self, key):
        corrx, corry = key[0], key[1]

        indx = int(corrx / self.grid)
        indy = int(corry / self.grid)

        remx = int(corrx % self.grid)
        remy = int(corry % self.grid)

        if (indy == 0):
            return (0, remx, remy)
        elif (indy == 2):
            return (5, remx, remy)
        else:
            return (indx + 1, remx, remy)


mousex,mousey=0.5,0.5
changed=True
pitch,yaw,x,y,z = 0,0,0,0,0
roll = 0
org_pitch, org_yaw, org_x, org_y, org_z = 0,0,0,0,0
org_roll = 0
mousedown = False
clickstart = (0,0)
fps = 0

dll=np.ctypeslib.load_library('render_cuda_f','.')


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

    
def mat_to_str(matrix):
    s = ""
    for row in range(4):
        for col in range(4): 
            s = s + " " + str(matrix[row][col])
    return s.strip()

coords = np.load('coord.npy')

def convert_array(img_array):
    inimg = InImg()

    wo, ho = inimg.grid * 4, inimg.grid * 3

    # Calculate height and width of output image, and size of each square face
    h = wo/3
    w = 2*h
    n = ho/3

    # Create new image with width w, and height h
    # outimg = np.zeros((h,w,1)) #.astype(np.uint8)

    in_imgs = None
    print("converting images", len(img_array))

    # print("Passed in image array", len(img_array), np.max(img_array[0]))
    in_imgs = img_array

    # For each pixel in output image find colour value from input image
    # print(outimg.shape)

    # todo: for some reason the image is flipped 180 degrees
    outimg = transfer2(in_imgs, coords, h, w)[:, ::-1, :]

    return outimg



def showpoints(imgs, depths, poses, model, target, tdepth, target_pose):
    global mousex,mousey,changed
    global pitch,yaw,x,y,z,roll
    global fps

    showsz = target.shape[0]
    rotation = np.array([[0,-1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
    target_pose2 = rotation.dot(target_pose)
    print('target pose', target_pose)
    
    
    show=np.zeros((showsz,showsz * 2,3),dtype='uint8')
    target_depth = np.zeros((showsz,showsz * 2)).astype(np.int32)

    #target_depth[:] = (tdepth[:,:,0] * 12800).astype(np.int32)
    #from IPython import embed; embed()
    overlay = False
    show_depth = False
    cv2.namedWindow('show3d')
    cv2.namedWindow('target depth')
    
    cv2.moveWindow('show3d',0,0)
    cv2.setMouseCallback('show3d',onmouse)

    imgv = Variable(torch.zeros(1,3, showsz, showsz*2), volatile=True).cuda()
    maskv = Variable(torch.zeros(1,1, showsz, showsz*2), volatile=True).cuda()

    cpose = np.eye(4)

    def render(imgs, depths, pose, model, poses):
        global fps
        t0 = time.time()
        #target_depth[:] = 65535
        #get target depth
        
        p = pose.dot(np.linalg.inv(poses[0])).dot(target_pose)
        
        p = rotation.dot(p)
        
        print("\n\nSending request ..." , type(p))
        trans = -np.dot(p[:3, :3].T, p[:3, -1])
        rot = np.dot(np.array([[-1,0,0],[0,-1,0],[0,0,1]]),  np.linalg.inv(p[:3, :3]))
        p2 = np.eye(4)
        p2[:3, :3] = rot
        p2[:3, -1] = trans
        s = mat_to_str(p2)        

        with Profiler("Depth request round-trip"):        
            socket.send(s)
            message = socket.recv()

        with Profiler("Read from framebuffer and make pano"):  
            data = np.array(np.frombuffer(message, dtype=np.uint16)).reshape((6, 768, 768, 1))
            data = np.copy(data[:, ::-1,::-1,:])
            opengl_arr = convert_array(data)

        def render_depth(opengl_arr):
            with Profiler("Render Depth"):  
                opengl_depth = np.copy(opengl_arr[...,0])
                opengl_depth[opengl_depth>2**11] = 2**11
                opengl_depth *= 32
                cv2.imshow('target depth', opengl_depth)

        def render_pc(opengl_arr):
            with Profiler("Render pointcloud"):
                scale = 65536 / 12800  # 512
                target_depth = np.int32(opengl_arr[..., 0] / scale)
                show[:] = 0
                poses_after = [
                    pose.dot(np.linalg.inv(poses[0])).dot(poses[i]).astype(np.float32)
                    for i in range(len(imgs))]

                for i in range(len(imgs)):
                    dll.render(ct.c_int(imgs[i].shape[0]),
                            ct.c_int(imgs[i].shape[1]),
                            imgs[i].ctypes.data_as(ct.c_void_p),
                            depths[i].ctypes.data_as(ct.c_void_p),
                            poses_after[i].ctypes.data_as(ct.c_void_p),
                            show.ctypes.data_as(ct.c_void_p),
                            target_depth.ctypes.data_as(ct.c_void_p)
                            )
        
        threads = [
            Process(target=render_pc, args=(opengl_arr,)),
            Process(target=render_depth, args=(opengl_arr,))]
        [t.start() for t in threads]
        [t.join() for t in threads]
        # render_pc(opengl_arr)
        # render_pc(opengl_arr)

        if model:
            tf = transforms.ToTensor()
            before = time.time()
            source = tf(show)
            source_depth = tf(np.expand_dims(target_depth, 2).astype(np.float32)/65536 * 255)
            #print(source.size(), source_depth.size())
            imgv.data.copy_(source)
            maskv.data.copy_(source_depth)
            print('Transfer time', time.time() - before)
            before = time.time()
            recon = model(imgv, maskv)
            print('NNtime:', time.time() - before)
            before = time.time()
            show2 = recon.data.cpu().numpy()[0].transpose(1,2,0)
            show[:] = (show2[:] * 255).astype(np.uint8)
            print('Transfer to CPU time:', time.time() - before)

        t1 =time.time()
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

            # print('cpose',cpose)
            with Profiler("Full render"):
                render(imgs, depths, cpose.astype(np.float32), model, poses)
                changed = False

        if overlay:
            show_out = (show/2 + target/2).astype(np.uint8)
        elif show_depth:
            show_out = (target_depth * 10).astype(np.uint8)
        else:
            show_out = show

        #cv2.putText(show,'pitch %.3f yaw %.2f roll %.3f x %.2f y %.2f z %.2f'%(pitch, yaw, roll, x, y, z),(15,showsz-15),0,0.5,cv2.CV_RGB(255,255,255))
        cv2.putText(show,'pitch %.3f yaw %.2f roll %.3f x %.2f y %.2f z %.2f'%(pitch, yaw, roll, x, y, z),(15,showsz-15),0,0.5,(255,255,255))
        #cv2.putText(show,'fps %.1f'%(fps),(15,15),0,0.5,cv2.cv.CV_RGB(255,255,255))
        cv2.putText(show,'fps %.1f'%(fps),(15,15),0,0.5,(255,255,255))

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
            pose = poses[0]
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
        elif cmd == ord('v'):
            cv2.imwrite('save.jpg', show_rgb)


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
    d = ViewDataSet3D(root=opt.dataroot, transform = np.array, mist_transform = np.array, seqlen = 5, off_3d = False, train = False)
    idx = opt.idx

    data = d[idx]

    sources = data[0]
    target = data[1]
    source_depths = data[2]
    target_depth = data[3]
    poses = [item.numpy() for item in data[-1]]
    print('target', np.max(target_depth[:]))

    model = None
    if opt.model != '':
        comp = CompletionNet()
        comp = torch.nn.DataParallel(comp).cuda()
        comp.load_state_dict(torch.load(opt.model))
        model = comp.module
        model.eval()
    # print(model)
    # print(poses[0])
    # print(source_depth)
    # print(sources[0].shape, source_depths[0].shape)
    
    context = zmq.Context()
    print("Connecting to hello world server...")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    context2 = zmq.Context()
    socket2 = context2.socket(zmq.REQ)
    socket2.connect("tcp://localhost:5556")
    
    uuids, rts = d.get_scene_info(0)
    print(uuids[idx])
    
    show_target(target)
    showpoints(sources, source_depths, poses, model, target, target_depth, rts[idx])
