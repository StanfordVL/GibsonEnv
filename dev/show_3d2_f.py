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
import transforms3d
import json

import zmq
from cube2equi import find_corresponding_pixel
from transfer import transfer2


PHYSICS_FIRST = True

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
    outimg = np.zeros((h,w,1)) #.astype(np.uint8)

    in_imgs = None
    #print("converting images", len(img_array))

    #print("Passed in image array", len(img_array), np.max(img_array[0]))
    in_imgs = img_array

    # For each pixel in output image find colour value from input image
    #print(outimg.shape)

    # todo: for some reason the image is flipped 180 degrees
    outimg = transfer2(in_imgs, coords, h, w)[:, ::, :]

    return outimg

def mat_to_posi(cpose):
    return cpose[:3, -1]

def mat_to_quat(cpose):
    rot = cpose[:3, :3]
    ## Return: [r_x, r_y, r_z, r_w]
    wxyz = transforms3d.quaternions.mat2quat(rot)
    return np.concatenate((wxyz[1:], wxyz[:1]))

def quat_pos_to_mat(pos, quat):
    r_x, r_y, r_z, r_w = quat
    print("quat", r_w, r_x, r_y, r_z)
    mat = np.eye(4)
    mat[:3, :3] = transforms3d.quaternions.quat2mat([r_w, r_x, r_y, r_z])
    mat[:3, -1] = pos
    # Return: roll, pitch, yaw
    return mat

## Talking to physics simulation
## New state: [x, y, z, r_w, r_x, r_y, r_z]
def hasNoCollision(new_state):
    ## Change quaternion basis
    #print("new_state 0", new_state)
    #print("new temp st", [new_state[4], new_state[5], new_state[6], new_state[3]],  transforms3d.euler.euler2quat(-np.pi/2, 0, 0))
    new_state[3:] =  transforms3d.quaternions.qmult(transforms3d.euler.euler2quat(0, 0, -np.pi/2), [new_state[4], new_state[5], new_state[6], new_state[3]])
    ## URDF file uses z-up as front, so we need to rotate new_state around z for -90
    #print("new_state 1", new_state)
    socket_phys.send(" ".join(map(str, new_state)))
    collisions_count = int(socket_phys.recv().decode("utf-8"))
    print("collisions", collisions_count)
    return collisions_count == 0

def getNewPoseFromPhysics(view_pose):
    socket_phys.send(json.dumps(view_pose))
    new_pos, new_quat = json.loads(socket_phys.recv().decode("utf-8"))
    return new_pos, new_quat


def showpoints(imgs, depths, poses, model, target, tdepth, target_pose):
    global mousex,mousey,changed
    global pitch,yaw,x,y,z,roll
    global fps

    showsz = target.shape[0]
    rotation = np.array([[0,1,0,0],[0,0,1,0],[-1,0,0,0],[0,0,0,1]])
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
    old_state = [x, y, z, roll, pitch, yaw]

    def getViewerCpose(cpose):
        alpha = yaw
        beta = pitch
        gamma = roll
        old_cpose = np.copy(cpose)
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
        return cpose
        #print('new_cpose', cpose)

    def render(imgs, depths, pose, model, poses):
        global fps
        t0 = time.time()

        v_cam2world = target_pose.dot(poses[0])
        p = (v_cam2world).dot(np.linalg.inv(pose))
        p = p.dot(np.linalg.inv(rotation))

        #print("Sending request ...")
        #print(v_cam2world)
        #print('current viewer pose', pose)
        #print("camera pose", p)
        #print("target pose", target_pose)
        s = mat_to_str(p)


        socket_mist.send(s)
        message = socket_mist.recv()
        #print("Received messages")
        data = np.array(np.frombuffer(message, dtype=np.float32)).reshape((6, 768, 768, 1))
        ## For some reason, the img passed back from opengl is upside down.
        ## This is still yet to be debugged
        data = data[:, ::-1,::,:]
        img_array = []
        for i in range(6):
            img_array.append(data[i])

        img_array2 = [img_array[0], img_array[1], img_array[2], img_array[3], img_array[4], img_array[5]]
        #print("max value", np.max(data[0]), "shape", np.array(img_array2).shape)

        opengl_arr = convert_array(np.array(img_array2))
        opengl_arr = opengl_arr[::, ::]

        
        #print("opengl array shape", opengl_arr.shape)
        #plot_histogram(opengl_arr)
        #print("zero values", np.sum(opengl_arr[:, :, 0] == 0), np.sum(opengl_arr[:, :, 1] == 0), np.sum(opengl_arr[:, :, 2] == 0))
        
        #print("opengl min", np.min(opengl_arr), "opengl max", np.max(opengl_arr))
        opengl_arr_err  = opengl_arr == 0

        #opengl_arr = np.maximum(opengl_arr + 30, opengl_arr)

        opengl_arr_show = (opengl_arr * 3500.0 / 128).astype(np.uint8)

        #print('arr shape', opengl_arr_show.shape, "max", np.max(opengl_arr_show), "total number of errors", np.sum(opengl_arr_err))
       
        opengl_arr_show[opengl_arr_err[:, :, 0], 1:3] = 0
        opengl_arr_show[opengl_arr_err[:, :, 0], 0] = 255
        cv2.imshow('target depth',opengl_arr_show)

        #from IPython import embed; embed()
        target_depth[:] = (opengl_arr[:,:,0] * 100).astype(np.int32)


        show[:] = 0
        before = time.time()
        for i in range(len(imgs)):
            #print(poses[0])

            pose_after = pose.dot(np.linalg.inv(poses[0])).dot(poses[i]).astype(np.float32)
            #if i == 0:
                #print('First pose after')
                #print(pose_after)
            #from IPython import embed; embed()
            #print('Received pose ' + str(i))
            #print(pose_after)

            dll.render(ct.c_int(imgs[i].shape[0]),
                       ct.c_int(imgs[i].shape[1]),
                       imgs[i].ctypes.data_as(ct.c_void_p),
                       depths[i].ctypes.data_as(ct.c_void_p),
                       pose_after.ctypes.data_as(ct.c_void_p),
                       show.ctypes.data_as(ct.c_void_p),
                       target_depth.ctypes.data_as(ct.c_void_p)
                      )

            #if i == 0:
            #    print(np.sum(show - imgs[0]))
            

        #print('PC render time:', time.time() - before)

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

        #cv2.waitKey(5)%256

    while True:
        cpose = getViewerCpose(cpose)
        v_cam2world = target_pose.dot(poses[0])
        world_cpose = (v_cam2world).dot(np.linalg.inv(cpose)).dot(np.linalg.inv(rotation))
        #print("world pose", world_cpose)


        ## Query physics engine to get [x, y, z, roll, pitch, yaw]
        if PHYSICS_FIRST:
            pos  = mat_to_posi(world_cpose).tolist()
            quat = mat_to_quat(world_cpose).tolist()
            new_pos, new_quat = getNewPoseFromPhysics({
                    'changed': changed,
                    'pos': pos, 
                    'quat': quat
                })
            world_cpose = quat_pos_to_mat(new_pos, new_quat)
            try:
                cpose = np.linalg.inv(np.linalg.inv(v_cam2world).dot(world_cpose).dot(rotation))
            except:
                continue
            changed = True

        if changed:
            #world_cpose = cpose.dot(v_cam2world)

            #old_quat = mat_to_quat(old_cpose)
            #old_posi = mat_to_posi(old_cpose)
            new_quat = mat_to_quat(world_cpose)
            new_posi = mat_to_posi(world_cpose)

            ## Entry point for change of view 
            ## If PHYSICS_FIRST mode, then collision is already handled
            ##   inside physics simulator
            if PHYSICS_FIRST or hasNoCollision(np.append(new_posi, new_quat)):
                print("no collisions")
                render(imgs, depths, cpose.astype(np.float32), model, poses)
                old_state = [x, y, z, roll, pitch, yaw]
            else:
                print("has collisions")
                x, y, z, roll, pitch, yaw = old_state
                cpose = old_cpose
            
            #render(imgs, depths, cpose.astype(np.float32), model, poses)
            #old_state = [x, y, z, roll, pitch, yaw]
            changed = False

        if overlay:
            show_out = (show/2 + target/2).astype(np.uint8)
        elif show_depth:
            show_out = (target_depth * 10).astype(np.uint8)
        else:
            show_out = show

        
        cv2.putText(show,'pitch %.3f yaw %.2f roll %.3f x %.2f y %.2f z %.2f'%(pitch, yaw, roll, x, y, z),(15,showsz-15),0,0.5,(255,255,255))
        #print("roll %f pitch %f yaw %f" % (roll, pitch, yaw))
        
        #cv2.putText(show,'fps %.1f'%(fps),(15,15),0,0.5,cv2.cv.CV_RGB(255,255,255))
        cv2.putText(show,'fps %.1f'%(fps),(15,15),0,0.5,(255,255,255))

        show_rgb = cv2.cvtColor(show_out, cv2.COLOR_BGR2RGB)
        cv2.imshow('show3d',show_rgb)

        cmd=cv2.waitKey(1)%256

        ## delta = [x, y, z, roll, pitch, yaw]
        #delta = [0, 0, 0, 0, 0, 0]

        if cmd==ord('q'):
            break
        elif cmd == ord('w'):
            x -= 0.05
            #delta = [-0.05, 0, 0, 0, 0, 0]
            changed = True
        elif cmd == ord('s'):
            x += 0.05
            #delta = [0.05, 0, 0, 0, 0, 0]
            changed = True
        elif cmd == ord('a'):
            y += 0.05
            #delta = [0, 0.05, 0, 0, 0, 0]
            changed = True
        elif cmd == ord('d'):
            y -= 0.05
            #delta = [0, -0.05, 0, 0, 0, 0]
            changed = True
        elif cmd == ord('z'):
            z += 0.01
            #delta = [0, 0, 0, 0, 0, 0.01]
            changed = True
        elif cmd == ord('x'):
            z -= 0.01
            #delta = [0, 0, 0, 0, 0, -0.01]
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

        '''
        state = [x, y, z, roll, pitch, yaw]
        if (hasNoCollision(state, delta)):
            x = x + delta[0]
            y = y + delta[1]
            z = z + delta[2]
            roll  = roll + delta[3]
            pitch = pitch + delta[4]
            yaw   = yaw + delta[5]
        '''
        #print("roll %d pitch %d yaw %d" % (roll, pitch, yaw))

def show_target(target_img):
    cv2.namedWindow('target')
    cv2.moveWindow('target',0,256 + 50)
    show_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('target', show_rgb)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug'  , action='store_true', help='debug mode')
    parser.add_argument('--datapath'  , required = True, help='dataset path')
    parser.add_argument('--idx'  , type = int, default = 0, help='index of data')
    parser.add_argument('--model'  , type = str, default = '', help='path of model')

    opt = parser.parse_args()
    d = ViewDataSet3D(root=opt.datapath, transform = np.array, mist_transform = np.array, seqlen = 5, off_3d = False, train = False)
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
    print(model)
    print('target', poses, poses[0])
    #print('no.1 pose', poses, poses[1])
    # print(source_depth)
    print(sources[0].shape, source_depths[0].shape)
    
    context_mist = zmq.Context()
    print("Connecting to hello world server...")
    socket_mist = context_mist.socket(zmq.REQ)
    socket_mist.connect("tcp://localhost:5555")

    context_phys = zmq.Context()
    socket_phys = context_phys.socket(zmq.REQ)
    socket_phys.connect("tcp://localhost:5556")

        
    uuids, rts = d.get_scene_info(0)
    #print(uuids, rts)
    print(uuids[idx])

    show_target(target)

    showpoints(sources, source_depths, poses, model, target, target_depth, rts[idx])
