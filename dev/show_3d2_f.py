## Blender generated poses and OpenGL default poses adhere to
## different conventions. To better understand the nitty-gritty
## transformations inside this file, check out this:
## https://en.wikibooks.org/wiki/OpenGL_Programming/Modern_OpenGL_Tutorial_Navigation
##  Blender: z-is-up
##      same with: csv, rt_camera_matrix
##  OpenGL: y-is-up
##      same with: obj
##  Default camera: y is up-direction, -z facing


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


dll=np.ctypeslib.load_library('render_cuda_f','.')



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

def mat_to_posi_xyz(cpose):
    return cpose[:3, -1]

def mat_to_quat_xyzw(cpose):
    rot = cpose[:3, :3]
    ## Return: [r_x, r_y, r_z, r_w]
    wxyz = transforms3d.quaternions.mat2quat(rot)
    return quat_wxyz_to_xyzw(wxyz)

## Quat(wxyz)
def quat_pos_to_mat(pos, quat):
    r_w, r_x, r_y, r_z = quat
    #print("quat", r_w, r_x, r_y, r_z)
    mat = np.eye(4)
    mat[:3, :3] = transforms3d.quaternions.quat2mat([r_w, r_x, r_y, r_z])
    mat[:3, -1] = pos
    # Return: roll, pitch, yaw
    return mat

## Used for URDF models that are default -x facing
##  Rotate the model around its internal x axis for 90 degrees
##  so that it is at "normal" pose when applied camera_rt_matrix 
## Format: wxyz for input & return
def z_up_to_y_up(quat_wxyz):
    ## Operations (1) rotate around y for pi/2, 
    ##            (2) rotate around z for pi/2
    to_y_up = transforms3d.euler.euler2quat(0, np.pi/2, np.pi/2)
    return transforms3d.quaternions.qmult(quat_wxyz, to_y_up)

## Models coming out of opengl are negative x facing
##  Transform the default to 
def y_up_to_z_up(quat_wxyz):
    to_z_up = transforms3d.euler.euler2quat(np.pi/2, 0, np.pi/2)
    return transforms3d.quaternions.qmult(to_z_up, quat_wxyz)


def quat_wxyz_to_euler(wxyz):
    q0, q1, q2, q3 = wxyz
    sinr = 2 * (q0 * q1 + q2 * q3)
    cosr = 1 - 2 * (q1 * q1 + q2 * q2)
    sinp = 2 * (q0 * q2 - q3 * q1)
    siny = 2 * (q0 * q3 + q1 * q2)
    cosy = 1 - 2 * (q2 * q2 + q3 * q3)

    roll  = np.arctan2(sinr, cosr)
    pitch = np.arcsin(sinp)
    yaw   = np.arctan2(siny, cosy)
    return [roll, pitch, yaw]


## wxyz: numpy array format
def quat_wxyz_to_xyzw(wxyz):
    return np.concatenate((wxyz[1:], wxyz[:1]))

## xyzw: numpy array format
def quat_xyzw_to_wxyz(xyzw):
    return np.concatenate((xyzw[-1:], xyzw[:-1]))


## Talking to physics simulation
## New state: [x, y, z, r_w, r_x, r_y, r_z]
def hasNoCollision(posi_xyz, quat_wxyz):
    ## Change quaternion basis
    #print("new_state 0", new_state)
    #print("new temp st", [new_state[4], new_state[5], new_state[6], new_state[3]],  transforms3d.euler.euler2quat(-np.pi/2, 0, 0))

    new_state = np.concatenate((posi_xyz, quat_wxyz)) 
    ## URDF file uses z-up as front, so we need to rotate new_state around z for -90
    #print("new_state 1", new_state)
    socket_phys.send(" ".join(map(str, new_state)))
    collisions_count = int(socket_phys.recv().decode("utf-8"))
    print("collisions", collisions_count)
    return collisions_count == 0

def getPoseOrientationFromPhysics():
    receive = socket_phys.recv().decode("utf-8")
    new_pos, new_quat = json.loads(receive)
    print("received from physics", new_pos, new_quat)
    socket_phys.send(json.dumps({"received": True}))
    return new_pos, new_quat


## Return pos(xyz), quat(wxyz)
def getNewPoseFromPhysics(view_pose):
    view_pose['quat'] = quat_wxyz_to_xyzw(view_pose['quat']).tolist()
    #print("sending over new pose", view_pose['pos'])
    #print("sending over new quat", view_pose['quat'])
    socket_phys.send(json.dumps(view_pose))
    new_pos, new_quat = json.loads(socket_phys.recv().decode("utf-8"))
    return new_pos, quat_xyzw_to_wxyz(new_quat)


class PCRenderer:
    def __init__(self):
        self.roll, self.pitch, self.yaw = 0, 0, 0
        self.x, self.y, self.z = 0, 0, 0
        self.fps = 0
        self.mousex, self.mousey = 0.5, 0.5
        self.changed = True
        self.org_pitch, self.org_yaw, self.org_roll = 0, 0, 0
        self.org_x, self.org_y, self.org_z = 0, 0, 0
        self.clickstart = (0,0)
        self.mousedown = False
        self.fps = 0
        self.rotation_const = np.array([[0,1,0,0],[0,0,1,0],[-1,0,0,0],[0,0,0,1]])

    def onmouse(self, *args):
        if args[0] == cv2.EVENT_LBUTTONDOWN:
            self.org_pitch, self.org_yaw, self.org_x, self.org_y, self.org_z =\
                self.pitch,self.yaw,self.x,self.y,self.z
            self.clickstart = (self.mousex, self.mousey)

        if args[0] == cv2.EVENT_RBUTTONDOWN:
            self.org_roll = self.roll
            self.clickstart = (self.mousex, self.mousey)

        if (args[3] & cv2.EVENT_FLAG_LBUTTON):
            self.pitch = self.org_pitch + (self.mousex - self.clickstart[0])/10
            self.yaw = self.org_yaw + (self.mousey - self.clickstart[1])
            self.changed=True

        if (args[3] & cv2.EVENT_FLAG_RBUTTON):
            self.roll = self.org_roll + (self.mousex - self.clickstart[0])/50
            self.changed=True

        my=args[1]
        mx=args[2]
        self.mousex=mx/float(256)
        self.mousey=my/float(256 * 2)

    def getViewerCpose(self):
        cpose = np.eye(4)
        alpha = self.yaw
        beta = self.pitch
        gamma = self.roll
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
        cpose2[0,3] = self.x
        cpose2[1,3] = self.y
        cpose2[2,3] = self.z

        cpose = np.dot(cpose, cpose2)
        return cpose
        #print('new_cpose', cpose)


    def render(self, imgs, depths, pose, model, poses, target_pose, show):
        t0 = time.time()

        v_cam2world = target_pose.dot(poses[0])
        p = (v_cam2world).dot(np.linalg.inv(pose))
        p = p.dot(np.linalg.inv(self.rotation_const))
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
        print(target_depth.shape, opengl_arr.shape)
        target_depth[:, :, 0] = (opengl_arr[:,:,0] * 100).astype(np.int32)


        show[:] = 0
        before = time.time()
        for i in range(len(imgs)):
            pose_after = pose.dot(np.linalg.inv(poses[0])).dot(poses[i]).astype(np.float32)

            dll.render(ct.c_int(imgs[i].shape[0]),
                       ct.c_int(imgs[i].shape[1]),
                       imgs[i].ctypes.data_as(ct.c_void_p),
                       depths[i].ctypes.data_as(ct.c_void_p),
                       pose_after.ctypes.data_as(ct.c_void_p),
                       show.ctypes.data_as(ct.c_void_p),
                       target_depth.ctypes.data_as(ct.c_void_p)
                      )

            
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
        self.fps = 1/t

        #cv2.waitKey(5)%256


    def instantCheckPos(self, pos, target_pose):
        cpose = self.getViewerCpose()
        v_cam2world = target_pose.dot(poses[0])
        world_cpose = (v_cam2world).dot(np.linalg.inv(cpose)).dot(np.linalg.inv(self.rotation_const))
        new_pos  = mat_to_posi_xyz(world_cpose).tolist()
        #print("Instant chec xyz", x, y, z)
        print("Instant check pose", new_pos)

    def showpoints(self, imgs, depths, poses, model, target, tdepth, target_pose):
        showsz = target.shape[0]
        #print('target pose', target_pose)

        v_cam2world = target_pose.dot(poses[0])
        cpose = self.getViewerCpose()
        p = (v_cam2world).dot(np.linalg.inv(cpose))
        p = p.dot(np.linalg.inv(self.rotation_const))
        pos  = mat_to_posi_xyz(p).tolist()
        quat = mat_to_quat_xyzw(p).tolist()
        receive = str(socket_phys.recv().decode("utf-8"))
        if (receive == "Initial"):
            socket_phys.send(json.dumps([pos, quat]))
        else:
            return

        show=np.zeros((showsz,showsz * 2,3),dtype='uint8')
        target_depth = np.zeros((showsz,showsz * 2)).astype(np.int32)

        overlay = False
        show_depth = False
        cv2.namedWindow('show3d')
        cv2.namedWindow('target depth')

        cv2.moveWindow('show3d',0,0)
        cv2.setMouseCallback('show3d',self.onmouse)

        imgv = Variable(torch.zeros(1,3, showsz, showsz*2), volatile=True).cuda()
        maskv = Variable(torch.zeros(1,1, showsz, showsz*2), volatile=True).cuda()

        old_state = [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]
        old_cpose = np.eye(4)

        while True:
            cpose = self.getViewerCpose()
            v_cam2world = target_pose.dot(poses[0])
            world_cpose = (v_cam2world).dot(np.linalg.inv(cpose)).dot(np.linalg.inv(self.rotation_const))
            #print("world pose", world_cpose)


            ## Query physics engine to get [x, y, z, roll, pitch, yaw]
            if PHYSICS_FIRST:
                pos  = mat_to_posi_xyz(world_cpose).tolist()
                quat_wxyz = quat_xyzw_to_wxyz(mat_to_quat_xyzw(world_cpose)).tolist()
                
                '''
                new_pos, new_quat_wxyz = getNewPoseFromPhysics({
                        'changed': self.changed,
                        'pos': pos, 
                        'quat': quat_wxyz
                    })
                '''

                #new_quat = y_up_to_z_up(new_quat_wxyz)
                #new_quat = new_quat_wxyz
                print("waiting for physics")
                new_pos, new_euler = getPoseOrientationFromPhysics()

                '''
                delta = 2
            
                cpose = np.linalg.inv(np.linalg.inv(v_cam2world).dot(world_cpose).dot(self.rotation_const))
                viewer_pose  = mat_to_posi_xyz(cpose).tolist()
                viewer_quat = quat_xyzw_to_wxyz(mat_to_quat_xyzw(cpose)).tolist()
                '''
                ## TODO: this part is hardcoded
                self.x = -new_pos[1]
                self.y = -new_pos[0]
                self.z = -new_pos[2]
                
                # rotation around intrinsic x, y, z
                alpha, beta, gamma = new_euler
                print("alpha, beta, gamma", alpha, beta, gamma)

                self.roll, self.pitch, self.yaw = -gamma, alpha, -beta
                #self.instantCheckPos(viewer_pose, target_pose)
                

                self.changed = True

            if self.changed:
                new_quat = mat_to_quat_xyzw(world_cpose)
                new_quat = z_up_to_y_up(quat_xyzw_to_wxyz(new_quat))

                new_posi = mat_to_posi_xyz(world_cpose)

                ## Entry point for change of view 
                ## If PHYSICS_FIRST mode, then collision is already handled
                ##   inside physics simulator
                if PHYSICS_FIRST or hasNoCollision(new_posi, new_quat):
                    print("no collisions")
                    self.render(imgs, depths, cpose.astype(np.float32), model, poses, target_pose, show)
                    old_state = [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]
                    old_cpose = np.copy(cpose)
                else:
                    print("has collisions")
                    self.x, self.y, self.z, self.roll, self.pitch, self.yaw = old_state
                    cpose = old_cpose
                
                #render(imgs, depths, cpose.astype(np.float32), model, poses)
                #old_state = [x, y, z, roll, pitch, yaw]
                self.changed = False

            if overlay:
                show_out = (show/2 + target/2).astype(np.uint8)
            elif show_depth:
                show_out = (target_depth * 10).astype(np.uint8)
            else:
                show_out = show

            #assert(np.sum(show) != 0)


            cv2.putText(show,'pitch %.3f yaw %.2f roll %.3f x %.2f y %.2f z %.2f'%(self.pitch, self.yaw, self.roll, self.x, self.y, self.z),(15,showsz-15),0,0.5,(255,255,255))
            #print("roll %f pitch %f yaw %f" % (roll, pitch, yaw))
            
            #cv2.putText(show,'fps %.1f'%(fps),(15,15),0,0.5,cv2.cv.CV_RGB(255,255,255))
            cv2.putText(show,'fps %.1f'%(self.fps),(15,15),0,0.5,(255,255,255))

            show_rgb = cv2.cvtColor(show_out, cv2.COLOR_BGR2RGB)
            cv2.imshow('show3d',show_rgb)

            cmd=cv2.waitKey(5)%256

            ## delta = [x, y, z, roll, pitch, yaw]
            #delta = [0, 0, 0, 0, 0, 0]

            if cmd==ord('q'):
                break
            elif cmd == ord('w'):
                self.x -= 0.05
                #delta = [-0.05, 0, 0, 0, 0, 0]
                self.changed = True
            elif cmd == ord('s'):
                self.x += 0.05
                #delta = [0.05, 0, 0, 0, 0, 0]
                self.changed = True
            elif cmd == ord('a'):
                self.y += 0.05
                #delta = [0, 0.05, 0, 0, 0, 0]
                self.changed = True
            elif cmd == ord('d'):
                self.y -= 0.05
                #delta = [0, -0.05, 0, 0, 0, 0]
                self.changed = True
            elif cmd == ord('z'):
                self.z += 0.01
                #delta = [0, 0, 0, 0, 0, 0.01]
                self.changed = True
            elif cmd == ord('x'):
                self.z -= 0.01
                #delta = [0, 0, 0, 0, 0, -0.01]
                self.changed = True

            elif cmd == ord('r'):
                self.pitch,self.yaw,self.x,self.y,self.z = 0,0,0,0,0
                self.roll = 0
                self.changed = True
            elif cmd == ord('t'):
                pose = poses[0]
                print('pose', pose)
                RT = pose.reshape((4,4))
                R = RT[:3,:3]
                T = RT[:3,-1]

                self.x,self.y,self.z = np.dot(np.linalg.inv(R),T)
                self.roll, self.pitch, self.yaw = (utils.rotationMatrixToEulerAngles(R))

                self.changed = True


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
    socket_phys = context_phys.socket(zmq.REP)
    socket_phys.connect("tcp://localhost:5556")

        
    uuids, rts = d.get_scene_info(0)
    #print(uuids, rts)
    print(uuids[idx])

    show_target(target)

    renderer = PCRenderer()
    renderer.showpoints(sources, source_depths, poses, model, target, target_depth, rts[idx])
