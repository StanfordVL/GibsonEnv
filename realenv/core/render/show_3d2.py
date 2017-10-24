from __future__ import print_function
import numpy as np
import ctypes as ct
import os
import cv2
import sys
import torch
import argparse
import time
import utils
import transforms3d
import json
import zmq

from torchvision import datasets, transforms
from torch.autograd import Variable
from numpy import cos, sin
from realenv.core.render.profiler import Profiler
from multiprocessing.dummy import Process

from realenv.data.datasets import ViewDataSet3D
from realenv.core.render.completion import CompletionNet


file_dir = os.path.dirname(os.path.abspath(__file__))
cuda_pc = np.ctypeslib.load_library(os.path.join(file_dir, 'render_cuda_f'),'.')
coords  = np.load(os.path.join(file_dir, 'coord.npy'))
context_mist = zmq.Context()
socket_mist = context_mist.socket(zmq.REQ)
socket_mist.connect("tcp://localhost:5555")


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

class PCRenderer:
    ROTATION_CONST = np.array([[0,1,0,0],[0,0,1,0],[-1,0,0,0],[0,0,0,1]])
    def __init__(self, port, imgs, depths, target, target_poses, scale_up):
        self.roll, self.pitch, self.yaw = 0, 0, 0
        self.quat = [1, 0, 0, 0]
        self.x, self.y, self.z = 0, 0, 0
        self.fps = 0
        self.mousex, self.mousey = 0.5, 0.5
        self.org_pitch, self.org_yaw, self.org_roll = 0, 0, 0
        self.org_x, self.org_y, self.org_z = 0, 0, 0
        self.clickstart = (0,0)
        self.mousedown  = False
        self.fps = 0
        self.overlay    = False
        self.show_depth = False
        self._context_phys = zmq.Context()
        self.socket_phys = self._context_phys.socket(zmq.REP)
        self.socket_phys.connect("tcp://localhost:%d" % port)
        self.target_poses = target_poses
        self.imgs = imgs
        self.depths = depths
        self.target = target
        self.model = None
        self.old_topk = set([])
        self.k = 5

        self.showsz = 1024
        self.show   = np.zeros((self.showsz,self.showsz * 2,3),dtype='uint8')
        self.show_rgb   = np.zeros((self.showsz,self.showsz * 2,3),dtype='uint8')

        self.scale_up = scale_up


        self.show   = np.zeros((768, 768, 3),dtype='uint8')
        self.show_rgb   = np.zeros((768, 768 ,3),dtype='uint8')



    def _onmouse(self, *args):
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

        if (args[3] & cv2.EVENT_FLAG_RBUTTON):
            self.roll = self.org_roll + (self.mousex - self.clickstart[0])/50

        my=args[1]
        mx=args[2]
        self.mousex=mx/float(256)
        self.mousey=my/float(256 * 2)

    def _updateStateFromKeyboard(self):
        cmd=cv2.waitKey(5)%256
        if cmd==ord('q'):
            return False
        elif cmd == ord('w'):
            self.x -= 0.05
        elif cmd == ord('s'):
            self.x += 0.05
        elif cmd == ord('a'):
            self.y += 0.05
        elif cmd == ord('d'):
            self.y -= 0.05
        elif cmd == ord('z'):
            self.z += 0.01
        elif cmd == ord('x'):
            self.z -= 0.01
        elif cmd == ord('r'):
            self.pitch,self.yaw,self.x,self.y,self.z = 0,0,0,0,0
            self.roll = 0
        elif cmd == ord('t'):
            pose = poses[0]
            RT = pose.reshape((4,4))
            R = RT[:3,:3]
            T = RT[:3,-1]
            self.x,self.y,self.z = np.dot(np.linalg.inv(R),T)
            self.roll, self.pitch, self.yaw = (utils.rotationMatrixToEulerAngles(R))
        elif cmd == ord('o'):
            self.overlay = not self.overlay
        elif cmd == ord('f'):
            self.show_depth = not self.show_depth
        elif cmd == ord('v'):
            cv2.imwrite('save.jpg', show_rgb)
        return True

    def _getPoseOrientationFromPhysics(self):
        receive = self.socket_phys.recv().decode("utf-8")
        new_pos, new_quat = json.loads(receive)
        self.socket_phys.send(json.dumps({"received": True}))
        return new_pos, new_quat

    def _getNewPoseFromPhysics(self, view_pose):
        '''Return pos(xyz), quat(wxyz)
        '''
        view_pose['quat'] = utils.quat_wxyz_to_xyzw(view_pose['quat']).tolist()
        self.socket_phys.send(json.dumps(view_pose))
        new_pos, new_quat = json.loads(self.socket_phys.recv().decode("utf-8"))
        return new_pos, utils.quat_xyzw_to_wxyz(new_quat)

    def _sendInitialPoseToPhysics(self, pose):
        pose[1] = utils.quat_wxyz_to_xyzw(pose[1]).tolist()
        receive = str(self.socket_phys.recv().decode("utf-8"))
        if (receive == "Initial"):
            self.socket_phys.send(json.dumps(pose))
            return True
        else:
            return False

    def _getViewerRelativePose(self):
        cpose = np.eye(4)
        gamma = self.yaw
        alpha = self.pitch
        beta  = -self.roll
        #cpose[:3, :3] = transforms3d.euler.euler2mat(alpha, beta, gamma)
        cpose[:3, :3] = transforms3d.quaternions.quat2mat(self.quat)
        cpose[ 0, -1] = self.x
        cpose[ 1, -1] = self.y
        cpose[ 2, -1] = self.z
        return cpose

    def _getViewerAbsolutePose(self, target_pose):
        v_cam2world  = target_pose
        v_cam2cam    = self._getViewerRelativePose()
        p     = v_cam2world.dot(np.linalg.inv(v_cam2cam))
        p     = p.dot(np.linalg.inv(PCRenderer.ROTATION_CONST))
        pos        = utils.mat_to_posi_xyz(p)
        quat_wxyz  = utils.quat_xyzw_to_wxyz(utils.mat_to_quat_xyzw(p))
        return pos, quat_wxyz


    def render(self, imgs, depths, pose, model, poses, target_pose, show):
        v_cam2world = target_pose
        p = (v_cam2world).dot(np.linalg.inv(pose))
        p = p.dot(np.linalg.inv(PCRenderer.ROTATION_CONST))
        s = utils.mat_to_str(p)

        #with Profiler("Depth request round-trip"):
        socket_mist.send(s)
        message = socket_mist.recv()

        #with Profiler("Read from framebuffer and make pano"):
        wo, ho = 768 * 4, 768 * 3

        # Calculate height and width of output image, and size of each square face
        h = wo/3
        w = 2*h
        n = ho/3


        pano = False
        if pano:
            opengl_arr = np.frombuffer(message, dtype=np.float32).reshape((h, w))
        else:
            opengl_arr = np.frombuffer(message, dtype=np.float32).reshape((n, n))

        def _render_depth(opengl_arr):
            #with Profiler("Render Depth"):
            cv2.imshow('target depth', opengl_arr/16.)

        def _render_pc(opengl_arr):

            with Profiler("Render pointcloud cuda"):
                poses_after = [
                    pose.dot(np.linalg.inv(poses[i])).astype(np.float32)
                    for i in range(len(imgs))]
                #opengl_arr = np.zeros((h,w), dtype = np.float32)

                cuda_pc.render(ct.c_int(len(imgs)),
                               ct.c_int(imgs[0].shape[0]),
                               ct.c_int(imgs[0].shape[1]),
                               ct.c_int(768),
                               ct.c_int(768),
                               imgs.ctypes.data_as(ct.c_void_p),
                               depths.ctypes.data_as(ct.c_void_p),
                               np.asarray(poses_after, dtype = np.float32).ctypes.data_as(ct.c_void_p),
                               show.ctypes.data_as(ct.c_void_p),
                               opengl_arr.ctypes.data_as(ct.c_void_p)
                              )

        threads = [
            Process(target=_render_pc, args=(opengl_arr,)),
            Process(target=_render_depth, args=(opengl_arr,))]
        [t.start() for t in threads]
        [t.join() for t in threads]



        if model:
            tf = transforms.ToTensor()
            before = time.time()
            source = tf(show)
            source_depth = tf(np.expand_dims(target_depth, 2).astype(np.float32)/65536 * 255)
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


    def renderOffScreenInitialPose(self):
        ## TODO (hzyjerry): error handling
        pos, quat_wxyz = self._getViewerAbsolutePose(self.target_poses[0])
        pos       = pos.tolist()
        quat_xyzw = utils.quat_wxyz_to_xyzw(quat_wxyz).tolist()
        return pos, quat_xyzw

    def renderOffScreen(self, pose):
        with Profiler("top k selection"):
            ## Query physics engine to get [x, y, z, roll, pitch, yaw]
            new_pos, new_quat = pose[0], pose[1]
            #print("receiving", new_pos, new_quat)
            self.x, self.y, self.z = new_pos
            self.quat = new_quat

            v_cam2world = self.target_poses[0]
            v_cam2cam   = self._getViewerRelativePose()
            cpose = np.linalg.inv(np.linalg.inv(v_cam2world).dot(v_cam2cam).dot(PCRenderer.ROTATION_CONST))

            ## Entry point for change of view
            ## Optimization
            #depth_buffer = np.zeros(self.imgs[0].shape[:2], dtype=np.float32)

            relative_poses = np.copy(self.target_poses)
            for i in range(len(relative_poses)):
                relative_poses[i] = np.dot(np.linalg.inv(relative_poses[i]), self.target_poses[0])

            poses_after = [cpose.dot(np.linalg.inv(relative_poses[i])).astype(np.float32) for i in range(len(self.imgs))]
            pose_after_distance = [np.linalg.norm(rt[:3,-1]) for rt in poses_after]

            topk = (np.argsort(pose_after_distance))[:self.k]

            if set(topk) != self.old_topk:
                self.imgs_topk = np.array([self.imgs[i] for i in topk])
                self.depths_topk = np.array([self.depths[i] for i in topk]).flatten()
                self.relative_poses_topk = [relative_poses[i] for i in topk]
                self.old_topk = set(topk)

        with Profiler("Render pointcloud all"):
            self.render(self.imgs_topk, self.depths_topk, cpose.astype(np.float32), self.model, self.relative_poses_topk, self.target_poses[0], self.show)

            self.show_rgb = cv2.cvtColor(self.show, cv2.COLOR_BGR2RGB)

        #return self.show_rgb

    def renderToScreenSetup(self):
        cv2.namedWindow('show3d')
        cv2.namedWindow('target depth')
        cv2.moveWindow('show3d',1140,0)
        cv2.moveWindow('target depth', 1140, 2048)
        cv2.setMouseCallback('show3d',self._onmouse)

    def renderToScreen(self, pose):
        t0 = time.time()
        self.renderOffScreen(pose)
        t1 = time.time()
        t = t1-t0
        self.fps = 1/t
        cv2.putText(self.show_rgb,'pitch %.3f yaw %.2f roll %.3f x %.2f y %.2f z %.2f'%(self.pitch, self.yaw, self.roll, self.x, self.y, self.z),(15,768-15),0,0.5,(255,255,255))
        cv2.putText(self.show_rgb,'fps %.1f'%(self.fps),(15,15),0,0.5,(255,255,255))

        cv2.imshow('show3d',self.show_rgb)

        ## TODO (hzyjerry): does this introduce extra time delay?
        cv2.waitKey(1)
        return self.show_rgb

def sync_coords():
    with Profiler("Transform coords"):
        new_coords = np.getbuffer(coords.flatten().astype(np.uint32))
    socket_mist.send(new_coords)
    message = socket_mist.recv()


def show_target(target_img):
    cv2.namedWindow('target')
    cv2.moveWindow('target',1032,256 + 50)
    show_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('target', show_rgb)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug'  , action='store_true', help='debug mode')
    parser.add_argument('--datapath'  , required = True, help='dataset path')
    parser.add_argument('--model_id'  , type = str, default = 0, help='model id')
    parser.add_argument('--model'  , type = str, default = '', help='path of model')

    opt = parser.parse_args()
    d = ViewDataSet3D(root=opt.datapath, transform = np.array, mist_transform = np.array, seqlen = 2, off_3d = False, train = True)

    scene_dict = dict(zip(d.scenes, range(len(d.scenes))))
    if not opt.model_id in scene_dict.keys():
        print("model not found")
    else:
        scene_id = scene_dict[opt.model_id]

    uuids, rts = d.get_scene_info(scene_id)
    print(uuids, rts)

    targets = []
    sources = []
    source_depths = []
    poses = []

    for k,v in uuids:
        #print(k,v)
        data = d[v]
        source = data[0][0]
        target = data[1]
        target_depth = data[3]
        source_depth = data[2][0]
        pose = data[-1][0].numpy()
        targets.append(target)
        poses.append(pose)
        sources.append(target)
        source_depths.append(target_depth)

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


    sync_coords()

    show_target(target)

    renderer = PCRenderer(5556, sources, source_depths, target, rts)
    #renderer.renderToScreen(sources, source_depths, poses, model, target, target_depth, rts)
    renderer.renderOffScreenSetup()
    while True:
        print(renderer.renderOffScreen().size)



