from __future__ import print_function
import numpy as np
import ctypes as ct
import os
import cv2
import sys
import torch
import argparse
import time
import gibson.core.render.utils as utils
import transforms3d
import json
import zmq

from gibson import assets

from torchvision import datasets, transforms
from torch.autograd import Variable
from numpy import cos, sin

from gibson.assets.assets_manager import AssetsManager
from gibson.core.render.profiler import Profiler
from multiprocessing import Process

from gibson.data.datasets import ViewDataSet3D
from gibson.learn.completion import CompletionNet
import torch.nn as nn

assetsManager = AssetsManager()
file_dir = os.path.dirname(os.path.abspath(__file__))
assets_file_dir = assetsManager.get_assets_path()

coords = np.load(os.path.join(AssetsManager().get_assets_path(), 'coord.npy'))


try:
    cuda_pc = np.ctypeslib.load_library(os.path.join(file_dir, 'librender_cuda_f'),'.')
except:
    print("Error: cuda renderer is not loaded, rendering will not work")

LINUX_OFFSET = {
    "x_delta": 10,
    "y_delta": 100
}


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    template = template[template > 0]

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def hist_match3(source, template):
    s0 = hist_match(source[:,:,0], template[:,:,0])
    s1 = hist_match(source[:,:,1], template[:,:,1])
    s2 = hist_match(source[:,:,2], template[:,:,2])
    return np.stack([s0,s1,s2], axis = 2)



class PCRenderer:
    ROTATION_CONST = np.array([[0,1,0,0],[0,0,1,0],[-1,0,0,0],[0,0,0,1]])
    def __init__(self, port, imgs, depths, target, target_poses, scale_up, semantics=None, \
                 gui=True,  use_filler=True, gpu_idx=0, windowsz=256, env = None):

        self.env = env
        self.roll, self.pitch, self.yaw = 0, 0, 0
        self.quat = [1, 0, 0, 0]
        self.x, self.y, self.z = 0, 0, 0
        self.fps = 0
        self.mousex, self.mousey = 0.5, 0.5
        self.org_pitch, self.org_yaw, self.org_roll = 0, 0, 0
        self.org_x, self.org_y, self.org_z = 0, 0, 0
        self.clickstart = (0,0)
        self.mousedown  = False
        self.overlay    = False
        self.show_depth = False

        self.port = port
        self._context_phys = zmq.Context()
        self._context_mist = zmq.Context()
        self._context_dept = zmq.Context()      ## Channel for smoothed depth
        self._context_norm = zmq.Context()      ## Channel for smoothed depth
        self._context_semt = zmq.Context()
        self.env = env

        self._require_semantics = 'semantics' in self.env.config["output"]#configs.View.SEMANTICS in configs.ViewComponent.getComponents()
        self._require_normal = 'normal' in self.env.config["output"] #configs.View.NORMAL in configs.ViewComponent.getComponents()

        self.socket_mist = self._context_mist.socket(zmq.REQ)
        self.socket_mist.connect("tcp://localhost:{}".format(self.port-1))
        #self.socket_dept = self._context_dept.socket(zmq.REQ)
        #self.socket_dept.connect("tcp://localhost:{}".format(5555 - 1))
        if self._require_normal:
            self.socket_norm = self._context_norm.socket(zmq.REQ)
            self.socket_norm.connect("tcp://localhost:{}".format(self.port-2))
        if self._require_semantics:
            self.socket_semt = self._context_semt.socket(zmq.REQ)
            self.socket_semt.connect("tcp://localhost:{}".format(self.port-3))

        self.target_poses = target_poses
        self.pose_locations = np.array([tp[:3,-1] for tp in self.target_poses])

        self.relative_poses = [np.dot(np.linalg.inv(tg), self.target_poses[0]) for tg in target_poses]

        self.imgs = imgs
        self.depths = depths
        self.target = target
        self.semantics = semantics
        self.model = None
        self.old_topk = set([])
        self.k = 5
        self.use_filler = use_filler

        self.showsz = windowsz
        self.capture_count = 0

        #print(self.showsz)
        #self.show   = np.zeros((self.showsz,self.showsz * 2,3),dtype='uint8')
        #self.show_rgb   = np.zeros((self.showsz,self.showsz * 2,3),dtype='uint8')

        self.show            = np.zeros((self.showsz, self.showsz, 3),dtype='uint8')
        self.show_rgb        = np.zeros((self.showsz, self.showsz ,3),dtype='uint8')
        self.show_semantics  = np.zeros((self.showsz, self.showsz ,3),dtype='uint8')

        self.show_prefilled  = np.zeros((self.showsz, self.showsz, 3),dtype='uint8')
        self.surface_normal  = np.zeros((self.showsz, self.showsz, 3),dtype='uint8')

        self.semtimg_count = 0

        if "fast_lq_render" in self.env.config and self.env.config["fast_lq_render"] == True:
            comp = CompletionNet(norm = nn.BatchNorm2d, nf = 24, skip_first_bn = True)
        else:
            comp = CompletionNet(norm=nn.BatchNorm2d, nf=64)
        comp = torch.nn.DataParallel(comp).cuda()
        #comp.load_state_dict(torch.load(os.path.join(assets_file_dir, "model_{}.pth".format(self.env.config["resolution"]))))

        if self.env.config["resolution"] <= 64:
            res = 64
        elif self.env.config["resolution"] <= 128:
            res = 128
        elif self.env.config["resolution"] <= 256:
            res = 256
        else:
            res = 512

        if "fast_lq_render" in self.env.config and self.env.config["fast_lq_render"] == True:
            comp.load_state_dict(
            torch.load(os.path.join(assets_file_dir, "model_small_{}.pth".format(res))))
        else:
            comp.load_state_dict(
            torch.load(os.path.join(assets_file_dir, "model_{}.pth".format(res))))

        #comp.load_state_dict(torch.load(os.path.join(file_dir, "model.pth")))
        #comp.load_state_dict(torch.load(os.path.join(file_dir, "model_large.pth")))
        self.model = comp.module
        self.model.eval()

        if not self.env.config["use_filler"]:
            self.model = None

        self.imgs_topk = None
        self.depths_topk = None
        self.relative_poses_topk = None
        self.old_topk = None

        self.imgv = Variable(torch.zeros(1, 3 , self.showsz, self.showsz), volatile = True).cuda()
        self.maskv = Variable(torch.zeros(1,2, self.showsz, self.showsz), volatile = True).cuda()
        self.mean = torch.from_numpy(np.array([0.57441127,  0.54226291,  0.50356019]).astype(np.float32))
        self.mean = self.mean.view(3,1,1).repeat(1,self.showsz,self.showsz)

        if gui and not self.env.config["display_ui"]:
            self.renderToScreenSetup()


    def _close(self):
        self._context_dept.destroy()
        self._context_mist.destroy()
        self._context_norm.destroy()
        self._context_phys.destroy()


    def renderToScreenSetup(self):
        cv2.namedWindow('RGB cam')
        cv2.namedWindow('Depth cam')
        #if MAKE_VIDEO:
        #    cv2.moveWindow('RGB cam', -1 , self.showsz + LINUX_OFFSET['y_delta'])
        #    cv2.moveWindow('Depth cam', self.showsz + LINUX_OFFSET['x_delta'] + LINUX_OFFSET['y_delta'], -1)
        cv2.namedWindow('RGB prefilled')
        cv2.namedWindow('Semantics')
        cv2.namedWindow('Surface Normal')
        #    cv2.moveWindow('Surface Normal', self.showsz + self.showsz + LINUX_OFFSET['x_delta'] + LINUX_OFFSET['y_delta'], -1)
        #    cv2.moveWindow('RGB prefilled', self.showsz + LINUX_OFFSET['x_delta'] + LINUX_OFFSET['y_delta'], self.showsz + LINUX_OFFSET['y_delta'])
        #    cv2.moveWindow('Semantics', self.showsz + self.showsz + LINUX_OFFSET['x_delta'] + LINUX_OFFSET['y_delta'], self.showsz + LINUX_OFFSET['y_delta'])
        #elif HIGH_RES_MONITOR:
        #    cv2.moveWindow('RGB cam', -1 , self.showsz + LINUX_OFFSET['y_delta'])
        #    cv2.moveWindow('Depth cam', self.showsz + LINUX_OFFSET['x_delta'] + LINUX_OFFSET['y_delta'], self.showsz + LINUX_OFFSET['y_delta'])
        #
        #if LIVE_DEMO:
        #    cv2.moveWindow('RGB cam', -1 , 768)
        #    cv2.moveWindow('Depth cam', 512, 768)


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
        #elif cmd == ord('v'):
        #    cv2.imwrite('save.jpg', show_rgb)
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


    def render(self, rgbs, depths, pose, model, poses, target_pose, show, show_prefilled=None, is_rgb=False):
        v_cam2world = target_pose
        p = (v_cam2world).dot(np.linalg.inv(pose))
        p = p.dot(np.linalg.inv(PCRenderer.ROTATION_CONST))
        s = utils.mat_to_str(p)

        #with Profiler("Render: depth request round-trip"):
        ## Speed bottleneck: 100fps for mist + depth
        self.socket_mist.send_string(s)
        mist_msg = self.socket_mist.recv()
        #self.socket_dept.send_string(s)
        #dept_msg = self.socket_dept.recv()
        if self._require_normal:
            self.socket_norm.send_string(s)
            norm_msg = self.socket_norm.recv()
        if self._require_semantics:
            self.socket_semt.send_string(s)
            semt_msg = self.socket_semt.recv()


        wo, ho = self.showsz * 4, self.showsz * 3

        # Calculate height and width of output image, and size of each square face
        h = wo//3
        w = 2*h
        n = ho//3

        need_filler = True
        pano = False
        if pano:
            opengl_arr = np.frombuffer(mist_msg, dtype=np.float32).reshape((h, w))
            #smooth_arr = np.frombuffer(dept_msg, dtype=np.float32).reshape((h, w))
            if self._require_normal:
                normal_arr = np.frombuffer(norm_msg, dtype=np.float32).reshape((n, n, 3))
            if self._require_semantics:
                semantic_arr = np.frombuffer(semt_msg, dtype=np.uint32).reshape((n, n, 3))
        else:
            opengl_arr = np.frombuffer(mist_msg, dtype=np.float32).reshape((n, n))
            #smooth_arr = np.frombuffer(dept_msg, dtype=np.float32).reshape((n, n))
            if self._require_normal:
                normal_arr = np.frombuffer(norm_msg, dtype=np.float32).reshape((n, n, 3))
            if self._require_semantics:
                semantic_arr = np.frombuffer(semt_msg, dtype=np.uint32).reshape((n, n, 3))

        debugmode = 0
        if debugmode and self._require_normal:
            print("Inside show3d: surface normal max", np.max(normal_arr), "mean", np.mean(normal_arr))

        def _render_pc(opengl_arr, imgs_pc, show_pc):
            #with Profiler("Render pointcloud cuda", enable=ENABLE_PROFILING):
            poses_after = [
                pose.dot(np.linalg.inv(poses[i])).astype(np.float32)
                for i in range(len(imgs_pc))]
            #opengl_arr = np.zeros((h,w), dtype = np.float32)
            cuda_pc.render(ct.c_int(len(imgs_pc)),
                           ct.c_int(imgs_pc[0].shape[0]),
                           ct.c_int(imgs_pc[0].shape[1]),
                           ct.c_int(self.showsz),
                           ct.c_int(self.showsz),
                           imgs_pc.ctypes.data_as(ct.c_void_p),
                           depths.ctypes.data_as(ct.c_void_p),
                           np.asarray(poses_after, dtype = np.float32).ctypes.data_as(ct.c_void_p),
                           show_pc.ctypes.data_as(ct.c_void_p),
                           opengl_arr.ctypes.data_as(ct.c_void_p),
                           ct.c_float(self.env.config["fov"])
                          )

        #threads = [
        #    Process(target=_render_pc, args=(opengl_arr,)),
        #    Process(target=_render_depth, args=(opengl_arr,))]
        #[t.start() for t in threads]
        #[t.join() for t in threads]

        #with Profiler("Render: render point cloud"):
        ## Speed bottleneck

        if is_rgb:
            _render_pc(opengl_arr, rgbs, show)
            # Store prefilled rgb
            show_prefilled[:] = show

        #with Profiler("Render: NN total time"):
        ## Speed bottleneck
        if self.use_filler and self.model and is_rgb and need_filler:
            tf = transforms.ToTensor()
            #from IPython import embed; embed()
            source = tf(show)
            mask = (torch.sum(source[:3,:,:],0)>0).float().unsqueeze(0)
            source += (1-mask.repeat(3,1,1)) * self.mean
            source_depth = tf(np.expand_dims(opengl_arr, 2).astype(np.float32)/128.0 * 255)
            mask = torch.cat([source_depth, mask], 0)
            #self.imgv.data.copy_(source)
            #self.maskv.data.copy_(mask)
            #print(torch.max(self.maskv), torch.max(self.imgv))

            imgv = Variable(source).cuda().unsqueeze(0)
            maskv = Variable(mask).cuda().unsqueeze(0)

            #print(imgv.size(), maskv.size())

            recon = model(imgv, maskv)
            show2 = recon.data.clamp(0,1).cpu().numpy()[0].transpose(1,2,0)
            show[:] = (show2[:] * 255).astype(np.uint8)

        self.target_depth = opengl_arr ## target depth
        #self.smooth_depth = smooth_arr
        if self._require_normal:
            self.surface_normal = normal_arr
        if self._require_semantics:
            self.show_semantics = semantic_arr
            debugmode = 0
            if debugmode:
                print("Semantics array", np.max(semantic_arr), np.min(semantic_arr), np.mean(semantic_arr), semantic_arr.shape)



    def renderOffScreenInitialPose(self):
        ## TODO (hzyjerry): error handling
        pos, quat_wxyz = self._getViewerAbsolutePose(self.target_poses[0])
        pos       = pos.tolist()
        quat_xyzw = utils.quat_wxyz_to_xyzw(quat_wxyz).tolist()
        return pos, quat_xyzw

    def setNewPose(self, pose):
        new_pos, new_quat = pose[0], pose[1]
        self.x, self.y, self.z = new_pos
        self.quat = new_quat
        v_cam2world = self.target_poses[0]
        v_cam2cam   = self._getViewerRelativePose()
        self.render_cpose = np.linalg.inv(np.linalg.inv(v_cam2world).dot(v_cam2cam).dot(PCRenderer.ROTATION_CONST))

    def getAllPoseDist(self, pose):
        ## Query physics engine to get [x, y, z, roll, pitch, yaw]
        new_pos, new_quat = pose[0], pose[1]
        pose_distances = np.linalg.norm(self.pose_locations - pose[0].reshape(1,3), axis = 1)
        #topk = (np.argsort(pose_after_distance))[:self.k]
        return pose_distances, self.pose_locations


    def renderOffScreen(self, pose, k_views=None, rgb=True):

        if k_views is not None:
            all_dist, _ = self.getAllPoseDist(pose)
            k_views = (np.argsort(all_dist))[:self.k]
        if rgb and set(k_views) != self.old_topk:
            self.imgs_topk = np.array([self.imgs[i] for i in k_views])
            self.depths_topk = np.array([self.depths[i] for i in k_views]).flatten()
            self.relative_poses_topk = [self.relative_poses[i] for i in k_views]
            #self.semantics_topk = np.array([self.semantics[i] for i in k_views])
            self.old_topk = set(k_views)

        self.render(self.imgs_topk, self.depths_topk, self.render_cpose.astype(np.float32), self.model, self.relative_poses_topk, self.target_poses[0], self.show, self.show_prefilled, is_rgb=rgb)

        self.show = np.reshape(self.show, (self.showsz, self.showsz, 3))
        self.show_rgb = self.show
        self.show_prefilled_rgb = self.show_prefilled

        return self.show_rgb, self.target_depth[:, :, None], self.show_semantics, self.surface_normal, self.show_prefilled_rgb


    def renderToScreen(self):
        def _render_depth(depth):
            cv2.imshow('Depth cam', depth/16.)

        def _render_rgb(rgb):
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            cv2.imshow('RGB cam',rgb)

        def _render_rgb_prefilled(prefilled_rgb):
            ## TODO: legacy MAKE_VIDEO
            cv2.imshow('RGB prefilled', prefilled_rgb)

        def _render_semantics(semantics):
            if not self._require_semantics: return
            cv2.imshow('Semantics', semantics)

        def _render_normal(normal):
            if not self._require_normal: return
            cv2.imshow("Surface Normal", normal)

        _render_depth(self.target_depth)
        #_render_depth(self.smooth_depth)
        _render_rgb(self.show_rgb)
        ## TODO (hzyjerry): does this introduce extra time delay?
        cv2.waitKey(1)
        #return self.show_rgb, self.target_depth[:, :, None]
        return self.show_rgb, self.target_depth[:, :, None]


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


    show_target(target)

    renderer = PCRenderer(5556, sources, source_depths, target, rts)
    #renderer.renderToScreen(sources, source_depths, poses, model, target, target_depth, rts)
    renderer.renderOffScreenSetup()
    while True:
        print("renderingOffScreen")
        print(renderer.renderOffScreen().size)
