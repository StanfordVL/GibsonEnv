from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os, time
import os.path
from multiprocessing import Pool
from functools import partial

from gibson.assets.assets_manager import AssetsManager
from gibson.core.render.profiler import Profiler
import errno
import torch
import json
import codecs
import cv2
import numpy as np
import ctypes as ct
import sys
from tqdm import *
import torchvision.transforms as transforms
import argparse
import json
from numpy.linalg import inv
import pickle
from gibson import assets

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    ## Heavy usage
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)#.convert('RGB')
    #img = Image.open(path)
    return img


def depth_loader(path):
    ## Heavy usage
    ## TODO: Image.open for depth image is main data loading bottleneck
    #img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)#.convert('I')
    img = Image.open(path)
    return img


def get_model_path(model_id):
    data_path = os.path.join(AssetsManager().get_assets_path(), 'dataset')
    assert (model_id in os.listdir(data_path)) or model_id == 'stadium', "Model {} does not exist".format(model_id)
    return os.path.join(data_path, model_id)

def get_item_fn(inds, select, root, loader, transform, off_3d, target_transform, depth_trans, off_pc_render, dll, train, require_rgb):
    """ Functional programming version of Dataset.__getitem__
    The advantage is that it is pickle-friendly and supports python multiprocessing
    
    Argument: 
        inds: tuple of scene index and output index
    """
    index, out_i = inds
    scene = select[index][0][0]
    uuids = [item[1] for item in select[index]]
    paths = ([os.path.join(root, scene, 'pano', 'rgb', "point_" + item + "_view_equirectangular_domain_rgb.png") for item in uuids])
    mist_paths = ([os.path.join(root, scene, 'pano', 'mist', "point_" + item + "_view_equirectangular_domain_mist.png") for item in uuids])
    normal_paths = ([os.path.join(root, scene, 'pano', 'normal', "point_" + item + "_view_equirectangular_domain_normal.png") for item in uuids])
    pose_paths = ([os.path.join(root, scene, 'pano', 'points', "point_" + item + ".json") for item in uuids])
    semantic_paths = ([os.path.join(root, scene, 'pano', 'semantic', "point_" + item + "_view_equirectangular_domain_semantic.png") for item in uuids])
    poses = []
    for i, item in enumerate(pose_paths):
        f = open(item)
        pose_dict = json.load(f)
        p = np.concatenate(np.array(pose_dict[1][u'camera_rt_matrix'])).astype(np.float32).reshape((4, 4))
        rotation = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        p = np.dot(p, rotation)
        poses.append(p)
        f.close()

    img_paths = paths[1:]
    target_path = paths[0]
    img_poses = poses[1:]
    target_pose = poses[0]

    mist_img_paths = mist_paths[1:]
    mist_target_path = mist_paths[0]

    normal_img_paths = normal_paths[1:]
    normal_target_path = normal_paths[0]

    semantic_img_paths = semantic_paths[1:]
    semantic_target_path = semantic_paths[0]
    poses_relative = []

    semantic_imgs = None
    semantic_target = None

    normal_imgs = None
    normal_target = None

    mist_imgs = None
    mist_target = None

    imgs = None
    target = None

    for pose_i, item in enumerate(img_poses):
        pose_i = pose_i + 1
        relative = np.dot(inv(target_pose), item)
        poses_relative.append(torch.from_numpy(relative))
    
    if require_rgb:
        imgs = [loader(item) for item in img_paths]
        target = loader(target_path)
        org_img = imgs[0].copy()

    if not off_3d and require_rgb:
        mist_imgs = [depth_loader(item) for item in mist_img_paths]
        mist_target = depth_loader(mist_target_path)
        if train:
            normal_imgs = [loader(item) for item in normal_img_paths]
            normal_target = loader(normal_target_path)


    if not transform is None and require_rgb:
        imgs = [transform(item) for item in imgs]
    if not target_transform is None and require_rgb:
        target = target_transform(target)

    if not off_3d and require_rgb:
        mist_imgs = [np.expand_dims(np.array(item).astype(np.float32) / 65536.0, 2) for item in mist_imgs]
        org_mist = mist_imgs[0][:, :, 0].copy()
        mist_target = np.expand_dims(np.array(mist_target).astype(np.float32) / 65536.0, 2)

        if not depth_trans is None:
            mist_imgs = [depth_trans(item) for item in mist_imgs]
        if not depth_trans is None:
            mist_target = depth_trans(mist_target)

        if train:
            if not transform is None:
                normal_imgs = [transform(item) for item in normal_imgs]
            if not target_transform is None:
                normal_target = target_transform(normal_target)

    if not off_pc_render and require_rgb:
        img = np.array(org_img)
        h, w, _ = img.shape
        render = np.zeros((h, w, 3), dtype='uint8')
        target_depth = np.zeros((h, w)).astype(np.float32)
        depth = org_mist
        pose = poses_relative[0].numpy()
        dll.render(ct.c_int(img.shape[0]),
                        ct.c_int(img.shape[1]),
                        img.ctypes.data_as(ct.c_void_p),
                        depth.ctypes.data_as(ct.c_void_p),
                        pose.ctypes.data_as(ct.c_void_p),
                        render.ctypes.data_as(ct.c_void_p),
                        target_depth.ctypes.data_as(ct.c_void_p)
                        )
        if not transform is None:
            render = transform(Image.fromarray(render))
        if not depth_trans is None:
            target_depth = depth_trans(np.expand_dims(target_depth, 2))

    if off_3d:
        out = (imgs, target, poses_relative)
    elif off_pc_render:
        out = (imgs, target, mist_imgs, mist_target, normal_imgs, normal_target, poses_relative)
    else:
        out = (imgs, target, mist_imgs, mist_target, normal_imgs, normal_target, poses_relative, render, target_depth)
    return (out_i, out)


class ViewDataSet3D(data.Dataset):
    def __init__(self, root=None, train=False, transform=None, mist_transform=None, loader=default_loader, seqlen=5,
                 debug=False, dist_filter=None, off_3d=True, off_pc_render=True, overwrite_fofn=False,
                 semantic_transform=np.array, env = None, only_load = None):
        print('Processing the data:')
        if not root:
            self.root = os.path.join(AssetsManager().get_assets_path(), "dataset")
        else:
            self.root = root
        self.train = train
        self.env = env
        self.loader = loader
        self.seqlen = seqlen
        self.transform = transform
        self.target_transform = transform
        self.depth_trans = mist_transform
        self.semantic_trans = semantic_transform
        self._require_semantics = "SEMANTICS" in self.env.config["ui_components"]
        self._require_rgb = "RGB_FILLED" in self.env.config["ui_components"] or "RGB_PREFILLED" in self.env.config["ui_components"] or "rgb_filled" in self.env.config["output"] or "rgb_prefill" in self.env.config["output"]
        self.off_3d = off_3d
        self.select = []
        self.fofn = self.root + '_fofn' + str(int(train)) + '.pkl'
        self.off_pc_render = off_pc_render
        self.dll = None

        if not self.off_pc_render:
            self.dll = np.ctypeslib.load_library('render', '.')

        if overwrite_fofn or not os.path.isfile(self.fofn):

            if only_load is None:
                self.scenes = sorted([d for d in (os.listdir(self.root)) if
                                  os.path.isdir(os.path.join(self.root, d)) and os.path.isfile(
                                      os.path.join(self.root, d, 'camera_poses.csv')) and os.path.isdir(
                                      os.path.join(self.root, d, 'pano'))])

                num_scenes = len(self.scenes)
                num_train = int(num_scenes * 0.9)

            else:

                self.scenes = sorted([only_load])
                num_scenes = 1
                num_train = 0


            print("Total %d scenes %d train %d test" % (num_scenes, num_train, num_scenes - num_train))
            if train:
                self.scenes = self.scenes[:num_train]

            self.meta = {}

            last = len(self.scenes)

            for scene in self.scenes[:last]:
                posefile = os.path.join(self.root, scene, 'camera_poses.csv')
                with open(posefile) as f:
                    for line in f:
                        l = line.strip().split(',')
                        uuid = l[0]
                        xyz = list(map(float, l[1:4]))
                        quat = list(map(float, l[4:8]))
                        if not scene in self.meta:
                            self.meta[scene] = {}
                        metadata = (uuid, xyz, quat)
                        # print(uuid, xyz)
                        if os.path.isfile(os.path.join(self.root, scene, 'pano', 'points', 'point_' + uuid + '.json')):
                            if np.linalg.norm( np.array(xyz) - np.array([0,0,0])) > 1e-5: #remove scans that are not registered
                                self.meta[scene][uuid] = metadata
            print("Indexing")

            for scene, meta in tqdm(list(self.meta.items())):
                if len(meta) < self.seqlen:
                    continue
                for uuid, v in list(meta.items()):
                    dist_list = [(uuid2, np.linalg.norm(np.array(v2[1]) - np.array(v[1]))) for uuid2, v2 in list(meta.items())]
                    dist_list = sorted(dist_list, key=lambda x: x[-1])

                    if not dist_filter is None:
                        if dist_list[1][-1] < dist_filter:
                            self.select.append([[scene, dist_list[i][0], dist_list[i][1]] for i in range(self.seqlen)])

                    else:
                        self.select.append([[scene, dist_list[i][0], dist_list[i][1]] for i in range(self.seqlen)])

            with open(self.fofn, 'wb') as fp:
                pickle.dump([self.scenes, self.meta, self.select, num_scenes, num_train], fp)

        else:
            with open(self.fofn, 'rb') as fp:
                self.scenes, self.meta, self.select, num_scenes, num_train = pickle.load(fp)
                print("Total %d scenes %d train %d test" % (num_scenes, num_train, num_scenes - num_train))


    def get_scene_info(self, index):
        scene = self.scenes[index]
        data = [(i, item) for i, item in enumerate(self.select) if item[0][0] == scene]
        uuids = ([(item[1][0][1], item[0]) for item in data])
        
        pose_paths = (
        [os.path.join(self.root, scene, 'pano', 'points', "point_" + item[0] + ".json") for item in uuids])
        poses = []
        for item in pose_paths:
            f = open(item)
            pose_dict = json.load(f)
            p = np.concatenate(np.array(pose_dict[1][u'camera_rt_matrix'])).astype(np.float32).reshape((4, 4))

            rotation = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])

            p = np.dot(p, rotation)
            poses.append(p)
            f.close()

        return uuids, poses

    def __getitem__(self, index):
        scene = self.select[index][0][0]
        uuids = [item[1] for item in self.select[index]]
        paths = (
        [os.path.join(self.root, scene, 'pano', 'rgb', "point_" + item + "_view_equirectangular_domain_rgb.png") for
         item in uuids])
        mist_paths = (
        [os.path.join(self.root, scene, 'pano', 'mist', "point_" + item + "_view_equirectangular_domain_mist.png") for
         item in uuids])
        normal_paths = (
        [os.path.join(self.root, scene, 'pano', 'normal', "point_" + item + "_view_equirectangular_domain_normal.png")
         for item in uuids])
        pose_paths = ([os.path.join(self.root, scene, 'pano', 'points', "point_" + item + ".json") for item in uuids])
        semantic_paths = ([os.path.join(self.root, scene, 'pano', 'semantic',
                                        "point_" + item + "_view_equirectangular_domain_semantic.png") for item in
                           uuids])
        poses = []
        for i, item in enumerate(pose_paths):
            f = open(item)
            pose_dict = json.load(f)
            p = np.concatenate(np.array(pose_dict[1][u'camera_rt_matrix'])).astype(np.float32).reshape((4, 4))
            rotation = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
            p = np.dot(p, rotation)
            poses.append(p)
            f.close()

        img_paths = paths[1:]
        target_path = paths[0]
        img_poses = poses[1:]
        target_pose = poses[0]

        mist_img_paths = mist_paths[1:]
        mist_target_path = mist_paths[0]

        normal_img_paths = normal_paths[1:]
        normal_target_path = normal_paths[0]

        semantic_img_paths = semantic_paths[1:]
        semantic_target_path = semantic_paths[0]
        poses_relative = []

        semantic_imgs = None
        semantic_target = None

        normal_imgs = None
        normal_target = None

        mist_imgs = None
        mist_target = None

        imgs, target = None, None

        for pose_i, item in enumerate(img_poses):
            pose_i = pose_i + 1
            relative = np.dot(inv(target_pose), item)
            poses_relative.append(torch.from_numpy(relative))

        if self._require_rgb:
            imgs = [self.loader(item) for item in img_paths]
            target = self.loader(target_path)

        if not self.off_3d and self._require_rgb:
            mist_imgs = [depth_loader(item) for item in mist_img_paths]
            mist_target = depth_loader(mist_target_path)
            if self.train:      # Optimize
                normal_imgs = [self.loader(item) for item in normal_img_paths]
                normal_target = self.loader(normal_target_path)

        if not self.off_pc_render and self._require_rgb:
            org_img = imgs[0].copy()
            if not self.transform is None:
                imgs = [self.transform(item) for item in imgs]
            if not self.target_transform is None:
                target = self.target_transform(target)

        if not self.off_3d and self._require_rgb:
            mist_imgs = [np.expand_dims(np.array(item).astype(np.float32) / 65536.0, 2) for item in mist_imgs]
            if not self.off_pc_render:
                org_mist = mist_imgs[0][:, :, 0].copy()
            mist_target = np.expand_dims(np.array(mist_target).astype(np.float32) / 65536.0, 2)

            if not self.depth_trans is None:
                mist_imgs = [self.depth_trans(item) for item in mist_imgs]
            if not self.depth_trans is None:
                mist_target = self.depth_trans(mist_target)

            if self.train:
                if not self.transform is None:
                    normal_imgs = [self.transform(item) for item in normal_imgs]
                if not self.target_transform is None:
                    normal_target = self.target_transform(normal_target)

        if not self.off_pc_render and self._require_rgb:
            img = np.array(org_img)
            h, w, _ = img.shape
            render = np.zeros((h, w, 3), dtype='uint8')
            target_depth = np.zeros((h, w)).astype(np.float32)
            depth = org_mist
            pose = poses_relative[0].numpy()
            self.dll.render(ct.c_int(img.shape[0]),
                            ct.c_int(img.shape[1]),
                            img.ctypes.data_as(ct.c_void_p),
                            depth.ctypes.data_as(ct.c_void_p),
                            pose.ctypes.data_as(ct.c_void_p),
                            render.ctypes.data_as(ct.c_void_p),
                            target_depth.ctypes.data_as(ct.c_void_p)
                            )
            if not self.transform is None:
                render = self.transform(Image.fromarray(render))
            if not self.depth_trans is None:
                target_depth = self.depth_trans(np.expand_dims(target_depth, 2))

        if self.off_3d:
            return imgs, target, poses_relative
        elif self.off_pc_render:
            return imgs, target, mist_imgs, mist_target, normal_imgs, normal_target, poses_relative
        else:
            return imgs, target, mist_imgs, mist_target, normal_imgs, normal_target, poses_relative, render, target_depth

    def get_multi_index(self, uuids):
        indices = range(len(uuids))
        p = Pool(16)
        partial_fn = partial(get_item_fn, select=self.select, root=self.root, loader=self.loader, transform=self.transform, off_3d=self.off_3d, target_transform=self.target_transform, depth_trans=self.depth_trans, off_pc_render=self.off_pc_render, dll=self.dll, train=self.train, require_rgb=self._require_rgb)
        mapped_pairs = list(tqdm(p.imap(partial_fn, list(zip(uuids, indices))), total=len(uuids)))
        sorted_pairs = sorted(mapped_pairs, key=lambda x: x[0])
        out_data = [key_pair[1] for key_pair in sorted_pairs]
        p.close()
        p.join()
        return out_data

    def __len__(self):
        return len(self.select)


########### BELOW THIS POINT: Legacy code #################
########### KEEPING ONLY FOR REFERENCE ####################


class Places365Dataset(data.Dataset):
    def __init__(self, root, train=True, transform=None, loader=default_loader):
        self.root = root.rstrip('/')
        self.train = train
        self.fns = []
        self.fofn = os.path.basename(root) + '_fofn' + str(int(train)) + '.pkl'
        self.loader = loader
        self.transform = transform
        if not os.path.isfile(self.fofn):
            for subdir, dirs, files in os.walk(self.root):
                if self.train:
                    files = files[:len(files) / 10 * 9]
                else:
                    files = files[len(files) / 10 * 9:]
                print(subdir)
                for file in files:
                    self.fns.append(os.path.join(subdir, file))
            with open(self.fofn, 'wb') as fp:
                pickle.dump(self.fns, fp)
        else:
            with open(self.fofn, 'rb') as fp:
                self.fns = pickle.load(fp)

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, index):
        path = self.fns[index]
        img = self.loader(path)
        if not self.transform is None:
            img = self.transform(img)
        return img


class PairDataset(data.Dataset):
    def __init__(self, root, train=True, transform=None, mist_transform=None, loader=np.load):
        self.root = root.rstrip('/')
        self.train = train
        self.fns = []
        self.fofn = os.path.basename(root) + '_fofn' + str(int(train)) + '.pkl'
        self.loader = loader
        self.transform = transform
        self.mist_transform = mist_transform
        if not os.path.isfile(self.fofn):
            for subdir, dirs, files in os.walk(self.root):
                if self.train:
                    files = files[:len(files) / 10 * 9]
                else:
                    files = files[len(files) / 10 * 9:]
                print(subdir)
                for file in files:
                    if file[-3:] == 'npz':
                        self.fns.append(os.path.join(subdir, file))
            with open(self.fofn, 'wb') as fp:
                pickle.dump(self.fns, fp)
        else:
            with open(self.fofn, 'rb') as fp:
                self.fns = pickle.load(fp)

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, index):
        path = self.fns[index]
        data = self.loader(path)

        try:
            source, depth, target = data['source'], data['depth'], data['target']
            # print(source.shape, depth.shape, target.shape)
        except:
            source = np.zeros((1024, 2048, 3)).astype(np.uint8)
            target = np.zeros((1024, 2048, 3)).astype(np.uint8)
            depth = np.zeros((1024, 2048)).astype(np.float32)

        if not self.transform is None:
            source = self.transform(source)
            target = self.transform(target)
            # depth = self.mist_transform(depth)
            depth = torch.from_numpy(depth.astype(np.float32))
        return source, depth, target


if __name__ == '__main__':
    print('test')
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--dataset', required=True, help='dataset type')
    opt = parser.parse_args()

    if opt.dataset == 'view3d':
        d = ViewDataSet3D(root=opt.dataroot, debug=opt.debug, seqlen=2, dist_filter=0.8, off_3d=False,
                          off_pc_render=False)
        print(len(d))
        sample = (d[1])
        print(sample)
        if sample is not None:
            print('3d test passed')

        uuids, xyzs, poses = d.get_scene_info(0)
        print(uuids, xyzs, poses)

    elif opt.dataset == 'places365':
        d = Places365Dataset(root=opt.dataroot)
        print(len(d))
        sample = d[0]
        print(sample)
        if sample is not None:
            print('places 365 test passed')

    elif opt.dataset == 'pair':
        d = PairDataset(root=opt.dataroot)
        print(len(d))
        sample = d[0]
        print(sample)
        if sample is not None:
            print('pair test passed')
