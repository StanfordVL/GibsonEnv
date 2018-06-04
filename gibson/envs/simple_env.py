from gibson.envs.env_modalities import CameraRobotEnv, BaseRobotEnv, SemanticRobotEnv
from gibson.envs.env_bases import *
from gibson.core.physics.robot_locomotors import Husky
from gibson.data.datasets import ViewDataSet3D, get_model_path
from gibson.core.physics.scene_stadium import SinglePlayerStadiumScene
from transforms3d import quaternions
import pickle
import os
import numpy as np
import sys
import pybullet as p
import pybullet_data
import cv2
import pdb
import math
import transforms3d.euler
import time
import skimage.io

CALC_OBSTACLE_PENALTY = 1

tracking_camera = {
    'yaw': 110,
    'z_offset': 0.5,
    'distance': 1,
    'pitch': -20
}

tracking_camera_top = {
    'yaw': 20,  # demo: living room, stairs
    'z_offset': 0.5,
    'distance': 1,
    'pitch': -20
}

class CollisionDetectionEnv(BaseEnv):
    def __init__(self, model_id, config, gpu_count=1):
        self.config = self.parse_config(config)
        self.config["model_id"] = model_id
        assert(self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")
        print("stadium" if self.config["model_id"]=="stadium" else "building")
        BaseEnv.__init__(self, self.config, 
                                scene_type="stadium" if self.config["model_id"]=="stadium" else "building",
                                tracking_camera=tracking_camera)
        
        self.scale_up = 2
        self.robot_radius = 0.5
        self.octagon_length = 0.2
        self.get_obj_range()
        self.get_all_locations()
        
        self.robots = {}
        self.robot_introduce((-10., 20., 0.5))
        self.robot = self.robots[(-10., 20., 0.5)]
        self.robot.np_random = self.np_random
        self._robot_introduced = True
        self.total_reward = 0
        self.total_frame = 0

        self.mouse_params = {'box_added' : False,
                'loc' : [100,100],
                'shape_id' : p.createVisualShape(shapeType=p.GEOM_SPHERE,
                    radius=0.1, rgbaColor=[1.,0.,0.,1.])}
        
        self.windowsz = 256
        self._render_width = self.windowsz
        self._render_height = self.windowsz

        self.create_scene()
        self.ground_ids = set(self.scene.scene_obj_list)
        

    def get_ith_orn(self, i):
        rotation = 2. * math.pi * i / 8. - math.pi
        quat = transforms3d.euler.euler2quat( rotation, 0.,-math.pi )
        return quat

    def get_ith_orn_vertices(self, i):
        rotation = 2. * math.pi * i / 8. - math.pi 
        width = 0.25
        length_front = 0.4
        length_back = 0.3
        edges_matrix =  np.asarray([[0,-width],[0,width],[length_front,0],[-length_back,0],
                [length_front, width], [length_front, -width],
                [-length_back, width], [-length_back, -width]])

        rot_matrix = np.asarray([[ math.cos(rotation), math.sin(rotation)],
                                 [-math.sin(rotation), math.cos(rotation)]])
        vertices = np.dot(edges_matrix, rot_matrix)
        return vertices

    def get_all_locations(self):
        x_tics = np.arange(self.x_range[1], 
            self.x_range[0] + self.octagon_length, self.octagon_length)
        y_tics = np.arange(self.y_range[1], 
            self.y_range[0] + self.octagon_length, self.octagon_length)
        self.all_locs = np.stack(np.meshgrid(x_tics, y_tics), axis=2).reshape(-1, 2)
        self.all_locs = np.concatenate((self.all_locs, self.all_locs + self.octagon_length / 2.))

    def inspect_points_and_dir(self):
        with open('/ssd_local/around_ground.pkl', 'rb') as fp:
            points = pickle.load(fp)

        good_p = []
        good_p_d = []
        for po in points:
            print(po)
            start_p = (po[0], po[1], po[2] + 0.1)

            add = False
            for d in range(8):
                viable_d = True
                for loc in self.get_ith_orn_vertices(d):
                    if p.rayTest(start_p, (po[0] + loc[0], po[1] + loc[1], po[2] + 0.1))[0][0] != -1:
                        viable_d = False
                        break
                if viable_d:
                    add = True
                    good_p_d.append((po, d))
            if add:
                good_p.append(po)

        with open('/ssd_local/good_points.pkl', 'wb') as fp:
            pickle.dump(good_p, fp)
        with open('/ssd_local/good_points_and_dir.pkl', 'wb') as fp:
            pickle.dump(good_p_d, fp)

    def inspect_downsample2_points_and_dir(self):
        with open('/ssd_local/good_points_downsample2.pkl', 'rb') as fp:
            points = pickle.load(fp)

        good_p = []
        good_p_d = []
        for po in points:
            print(po)
            start_p = (po[0], po[1], po[2] + 0.1)

            avail_dir = []
            for d in range(8):
                viable_d = True
                for loc in self.get_ith_orn_vertices(d):
                    if p.rayTest(start_p, (po[0] + loc[0], po[1] + loc[1], po[2] + 0.1))[0][0] != -1:
                        viable_d = False
                        break
                if viable_d:
                    avail_dir.append(d)
            if len(avail_dir) > 0:
                good_p.append(po)
                good_p_d.append((*po, avail_dir))

        with open('/ssd_local/downsample2_loc_orn.pkl', 'wb') as fp:
            pickle.dump(good_p_d, fp)
    

    def plot_all_points(self):
        with open('/ssd_local/good_points.pkl', 'rb') as fp:
            data = pickle.load(fp)
        for loc in data:        
            p.createMultiBody(baseMass=1, baseInertialFramePosition=[0,0,0],
                baseVisualShapeIndex = self.mouse_params['shape_id'], basePosition = loc)

    def plot_downsample2_points(self):
        with open('/ssd_local/good_points_downsample2.pkl', 'rb') as fp:
            data = pickle.load(fp)
        for loc in data:        
            p.createMultiBody(baseMass=1, baseInertialFramePosition=[0,0,0],
                baseVisualShapeIndex = self.mouse_params['shape_id'], basePosition = loc)


    def get_grid_collisions(self):
        self.collision = []
        for loc in self.all_locs:
            pos = (loc[0], loc[1], -0.7)
            pos_below = (pos[0], pos[1], 1.5)
            pos_abyss = (pos[0], pos[1], -1e3)
            obj_id, _, _, hit_pos, hit_normal = p.rayTest(pos_below, pos_abyss)[0]
            print(obj_id, hit_pos)
            if obj_id == -1 or obj_id == 2:
                continue
            self.collision.append(hit_pos)
        with open('/ssd_local/locations.pkl', 'wb') as fp:
            pickle.dump(self.collision, fp)

    def set_dir_and_mark(self, i):
        if (len(self.balls) > 0):
            for b in self.balls:
                p.removeBody(b)
            self.balls = []
        self.robot.robot_body.set_orientation(self.get_ith_orn(i))
        vs = self.get_ith_orn_vertices(i)
        for loc in vs:
            self.balls.append(p.createMultiBody(baseMass=1, baseInertialFramePosition=[0,0,0],
                baseVisualShapeIndex = self.mouse_params['shape_id'], basePosition = (-16. + loc[0],14. + loc[1],0.5)))

    def _step(self,a):
        t = time.time()
        pos = ( -10., 20, 1.2 + self.tracking_camera['z_offset'])
        _,_,_,_,_,_,_,_,camY, camP,camDist,_ = p.getDebugVisualizerCamera()
        p.resetDebugVisualizerCamera(camDist,camY, camP,pos)
        time.sleep(3)
   
    def  _reset(self):
        BaseEnv._reset(self)
        self.robot.set_position((-10,20,0.5))
        self.i = 4
        self.balls = []

        self.set_dir_and_mark(self.i) 

        p.createMultiBody(baseMass=1, baseInertialFramePosition=[0,0,0],
                baseVisualShapeIndex = self.mouse_params['shape_id'], basePosition = (-9.,20.,0.5))
        pos = ( -10., 20, 1.2 + self.tracking_camera['z_offset'])
        p.resetDebugVisualizerCamera(self.tracking_camera['distance'], 2. * math.pi  / 8. - math.pi
, self.tracking_camera['pitch'],pos)


    def get_obj_range(self):
        datapath = "/ssd_local/GibsonEnv/gibson/assets/dataset/{}".format(self.model_id)
        input_file  = os.path.join(datapath, 'mesh_z_up.obj')
        f_original = open(input_file)
        vs = []
        fs = []
        for line in f_original:
            if line[:2] == 'v ':
                line = line.split()
                vs.append(list(map(float, line[1:])))
            if line[:2] == 'f ':
                line = line.split()
                fs.append(list(map(lambda x:int(x.split('/')[0]), line[1:])))
        vs = np.array(vs)
        fs = np.array(fs)
        self.x_range = (max(vs[:,0]), min(vs[:,0]))
        self.y_range = (max(vs[:,1]), min(vs[:,1]))

    def robot_introduce(self, loc):
        print("Robot at {}".format(loc))
        config = self.config
        config['initial_pos'] = [loc[0], loc[1], -3.]
        robot = Husky(config, env=self)
        self.robots[loc] = robot
        self.robots[loc].env = self


class RenderAllViewEnv(CameraRobotEnv):

    def __init__(self, config, gpu_count=1):

        self.config = self.parse_config(config)
        assert(self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")

        CameraRobotEnv.__init__(self, self.config, gpu_count, 
                                scene_type="stadium" if self.config["model_id"]=="stadium" else "building",
                                tracking_camera=tracking_camera)

        with open('/ssd_local/RoboCortex/discrete_env_graph/downsample2/downsample2_loc_orn_connected.pkl','rb') as fp:
            self.points = pickle.load(fp) 

        self.curr_point_idx = 0 
        self.store_dir = '/ssd_local/RoboCortex/discrete_env_graph/downsample2/state'

        self.robot_introduce(Husky(self.config, env=self))
        self.scene_introduce()

    def get_ith_orn(self, i):
        rotation = 2. * math.pi * i / 8. - math.pi
        quat = transforms3d.euler.euler2quat( rotation, 0.,-math.pi )
        return quat

    def set_robot_loc_orn(self, point):
        self.robot.set_position((point[0], point[1], point[2]))
        self.robot.robot_body.set_orientation(self.get_ith_orn(point[-1]))

    def _step(self, a):
        if self.curr_point_idx < len(self.points):
            curr_p = self.points[self.curr_point_idx]
            self.curr_point_idx = self.curr_point_idx + 1
            self.set_robot_loc_orn((curr_p[0], curr_p[1], curr_p[2] + 0.14, curr_p[3]))
        
            observations, _, _, _ = CameraRobotEnv._step(self, a)

            rgb = observations['rgb_filled']

            skimage.io.imsave( fname='{}/{}.png'.format(self.store_dir, self.curr_point_idx),
                    arr=rgb )
            depth = observations['depth']

            with open('{}/depth/{}.npy'.format(self.store_dir, self.curr_point_idx), 'wb') as fp:
                np.save(fp, depth)

        else:
            time.sleep(3)



    def _rewards(self, a):
        return [0]

    def _termination(self):
        return False


