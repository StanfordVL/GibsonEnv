from realenv.data.datasets import ViewDataSet3D, get_model_path
from realenv import configs
from realenv.core.render.show_3d2 import PCRenderer
from realenv.core.render.profiler import Profiler
from realenv.envs.env_bases import BaseEnv
import realenv
from gym import error
from gym.utils import seeding
from transforms3d import quaternions
import pybullet as p
from tqdm import *
import subprocess, os, signal
import numpy as np
import sys
import zmq
import pygame
from pygame import surfarray
import socket
import shlex
import gym
import cv2
import os

DEFAULT_TIMESTEP  = 1.0/(4 * 9)
DEFAULT_FRAMESKIP = 4
DEFAULT_DEBUG_CAMERA = {
    'yaw': 30,
    'distance': 2.5,
    'pitch': -35,
    'z_offset': 0
}

class SensorRobotEnv(BaseEnv):
    """Based on BaseEnv
    Handles action, reward
    """

    def __init__(self, scene_type="building", gpu_count=0):
        BaseEnv.__init__(self, scene_type)
        ## The following properties are already instantiated inside xxx_env.py:
        #   @self.human
        #   @self.timestep
        #   @self.frame_skip
        
        self.camera_x = 0
        self.walk_target_x = 1e3  # kilometer away
        self.walk_target_y = 0

        self.k = 5
        self.robot_tracking_id = -1

        self.scale_up  = 4
        self.dataset  = ViewDataSet3D(
            transform = np.array,
            mist_transform = np.array,
            seqlen = 2, 
            off_3d = False, 
            train = False, 
            overwrite_fofn=True)
        self.ground_ids = None
        if self.human:
            assert(self.tracking_camera is not None)
            
        self.action_space = self.robot.action_space
        ## Robot's eye observation, in sensor mode black pixels are returned
        self.observation_space = self.robot.observation_space
        self.sensor_space = self.robot.sensor_space
        
        self.gpu_count = gpu_count
        self.nframe = 0
        self.eps_reward = 0
        
    def get_keys_to_action(self):
        return self.robot.keys_to_action

    def _reset(self):
        debugmode = 1
        if debugmode:
            print("Episode: steps:{} score:{}".format(self.nframe, self.eps_reward))
        self.nframe = 0
        self.eps_reward = 0
        BaseEnv._reset(self)

        if not self.ground_ids:
            self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(
                    self.scene.scene_obj_list)
            self.ground_ids = set(self.scene.scene_obj_list)

        for i in range (p.getNumBodies()):
            if (p.getBodyInfo(i)[0].decode() == self.robot_body.get_name()):
               self.robot_tracking_id=i
            #print(p.getBodyInfo(i)[0].decode())
        i = 0

        state = self.robot.calc_state()

        return state


    electricity_cost     = -2.0 # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
    stall_torque_cost   = -0.1  # cost for running electric current through a motor even at zero rotational speed, small
    foot_collision_cost  = -1.0 # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
    wall_collision_cost = -0.1
    foot_ground_object_names = set(["buildingFloor"])  # to distinguish ground and other objects
    joints_at_limit_cost = -0.1 # discourage stuck joints


    def _step(self, a):
        self.nframe += 1

        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit
        self.rewards, done = self.calc_rewards_and_done(a, state)
        debugmode=0
        if (debugmode):
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))

        self.HUD(state, a, done)
        self.reward += sum(self.rewards)
        self.eps_reward += sum(self.rewards)

        debugmode = 0
        if debugmode:
            print("Eps frame {} reward {}".format(self.nframe, self.reward))
        if self.human:
            humanPos, humanOrn = p.getBasePositionAndOrientation(self.robot_tracking_id)
            humanPos = (humanPos[0], humanPos[1], humanPos[2] + self.tracking_camera['z_offset'])
            if configs.MAKE_VIDEO or configs.DEBUG_CAMERA_FOLLOW:
                p.resetDebugVisualizerCamera(self.tracking_camera['distance'],self.tracking_camera['yaw'], self.tracking_camera['pitch'],humanPos);       ## demo: kitchen, living room
            #p.resetDebugVisualizerCamera(distance,yaw,-42,humanPos);        ## demo: stairs

        eye_pos = self.robot.eyes.current_position()
        debugmode = 0
        if debugmode:
            print("Camera env eye position", eye_pos)
        x, y, z ,w = self.robot.eyes.current_orientation()
        eye_quat = quaternions.qmult([w, x, y, z], self.robot.eye_offset_orn)

        debugmode = 0
        if (debugmode):
            print("rewards")
            print(sum(self.rewards))

        #print(self.reward, self.rewards, self.robot.walk_target_dist_xyz)
        return state, sum(self.rewards), bool(done), dict(eye_pos=eye_pos, eye_quat=eye_quat)

    def calc_rewards(self, a, state):
        print("Please do not directly instantiate CameraRobotEnv")
        raise NotImplementedError()

    def get_eye_pos_orientation(self):
        """Used in CameraEnv.setup"""
        eye_pos = self.robot.eyes.current_position()
        x, y, z ,w = self.robot.eyes.current_orientation()
        eye_quat = quaternions.qmult([w, x, y, z], self.robot.eye_offset_orn)
        return eye_pos, eye_quat        

    def move_robot(self, init_x, init_y, init_z):
        "Used by multiplayer building to move sideways, to another running lane."
        self.cpp_robot.query_position()
        pose = self.cpp_robot.root_part.pose()
        pose.move_xyz(init_x, init_y, init_z)  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
        self.cpp_robot.set_pose(pose)

    def camera_adjust(self):
        x, y, z = self.body_xyz
        self.camera_x = 0.98*self.camera_x + (1-0.98)*x
        self.camera.move_and_look_at(self.camera_x, y-2.0, 1.4, x, y, 1.0)

    def find_best_k_views(self, eye_pos, all_dist, all_pos):
        least_order = (np.argsort(all_dist))
        if len(all_pos) <= p.MAX_RAY_INTERSECTION_BATCH_SIZE:
            collisions = list(p.rayTestBatch([eye_pos] * len(all_pos), all_pos))
        else:
            collisions = []
            curr_i = 0
            while (curr_i < len(all_pos)):
                curr_n = min(len(all_pos), curr_i + p.MAX_RAY_INTERSECTION_BATCH_SIZE - 1)
                collisions = collisions + list(p.rayTestBatch([eye_pos] * (curr_n - curr_i), all_pos[curr_i: curr_n]))
                curr_i = curr_n
        collisions  = [c[0] for c in collisions]
        top_k = []
        for i in range(len(least_order)):
            if len(top_k) >= self.k:
                break
            ## (hzyjerry): disabling ray_casting-based view selection because it gives unstable behaviour right now.
            #if collisions[least_order[i]] < 0:
            top_k.append(least_order[i])
        if len(top_k) < self.k:
            for o in least_order:
                if o not in top_k:
                    top_k.append(o)
                if len(top_k) >= self.k:
                    break 
        return top_k


    def getExtendedObservation(self):
        pass


class CameraRobotEnv(SensorRobotEnv):
    """CameraRobotEnv has full modalities. If it's initialized with mode="SENSOR",
    PC renderer is not initialized to save time. 
    """
    def __init__(self, mode, gpu_count, scene_type, use_filler=True):
        ## The following properties are already instantiated inside xxx_env.py:
        #   @self.human
        #   @self.timestep
        #   @self.frame_skip
        if self.human:
            #self.screen = pygame.display.set_mode([612, 512], 0, 32)
            self.screen_arr = np.zeros([612, 512, 3])
        self.test_env = "TEST_ENV" in os.environ.keys() and os.environ['TEST_ENV'] == "True"
        assert (mode in ["GREY", "RGB", "RGBD", "DEPTH", "SENSOR"]), \
            "Environment mode must be RGB/RGBD/DEPTH/SENSOR"
        assert (self.robot.resolution in ["SMALL", "XSMALL", "MID", "NORMAL", "LARGE", "XLARGE"]), \
            "Robot resolution must be in SMALL/XSMALL/MID/NORMAL/LARGE/XLARGE"
        self.mode = mode
        self.requires_camera_input = mode in ["GREY", "RGB", "RGBD", "DEPTH"]
        self.use_filler = use_filler
        if self.requires_camera_input:
            self.model_path = get_model_path(self.model_id)
        SensorRobotEnv.__init__(self, scene_type, gpu_count)
        if self.robot.resolution == "SMALL":
            self.windowsz = 64
            self.scale_up = 4
        elif self.robot.resolution == "XSMALL":
            self.windowsz = 32
            self.scale_up = 4
        elif self.robot.resolution == "MID":
            self.windowsz = 128
            self.scale_up = 4
        elif self.robot.resolution == "LARGE":
            self.windowsz = 512
            self.scale_up = 1
        elif self.robot.resolution == "NORMAL":
            self.windowsz = 256
            self.scale_up = 4
        elif self.robot.resolution == "XLARGE":
            self.windowsz = 1024
            self.scale_up = 1
        
        self.setup_rendering_camera()
        
        
    def setup_rendering_camera(self):
        if not self.requires_camera_input or self.test_env:
            return
        self.r_camera_rgb = None     ## Rendering engine
        self.r_camera_mul = None     ## Multi channel rendering engine
        self.r_camera_dep = None
        self.check_port_available()
        self.setup_camera_multi()
        self.setup_camera_rgb()

    def _reset(self):
        sensor_state = SensorRobotEnv._reset(self)
        ## This is important to ensure potential doesn't change drastically when reset
        self.potential = self.robot.calc_potential()

        if not self.requires_camera_input or self.test_env:
            visuals = self.get_blank_visuals()
            return visuals, sensor_state

        
        eye_pos, eye_quat = self.get_eye_pos_orientation()
        pose = [eye_pos, eye_quat]
        all_dist, all_pos = self.r_camera_rgb.rankPosesByDistance(pose)
        top_k = self.find_best_k_views(eye_pos, all_dist, all_pos)
        rgb, depth = self.r_camera_rgb.renderOffScreen(pose, top_k)

        #self.screen.fill([0, 0, 0])
        visuals = self.get_visuals(rgb, depth)
        #self.screen.blit(visuals, [200, 200]) 
        #self.screen_arr.fill(0)
        #self.screen_arr[0:rgb.shape[0], 0:rgb.shape[1], :] = rgb
        #surfarray.blit_array(self.screen, self.screen_arr)
        #pygame.display.flip()
        return visuals, sensor_state


    def _step(self, a):
        #with Profiler("Rendering visuals"):
        sensor_state, sensor_reward, done, sensor_meta = SensorRobotEnv._step(self, a)
        pose = [sensor_meta['eye_pos'], sensor_meta['eye_quat']]
        sensor_meta.pop("eye_pos", None)
        sensor_meta.pop("eye_quat", None)
        sensor_meta["sensor"] = sensor_state

        if not self.requires_camera_input or self.test_env:
            visuals = self.get_blank_visuals()
            return visuals, sensor_reward, done, sensor_meta
        
        ## Select the nearest points
        all_dist, all_pos = self.r_camera_rgb.rankPosesByDistance(pose)
        top_k = self.find_best_k_views(pose[0], all_dist, all_pos)
                
        #with Profiler("Render to screen"):
        if not self.human:
            rgb, depth = self.r_camera_rgb.renderOffScreen(pose, top_k)
        else:
            rgb, depth = self.r_camera_rgb.renderToScreen(pose, top_k)
        
        #self.screen_arr[0:rgb.shape[0], 0:rgb.shape[1], :] = rgb
        #surfarray.blit_array(self.screen, self.screen_arr)
        #pygame.display.flip()
        visuals = self.get_visuals(rgb, depth)
        #self.screen.blit(visuals, [200, 200])


        debugmode = 0
        if debugmode:
            print(sensor_meta['eye_pos'])
        debugmode = 0
        if debugmode:
            print("Environment visuals shape", visuals.shape)
        return visuals, sensor_reward, done, sensor_meta
        

    def _close(self):
        if not self.requires_camera_input or self.test_env:
            return
        self.r_camera_mul.terminate()
        if self.r_camera_dep is not None:
            self.r_camera_dep.terminate()
        if configs.SURFACE_NORMAL:
            self.r_camera_norm.terminate()

    def get_blank_visuals(self):
        return np.zeros((256, 256, 4))

    def get_small_blank_visuals(self):
        return np.zeros((64, 64, 1))

    def get_visuals(self, rgb, depth):
        ## Camera specific
        assert(self.requires_camera_input)
        if self.mode == "GREY":
            rgb = np.mean(rgb, axis=2, keepdims=True)
            visuals = np.append(rgb, depth, axis=2)
        elif self.mode == "RGBD":
            visuals = np.append(rgb, depth, axis=2)
        elif self.mode == "RGB":
            visuals = rgb
            if self.robot.observation_space.shape[2] == 4:  ## backward compatibility, will remove in the future
                visuals = np.append(rgb, depth, axis=2)
        elif self.mode == "DEPTH":
            visuals = np.append(rgb, depth, axis=2)         ## RC renderer: rgb = np.zeros()
            if self.robot.observation_space.shape[2] == 1:  ## backward compatibility, will remove in the future
                visuals = depth
        elif self.mode == "SENSOR":
            visuals = np.append(rgb, depth, axis=2)         ## RC renderer: rgb = np.zeros()
        else:
            print("Visual mode not supported: {}".format(self.mode))
            raise AssertionError()
        return visuals

    def setup_camera_rgb(self):
        ## Camera specific
        assert(self.requires_camera_input)
        scene_dict = dict(zip(self.dataset.scenes, range(len(self.dataset.scenes))))
        ## Todo: (hzyjerry) more error handling
        if not self.model_id in scene_dict.keys():
             raise error.Error("Dataset not found: model {} cannot be loaded".format(self.model_id))
        else:
            scene_id = scene_dict[self.model_id]
        uuids, rts = self.dataset.get_scene_info(scene_id)

        targets, sources, source_depths, poses = [], [], [], []
        source_semantics = []

        for k,v in tqdm((uuids)):
            data = self.dataset[v]
            target, target_depth = data[1], data[3]
            target_semantics = data[7]
            if self.scale_up !=1:
                target =  cv2.resize(
                    target,None,
                    fx=1.0/self.scale_up, 
                    fy=1.0/self.scale_up, 
                    interpolation = cv2.INTER_CUBIC)
                target_depth =  cv2.resize(
                    target_depth,None,
                    fx=1.0/self.scale_up, 
                    fy=1.0/self.scale_up, 
                    interpolation = cv2.INTER_CUBIC)
            pose = data[-1][0].numpy()
            targets.append(target)
            poses.append(pose)
            sources.append(target)
            source_depths.append(target_depth)
            source_semantics.append(target_semantics)
        #context_mist = zmq.Context()
        #socket_mist = context_mist.socket(zmq.REQ)
        #socket_mist.connect("tcp://localhost:" + str(5555 + self.gpu_count))
        #context_dept = zmq.Context()
        #socket_dept = context_mist.socket(zmq.REQ)
        #socket_dept.connect("tcp://localhost:" + str(5555 - 1))

        ## TODO (hzyjerry): make sure 5555&5556 are not occupied, or use configurable ports
        renderer = PCRenderer(5556, sources, source_depths, target, rts, self.scale_up, semantics=source_semantics, human=self.human, use_filler=self.use_filler, render_mode=self.mode, gpu_count=self.gpu_count, windowsz=self.windowsz)
        self.r_camera_rgb = renderer


    def setup_camera_multi(self):
        assert(self.requires_camera_input)
        def camera_multi_excepthook(exctype, value, tb):
            print("killing", self.r_camera_mul)
            self.r_camera_mul.terminate()
            if self.r_camera_dep is not None:
                self.r_camera_dep.terminate()
            if configs.SURFACE_NORMAL:
                self.r_camera_norm.terminate()
            while tb:
                filename = tb.tb_frame.f_code.co_filename
                name = tb.tb_frame.f_code.co_name
                lineno = tb.tb_lineno
                print('   File "%.500s", line %d, in %.500s' %(filename, lineno, name))
                tb = tb.tb_next
            print(' %s: %s' %(exctype.__name__, value))
        sys.excepthook = camera_multi_excepthook

        enable_render_smooth = 1 if configs.USE_SMOOTH_MESH else 0

        dr_path = os.path.join(os.path.dirname(os.path.abspath(realenv.__file__)), 'core', 'channels', 'depth_render')
        cur_path = os.getcwd()
        os.chdir(dr_path)
        render_main  = "./depth_render --modelpath {} --GPU {} -w {} -h {}".format(self.model_path, self.gpu_count, self.windowsz, self.windowsz)
        render_depth = "./depth_render --modelpath {} --GPU -1 -s {} -w {} -h {}".format(self.model_path, enable_render_smooth ,self.windowsz, self.windowsz)
        render_norm  = "./depth_render --modelpath {} -n 1 -w {} -h {}".format(self.model_path, self.windowsz, self.windowsz)
        self.r_camera_mul = subprocess.Popen(shlex.split(render_main), shell=False)
        self.r_camera_dep = subprocess.Popen(shlex.split(render_depth), shell=False)

        if configs.SURFACE_NORMAL and configs.MAKE_VIDEO:
            self.r_camera_norm = subprocess.Popen(shlex.split(render_norm), shell=False)

        os.chdir(cur_path)


    def check_port_available(self):
        assert(self.requires_camera_input)
        # TODO (hzyjerry) not working
        """
        s = socket.socket()
        try:
            s.connect(("127.0.0.1", 5555))
        except socket.error as e:
            raise e
            raise error.Error("Realenv starting error: port {} is in use".format(5555))
        try:
            s.connect(("127.0.0.1", 5556))
        except socket.error as e:
            raise error.Error("Realenv starting error: port {} is in use".format(5556))
        """
        return



        