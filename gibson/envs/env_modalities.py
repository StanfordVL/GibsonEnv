from gibson.data.datasets import ViewDataSet3D, get_model_path
from gibson.core.render.pcrender import PCRenderer
from gibson.core.render.profiler import Profiler
from gibson.envs.env_bases import BaseEnv
from gibson.envs.env_utils import *
import gibson
from gym import error
from gym.utils import seeding
from transforms3d import quaternions
from gibson.envs.env_ui import *
import pybullet as p
import pybullet_data
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
import os.path as osp
import os
from PIL import Image

from transforms3d.euler import euler2quat, euler2mat
from transforms3d.quaternions import quat2mat, qmult
import transforms3d.quaternions as quat
import time

DEFAULT_TIMESTEP  = 1.0/(4 * 9)
DEFAULT_FRAMESKIP = 4
DEFAULT_DEBUG_CAMERA = {
    'yaw': 30,
    'distance': 2.5,
    'pitch': -35,
    'z_offset': 0
}

DEPTH_SCALE_FACTOR = 35
DEPTH_OFFSET_FACTOR = 20

class BaseRobotEnv(BaseEnv):
    """Based on BaseEnv
    Handles action, reward
    """
    DEFAULT_PORT = 5556

    def __init__(self, config, tracking_camera, scene_type="building", gpu_idx=0):
        BaseEnv.__init__(self, config, scene_type, tracking_camera)

        self.camera_x = 0
        self.walk_target_x = 1e3  # kilometer away
        self.walk_target_y = 0
        self.k = 5
        self.robot_tracking_id = -1

        self.scale_up  = 4
        self.dataset = None
        self.ground_ids = None
        if self.gui:
            assert(self.tracking_camera is not None)
        self.gpu_idx = gpu_idx
        self.assign_ports()
        self.nframe = 0
        self.eps_reward = 0

        self.reward = 0
        self.eps_count = 0

        self._robot_introduced = False
        self._scene_introduced = False

    def assign_ports(self):
        '''Rendering multiple modalities (RGB, depth, normal) needs to be done 
        on different ports. Assign individual ports to each modality:

        | Rendering | Port         |
        | RGB       | Default      |
        | Depth     | Default - 1  |
        | Normal    | Default - 2  |
        | Semantics | Default - 3  |
        | UI        | Default - 4  |
        Default depends on how many Gibson environments are running simultanously
        '''
        self.port_rgb = self.DEFAULT_PORT - self.gpu_idx * 5
        self.port_depth = self.port_rgb - 1
        self.port_normal = self.port_rgb - 2
        self.port_sem  = self.port_rgb - 3
        self.port_ui   = self.port_rgb - 4

    def robot_introduce(self, robot):
        self.robot = robot
        self.robot.env = self
        self.action_space = self.robot.action_space
        ## Robot's eye observation, in sensor mode black pixels are returned
        self.observation_space = self.robot.observation_space
        self.sensor_space = self.robot.sensor_space
        # seed for robot
        self.robot.np_random = self.np_random
        self._robot_introduced = True
        assert (self.robot.resolution <= 512 and self.robot.resolution >= 64), \
            "Robot resolution must in [64, 512]"
        #if self.robot.resolution == 64:
        #    self.windowsz = 64
        #    self.scale_up = 4
        #elif self.robot.resolution == 128:
        #    self.windowsz = 128
        #    self.scale_up = 4
        #elif self.robot.resolution == 256:
        #    self.windowsz = 256
        #    self.scale_up = 2
        #else:
        #    self.windowsz = 512
        #    self.scale_up = 1

        self.windowsz = self.robot.resolution
        self.scale_up = int(512 / self.windowsz)

        if "fast_lq_render" in self.config and self.config["fast_lq_render"] == True:
            self.scale_up *= 2
        # if fast render, use lower quality point cloud

        self._render_width = self.windowsz
        self._render_height = self.windowsz

    def scene_introduce(self, gravity=9.8, collision_enabled=True):
        assert(self._robot_introduced)
        self.create_scene(gravity=gravity, collision_enabled=collision_enabled)
        self._scene_introduced = True

    def get_keys_to_action(self):
        return self.robot.keys_to_action

    def _reset(self):
        assert(self._robot_introduced)
        assert(self._scene_introduced)
        debugmode = 1
        if debugmode:
            print("Episode: steps:{} score:{}".format(self.nframe, self.reward))
            body_xyz = self.robot.body_xyz
            #print("[{}, {}, {}],".format(body_xyz[0], body_xyz[1], body_xyz[2]))
            print("Episode count: {}".format(self.eps_count))
            self.eps_count += 1
        self.nframe = 0
        self.eps_reward = 0
        BaseEnv._reset(self)

        if not self.ground_ids:
            self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(
                    [])
            self.ground_ids = set(self.scene.scene_obj_list)

        ## Todo: (hzyjerry) this part is not working, robot_tracking_id = -1
        for i in range (p.getNumBodies()):
            if (p.getBodyInfo(i)[0].decode() == self.robot_body.get_name()):
               self.robot_tracking_id=i
            #print(p.getBodyInfo(i)[0].decode())
        i = 0

        eye_pos, eye_quat = self.get_eye_pos_orientation()
        pose = [eye_pos, eye_quat]

        observations = self.render_observations(pose)
        pos = self.robot._get_scaled_position()
        orn = self.robot.get_orientation()

        pos = (pos[0], pos[1], pos[2] + self.tracking_camera['z_offset'])
        p.resetDebugVisualizerCamera(self.tracking_camera['distance'],self.tracking_camera['yaw'], self.tracking_camera['pitch'],pos)
        return observations

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

        self.rewards = self._rewards(a)
        done = self._termination()
        debugmode=0
        if (debugmode):
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))

        self.reward += sum(self.rewards)
        self.eps_reward += sum(self.rewards)

        debugmode = 0
        if debugmode:
            print("Eps frame {} reward {}".format(self.nframe, self.reward))
            print("position", self.robot.get_position())
        if self.gui:
            pos = self.robot._get_scaled_position()
            orn = self.robot.get_orientation()
            pos = (pos[0], pos[1], pos[2] + self.tracking_camera['z_offset'])
            pos = np.array(pos)
            dist = self.tracking_camera['distance'] / self.robot.mjcf_scaling
            p.resetDebugVisualizerCamera(dist, self.tracking_camera['yaw'], self.tracking_camera['pitch'], pos)

        eye_pos, eye_quat = self.get_eye_pos_orientation()
        pose = [eye_pos, eye_quat]
        observations = self.render_observations(pose)

        debugmode = 0
        if (debugmode):
            print("Camera env eye position", eye_pos)
            print("episode rewards", sum(self.rewards), "steps", self.nframe)

        episode = None
        if done:
            episode = {'r': self.reward,
                       'l': self.nframe}
            debugmode = 0
            if debugmode:
                print("return episode:", episode)
        return observations, sum(self.rewards), bool(done), dict(eye_pos=eye_pos, eye_quat=eye_quat, episode=episode)

    def _termination(self):
        raise NotImplementedError()

    def _reward(self, action):
        raise NotImplementedError()

    def calc_rewards(self, a, state):
        print("Please do not directly instantiate BaseRobotEnv")
        raise NotImplementedError()

    def get_eye_pos_orientation(self):
        """Used in CameraEnv.setup"""
        eye_pos = self.robot.eyes.get_position()
        x, y, z ,w = self.robot.eyes.get_orientation()
        eye_quat = [w, x, y, z]
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

    def find_best_k_views(self, eye_pos, all_dist, all_pos, avoid_block=False):
        least_order = (np.argsort(all_dist))
        top_k = []
        num_to_test = self.k * 2
        curr_num = 0
        all_pos = np.array(all_pos)

        if not avoid_block:
            return least_order[:self.k]

        while len(top_k) < self.k:
            curr_order = least_order[curr_num: curr_num + num_to_test]
            curr_pos =  all_pos[curr_order]
            print("Curr num", curr_num, "top k", len(top_k), self.k)
            if len(curr_pos) <= p.MAX_RAY_INTERSECTION_BATCH_SIZE:
                collisions = list(p.rayTestBatch([eye_pos] * len(curr_pos), curr_pos))
            else:
                collisions = []
                curr_i = 0
                while (curr_i < len(curr_pos)):
                    curr_n = min(len(curr_pos), curr_i + p.MAX_RAY_INTERSECTION_BATCH_SIZE - 1)
                    collisions = collisions + list(p.rayTestBatch([eye_pos] * (curr_n - curr_i), curr_pos[curr_i: curr_n]))
                    curr_i = curr_n
            has_collision = [c[0] > 0 for c in collisions]
            ## (hzyjerry): ray_casting-based view selection occasionally gives unstable behaviour. Will keep watching on this
            for i, x in enumerate(curr_order):
                if not has_collision[i]:
                    top_k.append(x)
        return top_k


    def getExtendedObservation(self):
        pass

    # Gym v0.10.5 compatibility
    reset = _reset
    step  = _step


class CameraRobotEnv(BaseRobotEnv):
    """CameraRobotEnv has full modalities. If it's initialized with mode="SENSOR",
    PC renderer is not initialized to save time.
    """
    multiprocessing = True
    def __init__(self, config, gpu_idx, scene_type, tracking_camera):
        ## The following properties are already instantiated inside xxx_env.py:
        BaseRobotEnv.__init__(self, config, tracking_camera, scene_type, gpu_idx)

        if self.gui:
            self.screen_arr = np.zeros([512, 512, 3])
        
        self.test_env = "TEST_ENV" in os.environ.keys() and os.environ['TEST_ENV'] == "True"
        self._use_filler = config["use_filler"]
        self._require_camera_input = 'rgb_filled' in self.config["output"] or \
                                     'rgb_prefilled' in self.config["output"] or \
                                     'depth' in self.config["output"] or \
                                     'normal' in self.config["output"] or \
                                     'semantics' in self.config["output"]
        self._require_rgb = 'rgb_filled' in self.config["output"] or "rgb_prefilled" in self.config["output"]
        self._require_depth = 'depth' in self.config["output"]
        self._require_normal = 'depth' in self.config["output"]
        self._require_semantics = 'semantics' in self.config["output"]
        self._semantic_source = 1
        self._semantic_color = 1
        if self._require_semantics:
            assert "semantic_source" in self.config.keys(), "semantic_source not specified in configuration"
            assert "semantic_color" in self.config.keys(), "semantic_color not specified in configuration"
            assert self.config["semantic_source"] in [1, 2], "semantic_source not valid"
            assert self.config["semantic_color"] in [1, 2, 3], "semantic_source not valid"
            self._semantic_source = self.config["semantic_source"]
            self._semantic_color  = self.config["semantic_color"]
        self._require_normal = 'normal' in self.config["output"]

        #if self._require_camera_input:
        self.model_path = get_model_path(self.model_id)

        self.save_frame  = 0
        self.fps = 0


    def reset_observations(self):
        ## Initialize blank render image
        self.render_rgb_filled = np.zeros((self.windowsz, self.windowsz, 3))
        self.render_rgb_prefilled = np.zeros((self.windowsz, self.windowsz, 3))
        self.render_depth = np.zeros((self.windowsz, self.windowsz, 1))
        self.render_normal = np.zeros((self.windowsz, self.windowsz, 3))
        self.render_semantics = np.zeros((self.windowsz, self.windowsz, 3))

    def robot_introduce(self, robot):
        BaseRobotEnv.robot_introduce(self, robot)
        self.setup_rendering_camera()

    def scene_introduce(self, gravity=9.8, collision_enabled=True):
        BaseRobotEnv.scene_introduce(self, gravity=gravity, collision_enabled=collision_enabled)

    def setup_rendering_camera(self):
        if self.test_env:
            return
        self.r_camera_rgb = None     ## Rendering engine
        self.r_camera_mul = None     ## Multi channel rendering engine
        self.r_camera_dep = None
        #self.check_port_available()

        ui_map = {
            1: OneViewUI,
            2: TwoViewUI,
            3: ThreeViewUI,
            4: FourViewUI,
        }

        assert self.config["ui_num"] == len(self.config['ui_components']), "In configuration, ui_num is not equal to the number of ui components"
        if self.config["display_ui"]:
            ui_num = self.config["ui_num"]
            self.UI = ui_map[ui_num](self.windowsz, self, self.port_ui)

        if self._require_camera_input:
            self.setup_camera_multi()
            self.setup_camera_pc()

        if self.config["mode"] == "web_ui":
            ui_num = self.config["ui_num"]
            self.webUI = ui_map[ui_num](self.windowsz, self, self.port_ui, use_pygame=False)

    def _reset(self):
        self.reset_observations()
        sensor_state = BaseRobotEnv._reset(self)
        self.potential = self.robot.calc_potential()
        eye_pos, eye_quat = self.get_eye_pos_orientation()
        pose = [eye_pos, eye_quat]
        
        observations = self.render_observations(pose)
        return observations #, sensor_state


    def add_text(self, img):
        return img

    def _step(self, a):
        t = time.time()
        base_obs, sensor_reward, done, sensor_meta = BaseRobotEnv._step(self, a)
        dt = time.time() - t
        # Speed bottleneck
        observations = base_obs
        self.fps = 0.9 * self.fps + 0.1 * 1/dt

        pose = [sensor_meta['eye_pos'], sensor_meta['eye_quat']]
        sensor_meta.pop("eye_pos", None)
        sensor_meta.pop("eye_quat", None)
        #sensor_meta["sensor"] = sensor_state

        if self.gui:
            if self.config["display_ui"]:
                self.render_to_UI()
                self.save_frame += 1
            elif self._require_camera_input:
                # Use non-pygame GUI
                self.r_camera_rgb.renderToScreen()

        if self.config["mode"] == 'web_ui':
            self.render_to_webUI()

        if not self._require_camera_input or self.test_env:
            ## No camera input (rgb/depth/normal/semantics)
            return base_obs, sensor_reward, done, sensor_meta
        else:
            if self.config["show_diagnostics"] and self._require_rgb:
                self.render_rgb_filled = self.add_text(self.render_rgb_filled)

            robot_pos = self.robot.get_position()
            debugmode = 0
            if debugmode:
                print("Eye position", sensor_meta['eye_pos'])
            debugmode = 0
            if debugmode:
                print("Environment observation keys", observations.keys)
            return observations, sensor_reward, done, sensor_meta


    def render_component(self, tag):
        if tag == View.RGB_FILLED:
            return self.render_rgb_filled
        if tag == View.RGB_PREFILLED:
            return self.render_rgb_prefilled
        if tag == View.DEPTH:
            scaled_depth = self.render_depth.copy()
            scaled_depth = scaled_depth * DEPTH_SCALE_FACTOR + DEPTH_OFFSET_FACTOR
            overflow = scaled_depth > 255.
            scaled_depth[overflow] = 255.
            return scaled_depth
        if tag == View.NORMAL:
            return self.render_normal
        if tag == View.SEMANTICS:
            print("Render components: semantics", np.mean(self.render_semantics))
            return self.render_semantics
        if tag == View.PHYSICS:
            return self.render_physics()
        if tag == View.MAP:
            return self.render_map()

    def render_to_UI(self):
        '''Works for different UI: UI_SIX, UI_FOUR, UI_TWO
        '''
        if not self.config["display_ui"]:
            return

        self.UI.refresh()

        for component in self.UI.components:
            self.UI.update_view(self.render_component(component), component)

    def render_to_webUI(self):
        '''Works for different UI: UI_SIX, UI_FOUR, UI_TWO
        '''
        if not self.config["mode"] == 'web_ui':
            return

        self.webUI.refresh()

        for component in self.webUI.components:
            self.webUI.update_view(self.render_component(component), component)



    def _close(self):
        BaseEnv._close(self)

        if self._require_camera_input:
            self.r_camera_mul.terminate()
            self.r_camera_rgb._close()

            if self.r_camera_dep is not None:
                self.r_camera_dep.terminate()
            if self._require_normal:
                self.r_camera_norm.terminate()
            if self._require_semantics:
                self.r_camera_semt.terminate()

        if self.config["display_ui"]:
            self.UI._close()

    def get_key_pressed(self, relevant=None):
        pressed_keys = []
        events = p.getKeyboardEvents()
        key_codes = events.keys()
        for key in key_codes:
            pressed_keys.append(key)
        return pressed_keys

    def get_blank_visuals(self):
        return np.zeros((256, 256, 4))

    def render_observations(self, pose):
        '''Render all environment observations, called inside every step()
        Input
            @pose: current robot pose
        Return:
            @observation: dict with key values being output components user
            specified in config file
        TODO:
            @hzyjerry: add noise to observation
        '''

        self.render_nonviz_sensor = self.robot.calc_state()

        if self._require_camera_input:
            self.r_camera_rgb.setNewPose(pose)
            all_dist, all_pos = self.r_camera_rgb.getAllPoseDist(pose)
            top_k = self.find_best_k_views(pose[0], all_dist, all_pos, avoid_block=False)
            #with Profiler("Render to screen"):
            self.render_rgb_filled, self.render_depth, self.render_semantics, self.render_normal, self.render_rgb_prefilled = self.r_camera_rgb.renderOffScreen(pose, top_k, rgb=self._require_rgb)

        observations = {}
        for output in self.config["output"]:
            try:
                observations[output] = getattr(self, "render_" + output)
            except Exception as e:
                raise Exception("Output component {} is not available".format(output))

        #visuals = np.concatenate(visuals, 2)
        return observations

    def get_observations(self):
        observations = {}
        for output in self.config["output"]:
            try:
                observations[output] = getattr(self, "render_" + output)
            except Exception as e:
                raise Exception("Output component {} is not available".format(output))
        
        #visuals = np.concatenate(visuals, 2)
        return observations

    def setup_camera_pc(self):
        ## Camera specific
        assert(self._require_camera_input)
        if self.scene_type == "building":
            self.dataset = ViewDataSet3D(
                transform = np.array,
                mist_transform = np.array,
                seqlen = 2,
                off_3d = False,
                train = False,
                overwrite_fofn=True, env = self, only_load = self.config["model_id"])

        scene_dict = dict(zip(self.dataset.scenes, range(len(self.dataset.scenes))))
        ## Todo: (hzyjerry) more error handling
        if not self.model_id in scene_dict.keys():
             raise error.Error("Dataset not found: model {} cannot be loaded".format(self.model_id))
        else:
            scene_id = scene_dict[self.model_id]
        uuids, rts = self.dataset.get_scene_info(scene_id)

        targets, sources, source_depths, poses = [], [], [], []
        source_semantics = []

        if not self.multiprocessing or self.config["envname"] == "TestEnv":
            all_data = self.dataset.get_multi_index([v for k, v in uuids])
            for i, data in enumerate(all_data):
                target, target_depth = data[1], data[3]
                if not self._require_rgb:
                    continue
                ww = target.shape[0] // 8 + 2
                target[:ww, :, :] = target[ww, :, :]
                target[-ww:, :, :] = target[-ww, :, :]

                if self.scale_up !=1:
                    target = cv2.resize(
                        target,None,
                        fx=1.0/self.scale_up,
                        fy=1.0/self.scale_up,
                        interpolation = cv2.INTER_CUBIC)
                    target_depth =  cv2.resize(
                        target_depth, None,
                        fx=1.0/self.scale_up,
                        fy=1.0/self.scale_up,
                        interpolation = cv2.INTER_CUBIC)
                pose = data[-1][0].numpy()
                targets.append(target)
                poses.append(pose)
                sources.append(target)
                source_depths.append(target_depth)
        else:
            all_data = self.dataset.get_multi_index([v for k, v in uuids])
            for i, data in enumerate(all_data):
                target, target_depth = data[1], data[3]
                if not self._require_rgb:
                    continue
                ww = target.shape[0] // 8 + 2
                target[:ww, :, :] = target[ww, :, :]
                target[-ww:, :, :] = target[-ww, :, :]

                if self.scale_up !=1:

                    target = cv2.resize(
                        target,None,
                        fx=1.0/self.scale_up,
                        fy=1.0/self.scale_up,
                        interpolation = cv2.INTER_CUBIC)
                    target_depth =  cv2.resize(
                        target_depth, None,
                        fx=1.0/self.scale_up,
                        fy=1.0/self.scale_up,
                        interpolation = cv2.INTER_CUBIC)
                pose = data[-1][0].numpy()
                targets.append(target)
                poses.append(pose)
                sources.append(target)
                source_depths.append(target_depth) 
        
        self.r_camera_rgb = PCRenderer(self.port_rgb, sources, source_depths, target, rts, 
                                       scale_up=self.scale_up, 
                                       semantics=source_semantics,
                                       gui=self.gui, 
                                       use_filler=self._use_filler,  
                                       gpu_idx=self.gpu_idx,
                                       windowsz=self.windowsz, 
                                       env = self)

    def setup_camera_multi(self):
        assert(self._require_camera_input)
        def camera_multi_excepthook(exctype, value, tb):
            print("killing", self.r_camera_mul)
            self.r_camera_mul.terminate()
            if self.r_camera_dep is not None:
                self.r_camera_dep.terminate()
            if self._require_normal:
                self.r_camera_norm.terminate()
            if self._require_semantics:
                self.r_camera_semt.terminate()
            while tb:
                if exctype == KeyboardInterrupt:
                    print("Exiting Gibson...")
                    return
                filename = tb.tb_frame.f_code.co_filename
                name = tb.tb_frame.f_code.co_name
                lineno = tb.tb_lineno
                print('   File "%.500s", line %d, in %.500s' %(filename, lineno, name))
                tb = tb.tb_next
            print(' %s: %s' %(exctype.__name__, value))

        sys.excepthook = camera_multi_excepthook
        enable_render_smooth = 0

        dr_path = osp.join(osp.dirname(osp.abspath(gibson.__file__)), 'core', 'channels', 'depth_render')
        cur_path = os.getcwd()
        os.chdir(dr_path)

        render_main  = "./depth_render --GPU {} --modelpath {} -w {} -h {} -f {} -p {}".format(self.gpu_idx, self.model_path, self.windowsz, self.windowsz, self.config["fov"]/np.pi*180, self.port_depth)
        render_norm  = "./depth_render --GPU {} --modelpath {} -n 1 -w {} -h {} -f {} -p {}".format(self.gpu_idx, self.model_path, self.windowsz, self.windowsz, self.config["fov"]/np.pi*180, self.port_normal)
        render_semt  = "./depth_render --GPU {} --modelpath {} -t 1 -r {} -c {} -w {} -h {} -f {} -p {}".format(self.gpu_idx, self.model_path, self._semantic_source, self._semantic_color, self.windowsz, self.windowsz, self.config["fov"]/np.pi*180, self.port_sem)
        
        self.r_camera_mul = subprocess.Popen(shlex.split(render_main), shell=False)
        #self.r_camera_dep = subprocess.Popen(shlex.split(render_depth), shell=False)
        if self._require_normal:
            self.r_camera_norm = subprocess.Popen(shlex.split(render_norm), shell=False)
        if self._require_semantics:
            self.r_camera_semt = subprocess.Popen(shlex.split(render_semt), shell=False)

        os.chdir(cur_path)


    def check_port_available(self):
        assert(self._require_camera_input)
        # TODO (hzyjerry)
        ports = []
        if self._require_depth: ports.append(self.port_depth)
        if self._require_normal: ports.append(self.port_normal)
        if self._require_semantics: ports.append(self.port_sem)
        for port in ports:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                result = s.bind(("127.0.0.1", port - 1))
            except socket.error as e:
                raise e
                raise error.Error("Gibson initialization Error: port {} is in use".format(port))
    # Gym v0.10.5 compatibility
    reset = _reset
    step  = _step
    close = _close


class SemanticRobotEnv(CameraRobotEnv):
    def __init__(self, config, gpu_idx, scene_type, tracking_camera):
        CameraRobotEnv.__init__(self, config, gpu_idx, scene_type, tracking_camera)

    def robot_introduce(self, robot):
        CameraRobotEnv.robot_introduce(self, robot)
        self.setup_semantic_parser()

    def setup_semantic_parser(self):
        #assert('semantics' in self.config["output"])
        def semantic_excepthook(exctype, value, tb):
            print("killing", self.r_camera_mul)
            self.r_camera_mul.terminate()
            if self.r_camera_dep:
                self.r_camera_dep.terminate()
            if self._require_normal:
                self.r_camera_norm.terminate()
            if self._require_semantics:
                self.r_camera_semt.terminate()
            while tb:
                if exctype == KeyboardInterrupt:
                    print("Exiting Gibson...")
                    return
                filename = tb.tb_frame.f_code.co_filename
                name = tb.tb_frame.f_code.co_name
                lineno = tb.tb_lineno
                print('   File "%.500s", line %d, in %.500s' %(filename, lineno, name))
                tb = tb.tb_next
            print(' %s: %s' %(exctype.__name__, value))

        sys.excepthook = semantic_excepthook
        dr_path = osp.join(osp.dirname(osp.abspath(gibson.__file__)), 'core', 'channels', 'depth_render')
        cur_path = os.getcwd()
        os.chdir(dr_path)
        load_semantic  = "./semantic --modelpath {} -r {} ".format(self.model_path, self._semantic_source)
        self.semantic_server = subprocess.Popen(shlex.split(load_semantic), shell=False)
        os.chdir(cur_path)

        self._context_sem = zmq.Context()
        self.semantic_client = self._context_sem.socket(zmq.REQ)
        self.semantic_client.connect("tcp://localhost:{}".format(5055))

        self.semantic_client.send_string("Ready")
        semantic_msg = self.semantic_client.recv()
        self.semantic_pos = np.frombuffer(semantic_msg, dtype=np.float32).reshape((-1, 3))

        if self._semantic_source == 2:
            _, semantic_ids, _ = get_segmentId_by_name_MP3D(osp.join(self.model_path, "semantic.house"), "chair")
        elif self._semantic_source == 1:
            _, semantic_ids, _ = get_segmentId_by_name_2D3DS(osp.join(self.model_path, "semantic.mtl"), osp.join(self.model_path, "semantic.obj"), "chair")

        self.semantic_pos = self.semantic_pos[semantic_ids, :]

        debugmode=0
        if debugmode:
            self.semantic_pos = np.array([[0, 0, 0.2]])
    
    def dist_to_semantic_pos(self):
        pos = self.robot.get_position()
        x, y, z, w = self.robot.get_orientation()
        #print(self.semantic_pos)
        #print(pos, orn)

        diff_pos = self.semantic_pos - pos
        dist_to_robot = np.sqrt(np.sum(diff_pos * diff_pos, axis = 1))
        diff_unit = (diff_pos.T / dist_to_robot).T

        #TODO: (hzyjerry) orientation is still buggy
        orn_unit = quat2mat([w, x, y, z]).dot(np.array([-1, 0, 0]))
        orn_to_robot = np.arccos(diff_unit.dot(orn_unit))
        return dist_to_robot, orn_to_robot

    def get_close_semantic_pos(self, dist_max=1.0, orn_max=np.pi/5):
        '''Find the index of semantic positions close to the agent, within max
        distance and max orientation
        Return: list of index of the semantic positions, corresponding the index
            in self.semantic_pos
        '''
        dists, orns = self.dist_to_semantic_pos()
        return [i for i in range(self.semantic_pos.shape[0]) if dists[i] < dist_max and orns[i] < orn_max]

    def step(self, action, tag=True):
        #self.close_semantic_ids = self.get_close_semantic_pos()
        return CameraRobotEnv.step(self, action)


STR_TO_PYGAME_KEY = {
    'a': pygame.K_a,
    'b': pygame.K_b,
    'c': pygame.K_c,
    'd': pygame.K_d,
    'e': pygame.K_e,
    'f': pygame.K_f,
    'g': pygame.K_g,
    'h': pygame.K_h,
    'i': pygame.K_i,
    'j': pygame.K_j,
    'k': pygame.K_k,
    'l': pygame.K_l,
    'm': pygame.K_m,
    'n': pygame.K_n,
    'o': pygame.K_o,
    'p': pygame.K_p,
    'q': pygame.K_q,
    'r': pygame.K_r,
    's': pygame.K_s,
    't': pygame.K_t,
    'u': pygame.K_u,
    'v': pygame.K_v,
    'w': pygame.K_w,
    'x': pygame.K_x,
    'y': pygame.K_y,
    'z': pygame.K_z,
}
