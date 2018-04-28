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


DEFAULT_TIMESTEP  = 1.0/(4 * 9)
DEFAULT_FRAMESKIP = 4
DEFAULT_DEBUG_CAMERA = {
    'yaw': 30,
    'distance': 2.5,
    'pitch': -35,
    'z_offset': 0
}

DEPTH_SCALE_FACTOR = 15

class BaseRobotEnv(BaseEnv):
    """Based on BaseEnv
    Handles action, reward
    """

    def __init__(self, config, tracking_camera, scene_type="building", gpu_count=0):
        BaseEnv.__init__(self, config, scene_type, tracking_camera)

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
            overwrite_fofn=True, env = self)
        self.ground_ids = None
        if self.gui:
            assert(self.tracking_camera is not None)
        self.gpu_count = gpu_count
        self.nframe = 0
        self.eps_reward = 0

        self.reward = 0
        self.eps_count = 0

        self._robot_introduced = False
        self._scene_introduced = False

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
        assert (self.robot.resolution in [64,128,256,512]), \
            "Robot resolution must be in 64/128/256/512"
        if self.robot.resolution == 64:
            self.windowsz = 64
            self.scale_up = 4
        elif self.robot.resolution == 128:
            self.windowsz = 128
            self.scale_up = 4
        elif self.robot.resolution == 256:
            self.windowsz = 256
            self.scale_up = 2
        else:
            self.windowsz = 512
            self.scale_up = 1

        self._render_width = self.windowsz
        self._render_height = self.windowsz

    def scene_introduce(self):
        assert(self._robot_introduced)
        self.create_scene()
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
            print("[{}, {}, {}],".format(body_xyz[0], body_xyz[1], body_xyz[2]))
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
            p.resetDebugVisualizerCamera(self.tracking_camera['distance'],self.tracking_camera['yaw'], self.tracking_camera['pitch'],pos)

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


class CameraRobotEnv(BaseRobotEnv):
    """CameraRobotEnv has full modalities. If it's initialized with mode="SENSOR",
    PC renderer is not initialized to save time.
    """
    multiprocessing = True
    def __init__(self, config, gpu_count, scene_type, tracking_camera):
        ## The following properties are already instantiated inside xxx_env.py:
        BaseRobotEnv.__init__(self, config, tracking_camera, scene_type, gpu_count)

        if self.gui:
            self.screen_arr = np.zeros([612, 512, 3])
        
        self.test_env = "TEST_ENV" in os.environ.keys() and os.environ['TEST_ENV'] == "True"
        self._use_filler = config["use_filler"]
        self._require_camera_input = 'rgb_filled' in self.config["output"] or \
                                     'rgb_prefilled' in self.config["output"] or \
                                     'depth' in self.config["output"] or \
                                     'normal' in self.config["output"] or \
                                     'semantics' in self.config["output"]
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


    def robot_introduce(self, robot):
        BaseRobotEnv.robot_introduce(self, robot)
        self.setup_rendering_camera()

    def scene_introduce(self):
        BaseRobotEnv.scene_introduce(self)

    def setup_rendering_camera(self):
        if self.test_env or not self._require_camera_input:
            return
        self.r_camera_rgb = None     ## Rendering engine
        self.r_camera_mul = None     ## Multi channel rendering engine
        self.r_camera_dep = None
        self.check_port_available()
        self.setup_camera_multi()
        self.setup_camera_rgb()
        
        ui_map = {
            1: OneViewUI,
            2: TwoViewUI,
            3: ThreeViewUI,
            4: FourViewUI,
        }

        assert self.config["ui_num"] == len(self.config['ui_components']), "In configuration, ui_num is not equal to the number of ui components"
        if self.config["display_ui"]:
            self.UI = ui_map[self.config["ui_num"]](self.windowsz, self)


    def _reset(self):
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

        if not self._require_camera_input or self.test_env:
            return base_obs, sensor_reward, done, sensor_meta

        if self.config["show_diagnostics"]:
            self.render_rgb_filled = self.add_text(self.render_rgb_filled)

        if self.config["display_ui"]:
            self.render_to_UI()
            self.save_frame += 1

        elif self.gui:
            # Speed bottleneck 2, 116fps
            self.r_camera_rgb.renderToScreen()

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
            scaled_depth = self.render_depth * DEPTH_SCALE_FACTOR
            return scaled_depth
        if tag == View.NORMAL:
            return self.render_normal
        if tag == View.SEMANTICS:
            print("Render components: semantics", np.mean(self.render_semantics))
            return self.render_semantics

    def render_to_UI(self):
        '''Works for different UI: UI_SIX, UI_FOUR, UI_TWO
        '''
        if not self.config["display_ui"]:
            return

        self.UI.refresh()

        for component in self.UI.components:
            self.UI.update_view(self.render_component(component), component)


    def _close(self):
        BaseEnv._close(self)

        if not self._require_camera_input or self.test_env:
            return
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
            all_dist, all_pos = self.r_camera_rgb.rankPosesByDistance(pose)
            top_k = self.find_best_k_views(pose[0], all_dist, all_pos)
            self.render_rgb_filled, self.render_depth, self.render_semantics, self.render_normal, self.render_prefilled = self.r_camera_rgb.renderOffScreen(pose, top_k)

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

    def setup_camera_rgb(self):
        ## Camera specific
        assert(self._require_camera_input)
        scene_dict = dict(zip(self.dataset.scenes, range(len(self.dataset.scenes))))
        ## Todo: (hzyjerry) more error handling
        if not self.model_id in scene_dict.keys():
             raise error.Error("Dataset not found: model {} cannot be loaded".format(self.model_id))
        else:
            scene_id = scene_dict[self.model_id]
        uuids, rts = self.dataset.get_scene_info(scene_id)

        targets, sources, source_depths, poses = [], [], [], []
        source_semantics = []

        if not self.multiprocessing:
            for k,v in tqdm((uuids)):
                data = self.dataset[v]
                target, target_depth = data[1], data[3]
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
        
        ## TODO (hzyjerry): make sure 5555&5556 are not occupied, or use configurable ports
        self.r_camera_rgb = PCRenderer(5556, sources, source_depths, target, rts, self.scale_up, 
                                       semantics=source_semantics,
                                       gui=self.gui, 
                                       use_filler=self._use_filler,  
                                       gpu_count=self.gpu_count, 
                                       windowsz=self.windowsz, 
                                       env = self)
        
        ## Initialize blank render image
        self.render_rgb_filled = np.zeros((self.windowsz, self.windowsz, 3))
        self.render_rgb_prefilled = np.zeros((self.windowsz, self.windowsz, 3))


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

        render_main  = "./depth_render --modelpath {} --GPU {} -w {} -h {} -f {}".format(self.model_path, self.gpu_count, self.windowsz, self.windowsz, self.config["fov"]/np.pi*180)
        render_depth = "./depth_render --modelpath {} --GPU -1 -s {} -w {} -h {} -f {}".format(self.model_path, enable_render_smooth ,self.windowsz, self.windowsz, self.config["fov"]/np.pi*180)
        render_norm  = "./depth_render --modelpath {} -n 1 -w {} -h {} -f {}".format(self.model_path, self.windowsz, self.windowsz, self.config["fov"]/np.pi*180)
        render_semt  = "./depth_render --modelpath {} -t 1 -r {} -c {} -w {} -h {} -f {}".format(self.model_path, self._semantic_source, self._semantic_color, self.windowsz, self.windowsz, self.config["fov"]/np.pi*180)
        
        self.r_camera_mul = subprocess.Popen(shlex.split(render_main), shell=False)
        self.r_camera_dep = subprocess.Popen(shlex.split(render_depth), shell=False)
        if self._require_normal:
            self.r_camera_norm = subprocess.Popen(shlex.split(render_norm), shell=False)
        if self._require_semantics:
            self.r_camera_semt = subprocess.Popen(shlex.split(render_semt), shell=False)

        os.chdir(cur_path)

        ## Set up blank render images
        self.render_depth = np.zeros((self.windowsz, self.windowsz, 1))
        self.render_normal = np.zeros((self.windowsz, self.windowsz, 3))
        self.render_semantics = np.zeros((self.windowsz, self.windowsz, 3))


    def check_port_available(self):
        assert(self._require_camera_input)
        # TODO (hzyjerry)
        """
        s = socket.socket()
        try:
            s.connect(("127.0.0.1", 5555))
        except socket.error as e:
            raise e
            raise error.Error("gibson starting error: port {} is in use".format(5555))
        try:
            s.connect(("127.0.0.1", 5556))
        except socket.error as e:
            raise error.Error("gibson starting error: port {} is in use".format(5556))
        """
        return



class SemanticRobotEnv(CameraRobotEnv):
    def __init__(self, config, gpu_count, scene_type, tracking_camera):
        CameraRobotEnv.__init__(self, config, gpu_count, scene_type, tracking_camera)

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
                filename = tb.tb_frame.f_code.co_filename
                name = tb.tb_frame.f_code.co_name
                lineno = tb.tb_lineno
                print('   File "%.500s", line %d, in %.500s' %(filename, lineno, name))
                tb = tb.tb_next
            print(' %s: %s' %(exctype.__name__, value))

        #sys.excepthook = semantic_excepthook
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
