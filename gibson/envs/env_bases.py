## Issue related to time resolution/smoothness
#  http://bulletphysics.org/mediawiki-1.5.8/index.php/Stepping_The_World

from gibson.core.physics.scene_building import SinglePlayerBuildingScene
from gibson.core.physics.scene_stadium import SinglePlayerStadiumScene
import pybullet as p
import time
import random
import zmq
import math
import argparse
import os
import json
import numpy as np
from transforms3d import euler, quaternions
from gibson.core.physics.physics_object import PhysicsObject
from gibson.core.render.profiler import Profiler
import gym, gym.spaces, gym.utils, gym.utils.seeding
import sys
import yaml

class BaseEnv(gym.Env):
    """
    Base class for loading environments in a Scene.
    Handles scene loading, robot loading, pybullet client setting,
        camera setting

    These environments create single-player scenes and behave like normal Gym environments.
    Multiplayer is not yet supported
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
        }

    def __init__(self, config, scene_type):
        ## Properties already instantiated from SensorEnv/CameraEnv
        #   @self.human
        #   @self.robot

        if self.config is None:
            self.config = self.parse_config(config)


        if self.config["display_ui"]:
            #self.physicsClientId = p.connect(p.DIRECT)
            self.physicsClientId = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        elif (self.human):
            self.physicsClientId = p.connect(p.GUI)
        else:
            self.physicsClientId = p.connect(p.DIRECT)

        self.camera = Camera()
        self._seed()
        self._cam_dist = 3
        self._cam_yaw = 0
        self._cam_pitch = -30
        self.scene_type = scene_type
        self.scene = None

    
    def _close(self):
        p.disconnect()


    def parse_config(self, config):
        with open(config, 'r') as f:
            config_data = yaml.load(f)
        return config_data
        
    def create_scene(self):
        if self.scene is not None:
            return
        if self.scene_type == "stadium":
            self.scene = self.create_single_player_stadium_scene()
        elif self.scene_type == "building":
            self.scene = self.create_single_player_building_scene()
        else:
            raise AssertionError()
        self.robot.scene = self.scene
    
    def create_single_player_building_scene(self):
        return SinglePlayerBuildingScene(self.robot, model_id=self.model_id, gravity=9.8, timestep=self.timestep, frame_skip=self.frame_skip, env=self)
        
    def create_single_player_stadium_scene(self):
        return SinglePlayerStadiumScene(self.robot, gravity=9.8, timestep=self.timestep, frame_skip=self.frame_skip)


    def configure(self, args):
        self.robot.args = args
    
    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _reset(self):
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    
        #visualid = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'),
        #                               meshScale=[0.3, 0.3, 0.3], rgbaColor=[1, 0, 0, 0.7])
        #physicsid = p.createMultiBody(baseVisualShapeIndex=visualid, baseCollisionShapeIndex=-1, basePosition=[0, 0, 2])
        #keep code here for reference

        self.frame = 0
        self.done = 0
        self.reward = 0
        dump = 0
        ## TODO(hzyjerry): toggle for hard_reset: reload robot
        state = self.robot.reset()
        self.scene.episode_restart()
        return state

    def _render(self, mode, close):
        base_pos=[0,0,0]
        if (hasattr(self,'robot')):
            if (hasattr(self.robot,'body_xyz')):
                base_pos = self.robot.body_xyz
        
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(self._render_width)/self._render_height,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(
        width=self._render_width, height=self._render_height, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def render_physics(self):
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_tracking_id)
        
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=robot_pos,
            distance=self.tracking_camera["distance"],
            yaw=self.tracking_camera["yaw"],
            pitch=self.tracking_camera["pitch"],
            roll=0,
            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(self._render_width)/self._render_height,
            nearVal=0.1, farVal=100.0)
        with Profiler("render physics: Get camera image"):
            (_, _, px, _, _) = p.getCameraImage(
            width=self._render_width, height=self._render_height, viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_TINY_RENDERER
                )
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array


    def render_map(self):
        base_pos=[0, 0, -3]
        if (hasattr(self,'robot')):
            if (hasattr(self.robot,'body_xyz')):
                base_pos[0] = self.robot.body_xyz[0]
                base_pos[1] = self.robot.body_xyz[1]
        
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=35,
            yaw=0,
            pitch=-89,
            roll=0,
            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(self._render_width)/self._render_height,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(
        width=self._render_width, height=self._render_height, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def get_action_dim(self):
        return len(self.robot.ordered_joints)

    def get_observation_dim(self):
        return 1

    def _close(self):
        if (self.physicsClientId>=0):
            p.disconnect(self.physicsClientId)
            self.physicsClientId = -1
    
    def set_window(self, posX, posY, sizeX, sizeY):
        values = {      
            'name': "Robot",  
            'gravity': 0,
            'posX': int(posX),
            'posY': int(posY),
            'sizeX': int(sizeX),
            'sizeY': int(sizeY)
        }
        cmd = 'wmctrl -r \"Bullet Physics\" -e {gravity},{posX},{posY},{sizeX},{sizeY}'.format(**values)
        os.system(cmd)

        cmd = "xdotool search --name \"Bullet Physics\" set_window --name \"Robot's world\""
        os.system(cmd)

    def HUD(self, state, a, done):
        pass


class Camera:
    def __init__(self):
        pass

    def move_and_look_at(self,i,j,k,x,y,z):
        lookat = [x,y,z]
        distance = 10
        yaw = 10

